//! Conceptually endless stream backed by a sequence of block streams.
//!
//! It builds on [`block::Stream`] with dynamic allocation and release of
//! [`Blocks`] via an allocator implementing [`BlocksAllocator`] trait.
//!
//! [`Stream`] is the primary type, refer to the documentation for details.
//! Other items are revolving around that type. The stream can have `Blocks` of
//! varying sizes backing up each block stream - it's OK to change the
//! configuration and still have access to previously written block streams,
//! as long as the allocator is aware of that.

use core::sync::atomic;
use std::{
    collections::VecDeque,
    io,
    sync::{Arc, Mutex},
};

use crate::{block, Blocks};

/// An allocator of [`Blocks`]. It is responsible for keeping track of which
/// blocks were allocated and storing this state, so that the allocated blocks
/// can be retrieved after a restart.
///
/// # Safety
///
/// Implementation must ensure that each caller of this trait has a unique
/// allocator, so that no two callers share the same underlying blocks.
///
/// Concurrent calls may or may not be guaranteed by the implementation, in the
/// latter case calls to functions that use the allocator must be synchronized.
///
/// The value of a type implementing this trait must outlive the returned
/// values. The callers must ensure that there can be only one copy of each
/// unique value returned by the allocator that is referenced during usage.
pub unsafe trait BlocksAllocator {
    /// The type of `Blocks` returned by this allocator.
    type Blocks: Blocks;

    /// Allocates new `Blocks`. The storage backing up the returned blocks must
    /// outlive the caller which uses them. Every allocation returns unique
    /// `Blocks`, such that no two allocators can reference the same underlying
    /// part of the storage.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    fn alloc(&self) -> io::Result<Self::Blocks>;

    /// Releases `Blocks` previously allocated or retrieved. On success, this
    /// function makes other copies of the same blocks referencing the same
    /// underlying storage invalid.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered. The original value
    /// is also returned back to the caller for future retries.
    fn release(&self, blocks: Self::Blocks) -> Result<(), (Self::Blocks, io::Error)>;

    /// Retrieves the allocated `Blocks` in the order they were allocated.
    /// This function can pass new copies that point to the same underlying
    /// storage to `f` via repeated calls, and the caller must ensure that
    /// only a single copy is in use.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    fn retrieve(&self, f: impl FnMut(Self::Blocks)) -> io::Result<()>;
}

/// An endless persistent data stream, represented as lazily allocated
/// [`block::Stream`] sequence, where every append is on the right-most block
/// stream.
///
/// The reads are wait-free, and only one append is allowed at a time. It
/// guarantees that dirty data is never read.
///
/// The stream has an internal offset, which points to the start of relevant
/// data within the stream. The offset is set during initialization and then
/// updated during runtime when needed. It is up to the caller to recover
/// the offset after restarts, as this heavily depends on the usage patterns.
/// Block streams that are no longer reachable as the offset has been advanced
/// far enough can be released.
///
/// The only interface to access the data is through [`Stream::iter`]. See the
/// function and the relevant type documentation for details.
///
/// Appends that do not fit into the current block stream are handled according
/// to [`SpanBehavior`]. The size of a single append to a stream is limited
/// by a single block stream capacity, or by the limit imposed by the current
/// block stream implementation - see [`block::Stream::append_with_opts`].
///
/// Allocation and release logic should live upstream. Allocations and
/// de-allocations of the underlying streams are generally non-blocking in
/// relation to readers and writers, except for a few rare cases. This
/// functionality is designed to be "slow", prioritizing readers and writers,
/// as it is expected to be infrequent and should be done in advance. The
/// release is two-step: first, the underlying block streams are removed via
/// [`Stream::maybe_shrink`], and then the memory and actual blocks are released
/// via [`Stream::try_release`] or [`Stream::force_release`]. The reason for
/// this, is that there could be readers that may still reference the data, even
/// after removing the block stream.
#[doc(alias = "endlessstream")]
#[derive(Debug)]
pub struct Stream<A: BlocksAllocator> {
    /// The offset in bytes within this stream, including the size of the data
    /// from removed blocks. This means that the value is always incrementing
    /// and never goes backwards. This state is not persistent and must be
    /// recovered during `initialize`.
    offset: atomic::AtomicUsize,

    /// A collection of block streams, where new ones are allocated as needed.
    /// Due to infrequent allocations and de-allocations, wrapping it in
    /// read-optimized lock, cloning the current value during updates.
    /// Only 2 versions are maintained, which should be plenty. Note, that
    /// `force_release` relies on this vlock to have 2 versions maximum.
    streams: vlock::VLock<BlockStreams<A::Blocks>, 2>,

    /// Items that were removed from `streams` and waiting to be released
    /// back to `allocator`. These live behind a simple mutex due to
    /// infrequent access and to avoid keeping copies of `Arc` around.
    /// The values are unique, in that each is pointing to a unique address
    /// or is a unique instance of Blocks.
    releasables: Mutex<Vec<Option<Releasable<A::Blocks>>>>,

    /// The underlying allocator of `Blocks`.
    allocator: A,

    /// How to handle appends which span multiple block streams.
    span_behavior: SpanBehavior,
}

impl<A: BlocksAllocator> Stream<A> {
    /// Creates a new empty stream with [`SpanBehavior::Never`]. To make it
    /// usable, see [`Stream::grow`] and [`Stream::load`].
    #[inline(always)]
    #[must_use]
    pub fn new(allocator: A) -> Self {
        Self::with_span_behavior(allocator, SpanBehavior::Never)
    }

    /// Creates a new empty stream with a given `span_behavior`. To make it
    /// usable, see [`Stream::grow`] and [`Stream::load`].
    #[inline(always)]
    #[must_use]
    pub fn with_span_behavior(allocator: A, span_behavior: SpanBehavior) -> Self {
        Self {
            offset: 0.into(),
            streams: BlockStreams::new().into(),
            // Pre-allocate with some arbitrary capacity, which seems to be
            // sufficient and unlikely to be exceeded.
            releasables: Vec::with_capacity(128).into(),
            allocator,
            span_behavior,
        }
    }

    /// Clears underlying memory and loads this stream from the allocator
    /// state. The previously used blocks are discarded and not released.
    /// On success, returns stats about this stream, should it be needed for
    /// allocating more blocks.
    ///
    /// The offset is reset to the start. To set the offset again, refer to
    /// [`Stream::set_offset`] or [`Stream::advance`].
    ///
    /// This function does not allocate blocks by itself. Instead, use
    /// [`Stream::grow`] to grow the stream.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered. If the error is
    /// caused by this implementation, rather than the underlying `Blocks`,
    /// the error will be of [`io::ErrorKind::Other`] kind and the value set
    /// to one of:
    ///
    /// -   [`StreamError::BlockStreamError`] as returned during
    ///     [`block::Stream::initialize`].
    /// -   [`StreamError::BrokenSpan`] if block stream trailing and spilled
    ///     sections do not match.
    pub fn load(&mut self) -> io::Result<Stats> {
        // Drop old data completely to free up memory.
        self.streams = BlockStreams::new().into();

        let mut streams = VecDeque::new();
        self.allocator.retrieve(|blocks| {
            streams.push_back(block::Stream::new(blocks));
        })?;
        for stream in &mut streams {
            stream.load()?;
            stream.initialize().map_err(StreamError::BlockStreamError)?;
        }
        self.streams = BlockStreams::try_from(streams)
            .map_err(|_| StreamError::BrokenSpan)?
            .into();
        // The offset is always at 0, because block streams were re-created.
        *self.offset.get_mut() = 0;
        Ok(self.streams.get_mut().stats(0))
    }

    /// Sets the offset by searching through the block streams data. The offset
    /// is set only when the exact match is found, otherwise `false` is
    /// returned. If there are multiple matches, the match within the left-most
    /// block is selected. `false` may also be returned if the offset is before
    /// the previous offset on this stream.
    ///
    /// The `compare` function takes current and the very last data block as
    /// input and the return value controls the direction of the search.
    /// The current implementation does binary search within each block
    /// stream per block, starting from the left-most one, where buffers are
    /// passed in full. Therefore, searching to the left at the block stream
    /// or a buffer boundary will result in an error.
    ///
    /// For linear search, the offset can be recovered via [`Stream::iter`]
    /// followed by [`Stream::advance`].
    #[must_use = "are you sure the offset has been set?"]
    pub fn set_offset(&mut self, compare: impl Fn(&[u8], &[u8]) -> SearchControl) -> bool {
        let previous = self.offset.get_mut();
        if let Ok(offset) = self.streams.get_mut().offset_at(compare) {
            if *previous <= offset {
                *previous = offset;
                return true;
            }
        }
        false
    }

    /// A slightly tricky way to lock updates to buffers, and therefore
    /// spanned appends, without interfering with regular appends.
    ///
    /// This makes use of a lock on the first empty block stream following
    /// the stream to be used during the next append, to match the logic of
    /// the only place where `set_buffer` is called from concurrently. Once
    /// that empty stream is locked, spanned append will have to wait,
    /// or the lock will be acquired after a spanned append, i.e. after the
    /// buffer has been written successfully, preventing concurrent
    /// modification of buffers.
    ///
    /// `None` is returned if the lock does not have to be acquired, that is
    /// when there are no empty streams following current stream used for
    /// appends. Otherwise, the call will block until the lock is acquired.
    #[inline(always)]
    fn lock_buffers(
        streams: &BlockStreams<A::Blocks>,
    ) -> Option<block::LockedStream<'_, A::Blocks>> {
        streams
            .pick_ending()
            .zip(streams.pick_empty())
            .map(|(available, empty)| if available == empty { empty + 1 } else { empty })
            .and_then(|index| streams.get(index))
            .map(block::Stream::lock)
    }

    /// Releases all memory and returns the underlying allocator.
    #[inline(always)]
    #[must_use = "do you want to drop instead?"]
    pub fn into_inner(self) -> A {
        self.allocator
    }

    /// Returns the underlying block streams that allow only certain operations.
    /// The block streams are safe to move between threads.
    #[inline(always)]
    #[must_use]
    pub fn block_streams(&self) -> Vec<ProxyBlockStream<A::Blocks>> {
        self.streams
            .read()
            .streams
            .iter()
            .map(|stream| ProxyBlockStream(Arc::clone(stream)))
            .collect()
    }

    /// Returns stats about this stream. The returned value is a snapshot
    /// in time. Call this function again to get fresh stats.
    ///
    /// Be aware, that some stats may be slightly off and not add up due to
    /// concurrent nature of a stream, although the chances are low. This
    /// is the case primarily during concurrent appends which span block
    /// streams.
    #[inline(always)]
    #[must_use]
    pub fn stats(&self) -> Stats {
        // Relaxed, because operations on streams are not dependent on
        // changes to the state, hence interested only in the atomicity.
        let offset = self.offset.load(atomic::Ordering::Relaxed);
        let streams = self.streams.read();
        streams.stats(offset)
    }

    /// Returns an iterator over data in this stream. See documentation on
    /// [`Iter`] for more details.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, A::Blocks> {
        Iter::new(&self.offset, &self.streams)
    }

    /// Advances the stream offset by the context provided. Returns `true` if
    /// the offset has been updated, `false` otherwise. It's up to the caller
    /// to provide the context generated from the iterator associated with
    /// the stream to be advanced, or it could panic or advance to an offset
    /// which is incorrect.
    ///
    /// If stream is advanced at the same moment, which means that multiple
    /// threads pass the context taken from the same iterator state,
    /// the larger value will win. In other words, consider calls which
    /// increment by `3` and `5` from `0`, while another thread observes the
    /// state. The observed values could be `3`, then `5`, or just `5`, but
    /// never `8`.
    ///
    /// # Panics
    ///
    /// Panics if advancing past the end of the stream.
    pub fn advance(&self, ctx: AdvanceContext) -> bool {
        let streams = self.streams.read();
        assert!(
            ctx.0 <= streams.len().1 + streams.removed().1,
            "advance past the end of the stream"
        );
        // Using relaxed ordering, because this state is used merely as an atomic
        // counter, it is always consistent with the streams and is always
        // incrementing.
        self.offset.fetch_max(ctx.0, atomic::Ordering::Relaxed) < ctx.0
    }

    /// Appends `bytes` to this stream according to [`SpanBehavior`].
    ///
    /// The maximum allowed size of an append can be no larger than the
    /// capacity of a single [`block::Stream`] at which the write is happening.
    /// Furthermore, there is a hard limit imposed by the current block stream
    /// implementation - see [`block::Stream::append_with_opts`].
    ///
    /// This call expects block streams to be in a fully synced state and
    /// returns a context necessary to sync the memory, which if dropped,
    /// forces the sync. It also locks the block stream at which append is
    /// going to happen, preventing concurrent appends.
    ///
    /// Unless the data is synced via [`AppendContext::sync`] or by dropping
    /// the returned value, the underlying blocks are not modified and the
    /// following appends will fail due to the stream(s) being in a dirty state.
    ///
    /// Since it is guaranteed that no more appends will happen until the sync
    /// is complete, and dirty reads are not accessible to readers, the caller
    /// could choose to run some extra code during appends with atomic properties.
    ///
    /// # Errors
    ///
    /// If size of `bytes` exceeds the allowed size of a single block stream,
    /// [`StreamError::AppendTooLarge`] is returned, or if the limit is hit on
    /// a block stream side, [`block::StreamError::AppendTooLarge`]. See above
    /// description for details.
    ///
    /// A concurrent append will result in a [`block::StreamError::Busy`] or
    /// [`StreamError::Dirty`] error, depending on the state.
    ///
    /// If block streams are not available to handle the append,
    /// [`StreamError::Unavailable`] error is returned. This can be fixed by
    /// calling [`Stream::grow`], which can add empty block stream(s) to the
    /// end.
    ///
    /// Neither memory, nor the storage is modified if an errors is returned.
    ///
    /// # Panics
    ///
    /// Panics if the current stream used for appends is followed by a stream
    /// which is not an empty. It will also panic if the number of bytes written
    /// to a block stream does not match the input, which should never happen.
    pub fn append(&self, bytes: &[u8]) -> Result<AppendContext<'_, A::Blocks>, StreamError> {
        block::Stream::<A::Blocks>::verify_append(bytes).map_err(StreamError::BlockStreamError)?;

        let streams = self.streams.read();
        let index = streams.pick_ending().ok_or(StreamError::Unavailable)?;

        // Appending empty bytes always succeeds.
        if bytes.is_empty() {
            return Ok(AppendContext::new(&self.streams));
        }
        // This is an extra check due to the order of syncs during spanned
        // appends - first the right stream is synced, then the left one. If
        // the right sync succeeds, yet the left one fails, the following
        // appends would just continue appending to the right stream and
        // further. To prevent that, make sure that the left stream has been
        // synced too by checking whether it is dirty.
        // It is OK to check it here before locking the streams, as it is
        // not going to be used for appends anymore.
        if streams
            .get(index.saturating_sub(1))
            .is_some_and(block::Stream::is_dirty)
        {
            return Err(StreamError::Dirty);
        }

        macro_rules! acquire_stream {
            ($index:expr) => {{
                let stream = streams.get($index).ok_or(StreamError::Unavailable)?;
                stream.try_lock().map_err(StreamError::BlockStreamError)?
            }};
        }

        let stream = acquire_stream!(index);
        if stream.is_dirty() {
            return Err(StreamError::Dirty);
        }
        let remaining = stream.capacity() - stream.len();
        if bytes.len() <= remaining {
            let written = block::LockedStream::append(&stream, bytes)
                .expect("input is checked prior to writing");
            assert_eq!(written, bytes.len());
            return Ok(AppendContext::new(&self.streams).with_left(index));
        }
        if stream.is_empty() {
            return Err(StreamError::AppendTooLarge(stream.capacity()));
        }

        // If bytes do not fit into a single stream, choose based on the
        // configured behavior, but first lock the next stream in sequence,
        // which is supposed to be empty.
        let empty = acquire_stream!(index + 1);
        assert!(empty.is_empty());
        // We are going to write all bytes to the next stream at most, so
        // check that they can actually fit into that stream.
        if bytes.len() > empty.capacity() {
            return Err(StreamError::AppendTooLarge(empty.capacity()));
        }

        match self.span_behavior {
            SpanBehavior::Never => {
                let written = block::LockedStream::append(&empty, bytes)
                    .expect("input is checked prior to writing");
                assert_eq!(written, bytes.len());
                Ok(AppendContext::new(&self.streams).with_right(index + 1))
            }
            SpanBehavior::Sized(limit) if bytes.len() >= limit => {
                let written = block::LockedStream::append(&empty, bytes)
                    .expect("input is checked prior to writing");
                assert_eq!(written, bytes.len());
                Ok(AppendContext::new(&self.streams).with_right(index + 1))
            }
            SpanBehavior::Sized(_) => {
                // Writing whole bytes here, so it is correctly marked as trail
                // on a block stream side.
                let written = block::LockedStream::append(&stream, bytes)
                    .expect("input is checked prior to writing");
                assert_eq!(written, remaining);
                let trailing = &bytes[..remaining];
                let spilled = &bytes[remaining..];
                let written =
                    block::LockedStream::append_with_opts(&empty, spilled, /*spilled=*/ true)
                        .expect("input is checked prior to writing");
                assert_eq!(written, spilled.len());
                // SAFETY: Concurrent clones of streams are protected by a
                // lock on the `empty` block stream, hence preventing a clone
                // of buffers which are being updated. See `lock_buffers`.
                assert!(
                    unsafe { streams.set_buffer(index, trailing, spilled) },
                    "new buffer write"
                );
                Ok(AppendContext::new(&self.streams)
                    .with_left(index)
                    .with_right(index + 1))
            }
        }
    }

    /// Grows this stream by one allocation of `A::Blocks` via the underlying
    /// allocator `A`. This function is generally non-blocking for readers
    /// and appends, except when the append is spanning to an empty block
    /// stream in the sequence. In this case, this call will compete with
    /// the append due to potential buffer updates. It will, however, wait
    /// if readers are holding onto the underlying streams after another
    /// `grow`, `maybe_shrink` or `force_release`.
    ///
    /// Concurrent calls will be serialized, growing the stream by the
    /// number of calls in unspecified order.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered. If the error is
    /// caused by this implementation, rather than the underlying `Blocks`,
    /// the error will be of [`io::ErrorKind::Other`] kind and the value set
    /// to [`StreamError::BlockStreamError`] as returned during
    /// [`block::Stream::initialize`].
    ///
    /// # Panics
    ///
    /// If newly allocated stream happens to have data written.
    pub fn grow(&self) -> io::Result<()> {
        let mut stream = block::Stream::new(self.allocator.alloc()?);
        // No need to load - the new stream is assumed to be zero-initialized.
        stream.initialize().map_err(StreamError::BlockStreamError)?;
        assert!(stream.is_empty());
        self.streams.update_default(move |current, streams| {
            let _locked = Self::lock_buffers(current);
            streams.clone_from(current);
            streams
                .append(Arc::new(stream))
                .expect("appending an empty stream should always succeed");
        });
        Ok(())
    }

    /// Tries to shrink this stream, removing blocks which are no longer
    /// accessible after advancing the offset of this stream via
    /// [`Stream::advance`]. Returns the number of blocks and bytes removed,
    /// which are set to zero if there's nothing to remove.
    ///
    /// This function has similar blocking behavior with regards to readers
    /// and appends as [`Stream::grow`]. However, this one adds extra
    /// synchronization with [`Stream::try_release`], both of which refer to a
    /// state under a shared lock.
    ///
    /// Neither the underlying blocks, nor the memory backing the block stream
    /// is released immediately, instead block streams are are put into a queue.
    /// To release, call [`Stream::try_release`] or [`Stream::force_release`],
    /// which will clear the memory and attempt to release the blocks.
    pub fn maybe_shrink(&self) -> (usize, usize) {
        // Relaxed, because operations on streams are not dependent on
        // changes to the state, hence interested only in the atomicity.
        let offset = self.offset.load(atomic::Ordering::Relaxed);
        let streams = self.streams.read();
        let (index, _) = streams
            .stream_at(offset)
            .expect("offset should always point to a valid stream");
        // This branch is meant to duplicate with `compare_update_default`
        // below to avoid acquiring a lock without any need to update. It's
        // here since we hold the read reference to compute the index anyway.
        if index == streams.removed().0 {
            return (0, 0);
        }
        let mut result = (0, 0);
        let mut release = Vec::with_capacity(index - streams.removed().0);
        let updated = self.streams.compare_update_default(
            |current| index != current.removed().0,
            |current, streams| {
                let _locked = Self::lock_buffers(current);
                streams.clone_from(current);
                let removed = streams.removed();
                for _ in removed.0..index {
                    // While this removes the block streams from the new value,
                    // the current value still references them and there
                    // could be readers holding onto them. So the actual clean
                    // of blocks will happen later. What we do now, is fill
                    // the "queue" of block streams to be released.
                    release.push(streams.remove());
                }
                result = (
                    streams.removed().0 - removed.0,
                    streams.removed().1 - removed.1,
                );
            },
        );
        if !updated {
            return (0, 0);
        }
        let mut releasables = self.releasables.lock().expect("should not poison");
        let capacity = releasables.capacity();
        for stream in &mut release {
            let stream = stream
                .take()
                .expect("index range should always include valid streams");
            debug_assert!(
                !releasables.iter().any(|item| match item.as_ref() {
                    Some(Releasable::Stream(item)) => Arc::as_ptr(item) == Arc::as_ptr(&stream),
                    _ => false,
                }),
                "removed block streams are supposed to be unique"
            );
            releasables.push(Some(Releasable::Stream(stream)));
        }
        debug_assert_eq!(
            releasables.capacity(),
            capacity,
            "too many pending releasables"
        );
        result
    }

    /// Attempts to release blocks backing up previously removed block streams
    /// during [`Stream::maybe_shrink`]. This function always returns the number
    /// of blocks released via allocator and remaining in the queue.
    ///
    /// This call does not interfere with readers, appends, and even `grow`
    /// calls, but shares state with `maybe_shrink` under a mutually exclusive
    /// lock.
    ///
    /// # Errors
    ///
    /// The third return value is optionally set to the error encountered during
    /// [`BlocksAllocator::release`].
    ///
    /// On error, blocks that failed are put back into a queue. The order in
    /// which blocks are attempted to be released is the same, as the order
    /// they were removed from this stream.
    ///
    /// Note, that the memory backing up a block stream is released despite
    /// the error returned by the underlying allocator.
    pub fn try_release(&self) -> (usize, usize, Option<io::Error>) {
        let mut releasables = self.releasables.lock().expect("should not poison");
        for releasable in releasables.iter_mut() {
            if let Some(Releasable::Stream(ref stream)) = releasable {
                if Arc::strong_count(stream) != 1 {
                    continue;
                }
                if let Some(Releasable::Stream(stream)) = releasable.take() {
                    let blocks = Arc::into_inner(stream)
                        .expect("releasable streams are not accessed concurrently")
                        .into_inner();
                    releasable.replace(Releasable::Blocks(blocks));
                }
            }
        }
        let mut error = None;
        for releasable in releasables.iter_mut() {
            match releasable.take() {
                Some(stream @ Releasable::Stream(_)) => {
                    releasable.replace(stream);
                }
                Some(Releasable::Blocks(blocks)) => {
                    if let Err((blocks, err)) = self.allocator.release(blocks) {
                        releasable.replace(Releasable::Blocks(blocks));
                        error.replace(err);
                    }
                }
                None => unreachable!(),
            }
            if error.is_some() {
                break;
            }
        }
        let length = releasables.len();
        releasables.retain(Option::is_some);
        (length - releasables.len(), releasables.len(), error)
    }

    /// Forces a release of removed block streams from a queue. Unlike
    /// [`Stream::try_release`], this call may compete with [`Stream::append`]
    /// and will compete with [`Stream::grow`].
    ///
    /// If some readers hold onto removed blocks, this call will block.
    pub fn force_release(&self) -> (usize, usize, Option<io::Error>) {
        // Make use of the fact that there are only 2 versions. Just make
        // the old version drop references to the removed block streams.
        self.streams.update_default(|current, streams| {
            let _locked = Self::lock_buffers(current);
            streams.clone_from(current);
        });
        self.try_release()
    }
}

/// Errors specific to the [`Stream`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StreamError {
    /// An attempt to append from a buffer that is larger than the block stream
    /// capacity, included in the inner value.
    AppendTooLarge(usize),
    /// Error specific to the underlying block stream.
    BlockStreamError(block::StreamError),
    /// Block streams sequence have unmatched trailing or spilled sections.
    BrokenSpan,
    /// A block stream selected for an append has unsynced data.
    Dirty,
    /// There are no block streams available for appends.
    Unavailable,
}

impl From<StreamError> for io::Error {
    #[inline(always)]
    fn from(value: StreamError) -> Self {
        io::Error::new(io::ErrorKind::Other, value)
    }
}

impl std::error::Error for StreamError {}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BrokenSpan => {
                write!(f, "endlessstream: spanned data is broken")
            }
            Self::Unavailable => {
                write!(f, "endlessstream: no block streams available")
            }
            Self::Dirty => {
                write!(f, "endlessstream: previous sync has not completed")
            }
            Self::AppendTooLarge(limit) => {
                write!(
                    f,
                    "endlessstream: append exceeds block stream capacity of {limit} bytes"
                )
            }
            Self::BlockStreamError(err) => {
                write!(f, "endlessstream: {err}")
            }
        }
    }
}

/// A proxy type to allow only certain operations on a [`block::Stream`].
#[derive(Debug)]
pub struct ProxyBlockStream<B>(Arc<block::Stream<B>>);

// NOTE: The function signatures must match exactly what's on `block::Stream`.
// Do not add any extra functions specific to this type.
// This is a hack, but whatever.
impl<B: Blocks> ProxyBlockStream<B> {
    /// See [`block::Stream::verify`] for details.
    #[inline(always)]
    pub fn verify(&self, error: impl FnMut(block::Inconsistency)) -> bool {
        self.0.verify(error)
    }
}

/// Describes how to handle appends that fall in between [`block::Stream`]
/// boundaries.
///
/// If the data is appended to both block streams, it has to be reconstructed
/// in memory.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SpanBehavior {
    /// Never span between block streams, always append to the next one,
    /// leaving the end of the previous block stream unused.
    Never,
    /// Span only when the size of the append is less than or equals to the
    /// enclosed value. This effectively sets the limit on the temporary
    /// buffers on reads. If the append is larger than the limit, the next
    /// block stream is selected.
    Sized(usize),
}

/// A result of a comparison function, which controls the direction of a
/// search for an offset in [`Stream::set_offset`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SearchControl {
    /// The input matches, where the value indicates the starting position
    /// of the matched item.
    Match(usize),
    /// Tells to search in the left direction.
    SearchLeft,
    /// Tells to search in the right direction.
    SearchRight,
}

/// The context that captures the state necessary to sync an append after
/// [`Stream::append`]. If dropped, it will attempt to sync once.
#[derive(Debug)]
pub struct AppendContext<'a, B: Blocks> {
    streams: &'a vlock::VLock<BlockStreams<B>, 2>,
    left: Option<usize>,
    right: Option<usize>,
}

impl<'a, B: Blocks> AppendContext<'a, B> {
    /// Creates a new empty context, which is synced.
    #[inline(always)]
    #[must_use]
    fn new(streams: &'a vlock::VLock<BlockStreams<B>, 2>) -> Self {
        Self {
            streams,
            left: None,
            right: None,
        }
    }

    /// Sets the left index, i.e. the index that is synced after right.
    #[inline(always)]
    #[must_use]
    fn with_left(mut self, index: usize) -> Self {
        self.left = Some(index);
        self
    }

    /// Sets the right index, i.e. the index that is synced first.
    #[inline(always)]
    #[must_use]
    fn with_right(mut self, index: usize) -> Self {
        self.right = Some(index);
        self
    }

    /// Whether this append has been synced fully.
    #[inline(always)]
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Attempts to sync data appended during [`Stream::append`]. For spanned
    /// appends, it first syncs the new stream, then the preceding one. Succeeds
    /// if and only if both syncs complete without error.
    ///
    /// Keep in mind that unless synced, future appends are not allowed.
    ///
    /// This call can be retried indefinitely in theory without affecting
    /// integrity. Note, however, that certain implementations clear the error,
    /// making the retry succeed, yet the data still not being written to disk.
    /// Therefore, the strategy to handle sync errors depends on the caller.
    /// Be sure to check implementation details of `B` when using this function.
    ///
    /// Re-loading data after a failed sync may lead to a failure due to
    /// corrupted metadata and thus may require manual recovery.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered while calling
    /// [`block::Stream::sync`] on one of the underlying blocks.
    #[inline]
    pub fn sync(&mut self) -> io::Result<()> {
        if self.is_synced() {
            return Ok(());
        }
        // NOTE: If the right sync completes, yet the left one fails, the reads
        // at runtime are guarded by a condition based on the left sync. With
        // the left sync incomplete, restart will see the data as corrupted -
        // see `BlockStreams::append`. Appends on a dirty stream are not allowed,
        // so holding a lock is not required when we have unsynced changes.
        let streams = self.streams.read();
        self.right.map_or(Ok(()), |index| {
            streams.get(index).expect("stream index is stable").sync()
        })?;
        self.right.take();
        self.left.map_or(Ok(()), |index| {
            streams.get(index).expect("stream index is stable").sync()
        })?;
        self.left.take();
        Ok(())
    }
}

impl<B: Blocks> Drop for AppendContext<'_, B> {
    /// Attempts to sync via [`AppendContext::sync`] once. Errors are ignored.
    #[inline(always)]
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

/// An iterator over chunks of bytes stored in [`Stream`].
///
/// This is the only way to access data. It reads maximum-sized contiguous
/// chunks of bytes from the underlying storage starting from an offset, at
/// which the stream is currently at. It extends the regular iterator interface
/// by allowing to rewind the internal position back if needed, and can be
/// cloned safely.
///
/// This iterator is endless, much like the stream, and can be held onto
/// indefinitely. See [`Iter::next`] for details.
#[derive(Debug)] // manual: Clone
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, B> {
    /// A reference to the offset of the `Stream`.
    offset: &'a atomic::AtomicUsize,
    /// A reference to the underlying streams of the `Stream`.
    streams: &'a vlock::VLock<BlockStreams<B>, 2>,

    /// A snapshot of the `Stream` offset.
    state: usize,
    /// Block stream index and a relative offset within that block stream
    /// to get the next chunk from. Relative offset larger than the size of
    /// a data section is interpreted to be within the buffer.
    next: (usize, usize),
    /// The total number of bytes read by this iterator.
    read: usize,
    /// Whether to read chunk from a buffer of the previous block stream on the
    /// next iteration.
    next_from_buffer: bool,
}

impl<'a, B> Iter<'a, B> {
    /// Creates a new iterator.
    fn new(offset: &'a atomic::AtomicUsize, streams: &'a vlock::VLock<BlockStreams<B>, 2>) -> Self {
        let mut iter = Self {
            offset,
            streams,
            state: 0,
            next: (0, 0),
            read: 0,
            next_from_buffer: false,
        };
        iter.refresh(&streams.read());
        iter
    }

    /// Returns context, that can be passed to [`Stream::advance`] to advance
    /// the offset, from which data is read. The context is generated based on
    /// the total length of bytes returned by this iterator. To control partial
    /// reads of returned bytes, call [`Iter::rewind`] before creating the
    /// context.
    #[inline(always)]
    pub fn advance_context(&self) -> AdvanceContext {
        AdvanceContext(self.state + self.read)
    }

    /// Rewinds the iterator back by `dec` bytes, allowing repeated reads without
    /// rebuilding the iterator. This can be thought of as an alternative to
    /// [`std::io::Seek`], except that it goes backwards only and the iterator
    /// provides direct access to the buffer. Returns `false` if rewind failed
    /// because the stream advanced further than the state of the iterator,
    /// which is equivalent to a skipped chunk in `next`.
    ///
    /// If `dec` is larger than the bytes read, it will rewind to the original
    /// offset, which is equivalent to resetting the iterator.
    #[inline(always)]
    pub fn rewind(&mut self, dec: usize) -> bool {
        let streams = self.streams.read();
        if !self.refresh(&streams) {
            self.rewind_with_streams(dec, &streams);
            return true;
        }
        false
    }

    /// Internal version of `rewind`, which uses previously read streams.
    #[inline(always)]
    fn rewind_with_streams(&mut self, dec: usize, streams: &BlockStreams<B>) {
        // Follow the normal access path, since `next` is being reset.
        self.next_from_buffer = false;
        self.read = self.read.saturating_sub(dec);
        self.next = streams
            .stream_at(self.state + self.read)
            .expect("state and length should always be less than or equal to stream length");
    }

    /// Attempts to refresh the state of this iterator by comparing the offsets
    /// with the [`Stream`], from which this iterator was created. Returns
    /// `true` if the stream has advanced further than the current position
    /// being read by this iterator, which means that some data has been
    /// skipped.
    #[inline(always)]
    fn refresh(&mut self, streams: &BlockStreams<B>) -> bool {
        // Relaxed is okay, we are interested in the atomic state only.
        let offset = self.offset.load(atomic::Ordering::Relaxed);
        if offset != self.state {
            let diff = offset
                .checked_sub(self.state)
                .expect("offset should always increment");
            let skipped = diff > self.read;
            self.state = offset;
            self.rewind_with_streams(diff, streams);
            return skipped;
        }
        false
    }
}

impl<'a, B> Iterator for Iter<'a, B> {
    type Item = Chunk<'a, B>;

    /// Returns the next chunk of bytes from the underlying [`Stream`].
    ///
    /// The `Chunk` is either a reference to a memory of a [`block::Stream`], or
    /// of the temporary buffer in case when data spans multiple block streams.
    ///
    /// Every returned `Chunk` that points to memory is safe to read even
    /// when the stream is reduced in size. On every call, the internal state
    /// is checked against the current offset of the stream and, if different,
    /// the internal state of the iterator is updated to match. If the stream
    /// offset has been advanced past the position of this iterator, a special
    /// `Chunk` is returned, which indicates that some data has been skipped,
    /// no matter whether that data would have been returned in a single or
    /// multiple chunks.
    ///
    /// Due to reference counting, try not to hold onto a single chunk for
    /// too long, as it could block allocation and release of block streams.
    ///
    /// Once drained, call to `next` will return `None` until more data is
    /// available after appends. After append, the remaining bytes are returned.
    fn next(&mut self) -> Option<Self::Item> {
        // It's OK to read on every iteration, because the stream index will
        // never go backwards.
        let streams = self.streams.read();
        if self.refresh(&streams) {
            return Some(Chunk::Skipped);
        }
        macro_rules! read_from_buffer {
            ($index:expr, $offset:expr) => {{
                self.next_from_buffer = false;
                let pos = $index - streams.removed().0;
                {
                    let buffers = streams.buffers.read();
                    let buffer = buffers[pos].as_ref().unwrap();
                    assert_ne!(buffer.len(), 0);
                    self.read += buffer.len() - $offset;
                }
                Some(Chunk::Buffer(ChunkRef {
                    index_or_pos: pos,
                    offset: $offset,
                    streams,
                }))
            }};
        }
        if self.next_from_buffer {
            // At this point the next is the stream following the one where
            // the span has started. Read the corresponding buffer in full.
            assert_eq!(self.next.1, 0);
            return read_from_buffer!(self.next.0 - 1, self.next.1);
        }
        let (mut current, mut offset) = self.next;
        while let Some(stream) = streams.get(current) {
            // Skip any empty streams in between when advancing to the next one.
            let mut skip = 0;
            while let Some(next) = streams.get(current + skip + 1) {
                if !next.is_empty() {
                    break;
                }
                skip += 1;
            }

            // If the next stream is not empty and has been synced and the
            // current stream has been appended and synced, this or the
            // following iteration will surely drain the data, so we can advance
            // the stream position safely.
            // NOTE: This is a very important invariant in this stream
            // implementation! It requires that the next stream is synced after
            // the current stream when the append spans block streams, yet
            // buffers have to be updated prior to any sync.
            if !stream.is_dirty()
                && streams
                    .get(current + skip + 1)
                    .is_some_and(|next| !next.is_empty())
            {
                self.next.0 += skip + 1;
                self.next.1 = 0;
            }

            // NOTE: The order matters here. We take the trailing snapshot
            // first, which will indicate whether this buffer has been fully
            // written. Accessing data after that will give the full view. If
            // that were in a different order, the resulting data could be
            // incomplete: imagine that two appends happened after reading
            // data and before reading trailing section. One append extended
            // data, another added the trailing section. The iterator would
            // then return incomplete data and proceed to reading the trail.
            let trailing = stream.trailing();
            atomic::compiler_fence(atomic::Ordering::SeqCst);
            let data = stream.data();
            if offset < data.len() {
                let read = data.len() - offset;
                self.read += read;
                // If we are within the same stream, advance the position within
                // that stream by the amount of the bytes read and switch to
                // the next stream if we read exactly till the end and there
                // is no more data available after.
                if self.next.0 == current {
                    self.next.1 += read;
                } else if !trailing.is_empty() {
                    // NOTE: Checking trailing bytes on the stream only when the
                    // stream index has been advanced to ensure that the data
                    // was synced fully on both streams. If it has, there is
                    // definitely a buffer available. Otherwise it could
                    // happen that buffers were written and sync has not
                    // happened yet, or worse, the sync has finished only
                    // on one of both streams.
                    self.next_from_buffer = true;
                }
                return Some(Chunk::Data(ChunkRef {
                    index_or_pos: current,
                    offset,
                    streams,
                }));
            }
            if !trailing.is_empty() {
                return read_from_buffer!(current, offset - data.len());
            }
            if self.next == (current, offset) {
                return None;
            }
            (current, offset) = self.next;
        }
        None
    }
}

impl<B> Clone for Iter<'_, B> {
    fn clone(&self) -> Self {
        // Manual field-by-field implementation to avoid creating a dependency
        // on a `Clone` trait for `B`, which is added via derive macro.
        Self {
            offset: self.offset,
            streams: self.streams,
            state: self.state,
            next: self.next,
            read: self.read,
            next_from_buffer: self.next_from_buffer,
        }
    }
}

/// A reference to the data of a chunk. See [`Chunk`] for details.
#[doc(hidden)]
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct ChunkRef<'a, B> {
    index_or_pos: usize,
    offset: usize,
    streams: vlock::ReadRef<'a, BlockStreams<B>, 2>,
}

/// A single item returned by [`Iter::next`] referencing contiguous bytes in
/// memory, or just a value indicating that some data has been skipped.
#[derive(Debug, Eq, Hash, PartialEq)]
pub enum Chunk<'a, B> {
    /// A skipped chunk means that the stream has advanced further than the
    /// last position of the iterator and indicates that some data was skipped
    /// and is no longer accessible.
    Skipped,
    /// Chunk referring to [`block::Stream`] data.
    Data(ChunkRef<'a, B>),
    /// Chunk referring to a temporary buffer holding spanned data.
    Buffer(ChunkRef<'a, B>),
}

impl<B> Chunk<'_, B> {
    /// Whether this chunk is a skipped chunk.
    #[inline(always)]
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped)
    }
}

impl<B: Blocks> Chunk<'_, B> {
    /// Returns a reference to bytes in memory, or `None` if this chunk is
    /// a skipped chunk.
    #[inline(always)]
    #[must_use]
    pub fn bytes(&self) -> Option<BytesRef<'_>> {
        match self {
            Self::Skipped => None,
            Self::Data(ref cref) => Some(BytesRef(SliceOrBuffer::Slice(
                &cref
                    .streams
                    .get(cref.index_or_pos)
                    .expect("stream index is stable")
                    .data()[cref.offset..],
            ))),
            Self::Buffer(ref cref) => Some(BytesRef(SliceOrBuffer::Buffer(
                cref.index_or_pos,
                cref.offset,
                cref.streams.buffers.read(),
            ))),
        }
    }
}

#[derive(Debug)]
enum SliceOrBuffer<'a> {
    Slice(&'a [u8]),
    Buffer(
        usize,
        usize,
        vlock::ReadRef<'a, VecDeque<Option<Arc<Vec<u8>>>>, 2>,
    ),
}

/// A reference to bytes within [`Stream`], which is either memory backing
/// [`block::Stream`], or a reference to a buffer with spanned data.
#[derive(Debug)] // manual: Eq, Hash, Ord, PartialEq, PartialOrd
pub struct BytesRef<'a>(SliceOrBuffer<'a>);

impl AsRef<[u8]> for BytesRef<'_> {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl core::borrow::Borrow<[u8]> for BytesRef<'_> {
    #[inline(always)]
    fn borrow(&self) -> &[u8] {
        self
    }
}

impl Eq for BytesRef<'_> {}

impl core::hash::Hash for BytesRef<'_> {
    #[inline(always)]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

impl Ord for BytesRef<'_> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl PartialEq<BytesRef<'_>> for BytesRef<'_> {
    #[inline(always)]
    fn eq(&self, other: &BytesRef<'_>) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialOrd<BytesRef<'_>> for BytesRef<'_> {
    #[inline(always)]
    fn partial_cmp(&self, other: &BytesRef<'_>) -> Option<core::cmp::Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl core::ops::Deref for BytesRef<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        match self.0 {
            SliceOrBuffer::Slice(bytes) => bytes,
            SliceOrBuffer::Buffer(pos, offset, ref buffers) => {
                &buffers[pos].as_deref().unwrap()[offset..]
            }
        }
    }
}

/// A context needed to advance a stream via [`Stream::advance`]. Can only be
/// constructed via [`Iter::advance_context`].
#[derive(Clone, Copy, Debug)]
#[must_use = "the context does not do anything by itself unless used"]
pub struct AdvanceContext(usize);

/// Stats about an [`Stream`]. Can be useful for choosing whether to grow the
/// stream, or for reporting telemetry, or for whatever.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Stats(Vec<BlockStreamStats>);

impl Stats {
    /// Returns an iterator over stats about individual [`block::Stream`].
    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, BlockStreamStats> {
        self.0.iter()
    }

    /// Returns the total size in bytes of the underlying `Blocks`.
    #[inline(always)]
    #[must_use]
    pub fn blocks_size(&self) -> u64 {
        self.iter().map(BlockStreamStats::blocks_size).sum()
    }

    /// Returns the total capacity for data in bytes.
    #[inline(always)]
    #[must_use]
    pub fn data_capacity(&self) -> usize {
        self.iter().map(|stats| stats.data_capacity).sum()
    }

    /// Returns the total size of appended and synced data in bytes.
    #[inline(always)]
    #[must_use]
    pub fn data_used(&self) -> usize {
        self.iter().map(|stats| stats.data_used).sum()
    }

    /// Returns the total size of wasted capacity in bytes. This includes
    /// the data that has been spilled without preceding trailing section,
    /// data that is not accessible due to the offset within the stream, and
    /// spare capacity that cannot be used due to the position within streams.
    ///
    /// `data_capacity` - `data_wasted` is the total size of accessible data.
    #[inline(always)]
    #[must_use]
    pub fn data_wasted(&self) -> usize {
        self.iter().map(|stats| stats.data_wasted).sum()
    }

    /// Returns the total size in bytes available for appends.
    #[inline(always)]
    #[must_use]
    pub fn data_available(&self) -> usize {
        self.iter().map(|stats| stats.data_available).sum()
    }
}

/// Stats about an individual [`block::Stream`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub struct BlockStreamStats {
    /// The number of blocks backing up a block stream.
    pub block_count: u64,
    /// The size of a single block expressed as a power of two.
    pub block_shift: u32,
    /// The capacity for data in bytes.
    pub data_capacity: usize,
    /// The size of appended and synced data in bytes.
    pub data_used: usize,
    /// The size of inaccessible data in bytes.
    pub data_wasted: usize,
    /// The size in bytes available for appends.
    pub data_available: usize,
}

impl BlockStreamStats {
    /// Returns the size in bytes of the underlying `Blocks`.
    #[inline(always)]
    #[must_use]
    pub fn blocks_size(&self) -> u64 {
        self.block_count << self.block_shift
    }
}

/// Either blocks or a stream holding blocks, which are yet to be released.
/// This is to avoid creating block streams after extracting underlying blocks
/// from it.
#[derive(Debug)]
enum Releasable<B> {
    Blocks(B),
    Stream(Arc<block::Stream<B>>),
}

/// A sequence of [`block::Stream`]. It has two main purposes. First, it
/// stitches together data that spans two block streams. Second, it tracks the
/// indexes of streams by counting removed streams, making the indexes always
/// increment.
#[derive(Debug)] // manual: Clone
struct BlockStreams<B> {
    /// Block streams wrapped in Arc to allow quick cloning.
    /// After a first empty stream, all subsequent streams are empty.
    streams: VecDeque<Arc<block::Stream<B>>>,

    /// Buffers that store data that spans block streams stitched together
    /// into contiguous byte stream. Each buffer is created once either during
    /// `append` or a spanned append to the stream via `set_buffer`, and is
    /// dropped when block stream that hold the trail part is removed. The
    /// collection is under a lock to allow mutation via `set_buffer`.
    buffers: vlock::VLock<VecDeque<Option<Arc<Vec<u8>>>>, 2>,

    /// Cumulative number of streams and sum of all bytes removed so far.
    /// Used to recover the current offsets in streams. This allows for the
    /// state to always increment. For data that is written in between block
    /// streams, i.e. a trail on one and a spill on another, the sum is
    /// incremented by the size of that write when the block with the trail is
    /// removed.
    removed: (usize, usize),
}

impl<B> BlockStreams<B> {
    /// Creates a new empty instance of `BlockStreams`.
    #[inline(always)]
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    /// Returns the number of block streams and bytes removed via
    /// [`BlockStreams::remove`].
    #[inline(always)]
    #[must_use]
    fn removed(&self) -> (usize, usize) {
        self.removed
    }

    /// Returns the number of streams and total length in bytes of these streams.
    #[inline(always)]
    #[must_use]
    fn len(&self) -> (usize, usize) {
        let len: usize = self.streams.iter().map(|stream| stream.len()).sum();
        (
            self.streams.len(),
            // Everything except the spilled section on the first stream adds up
            // correctly, so just subtract that.
            len - self
                .streams
                .front()
                .map_or(0, |stream| stream.spilled().len()),
        )
    }

    /// Returns a block stream by index, if present. The index is always
    /// incrementing, despite block streams being removed.
    #[inline(always)]
    #[must_use]
    fn get(&self, index: usize) -> Option<&block::Stream<B>> {
        if index >= self.removed.0 {
            self.streams.get(index - self.removed.0).map(AsRef::as_ref)
        } else {
            None
        }
    }

    /// Appends a block stream to the end of this sequence of streams, returning
    /// the stream back if it is a spilled stream and the last stream does not
    /// have a trailing section.
    ///
    /// # Panics
    ///
    /// Panics if `stream` is uninitialized or has length trailing bytes equal
    /// to the data capacity of the stream.
    #[inline(always)]
    fn append(&mut self, stream: Arc<block::Stream<B>>) -> Result<(), Arc<block::Stream<B>>> {
        assert!(
            stream.trailing().len() != stream.capacity(),
            "append uninitialized, or with trail too large"
        );
        let buffers = self.buffers.get_mut();
        if let Some(last) = self.streams.back() {
            let buffer = match make_span_buffer(last.trailing(), stream.spilled()) {
                Ok(buffer) => buffer,
                Err(()) => return Err(stream),
            }
            .and_then(|buffer| buffers.back_mut().unwrap().replace(Arc::new(buffer)));
            assert_eq!(buffer, None);
        }
        self.streams.push_back(stream);
        buffers.push_back(None);
        Ok(())
    }

    /// Attempts to remove a stream from the front, returning `None` if there
    /// is nothing to remove.
    #[inline(always)]
    fn remove(&mut self) -> Option<Arc<block::Stream<B>>> {
        self.buffers.get_mut().pop_front().map(|buffer| {
            let stream = self.streams.pop_front().unwrap();
            self.removed.0 += 1;
            self.removed.1 += stream.data().len();
            self.removed.1 += buffer.map_or(0, |buffer| buffer.len());
            stream
        })
    }

    /// Returns the index of the first empty block stream following the
    /// non-empty stream, which may be the same block stream, as returned by
    /// `pick_ending`.
    #[inline(always)]
    #[must_use]
    fn pick_empty(&self) -> Option<usize> {
        let mut iter = self.streams.iter().enumerate().rev().peekable();
        while let stream @ Some(_) = iter.next() {
            if !stream.unwrap().1.is_empty() {
                break;
            }
            if iter.peek().filter(|next| next.1.is_empty()).is_some() {
                continue;
            }
            return stream
                .filter(|stream| stream.1.is_empty())
                .map(|stream| stream.0 + self.removed.0);
        }
        None
    }

    /// Returns the index of the last non-empty block stream that is not full,
    /// or the first empty stream following a full one otherwise. Returns
    /// `None` if none of the above are available.
    #[inline(always)]
    #[must_use]
    fn pick_ending(&self) -> Option<usize> {
        let mut iter = self.streams.iter().enumerate().rev().peekable();
        while let Some(stream) = iter.next() {
            if stream.1.is_full() {
                break;
            }
            if !stream.1.is_empty() {
                return Some(stream.0 + self.removed.0);
            }
            if let Some(next) = iter.peek() {
                if next.1.is_full() {
                    return Some(stream.0 + self.removed.0);
                }
            } else {
                return Some(stream.0 + self.removed.0);
            }
        }
        None
    }

    /// Returns the block stream index and the relative offset in data and
    /// buffer at a given `offset`. The `offset` provided must be cumulative
    /// and account for the previously removed streams.
    #[inline(always)]
    #[must_use]
    fn stream_at(&self, offset: usize) -> Option<(usize, usize)> {
        if offset < self.removed.1 {
            return None;
        }

        let mut cumulative = self.removed.1;
        let mut last = (0, 0);
        for (pos, stream) in self.streams.iter().enumerate() {
            if stream.is_empty() {
                continue;
            }
            let length = cumulative;
            // NOTE: Reading data from buffers, as they are guaranteed to be
            // correct as long as trailing section has been synced to the stream.
            if !stream.trailing().is_empty() {
                let buffers = self.buffers.read();
                if let Some(buffer) = buffers[pos].as_deref() {
                    assert_ne!(buffer.len(), 0);
                    cumulative += buffer.len();
                }
            }
            cumulative += stream.data().len();
            if offset < cumulative {
                return Some((pos + self.removed.0, offset - length));
            }
            last = (pos + self.removed.0, cumulative - length);
        }
        if offset == cumulative {
            if self.get(last.0).unwrap().capacity() != last.1 {
                return Some(last);
            }
            return Some((last.0 + 1, 0));
        }
        None
    }

    /// Sets the spanned data buffer at index to the value of trailing and
    /// spilled bytes. Return value indicates whether the buffer was set.
    /// More precisely, if the buffer already exists or both `trailing` and
    /// `spilled` are empty, `false` will be returned. In other words,
    /// overwrites of buffers are not allowed.
    ///
    /// # Safety
    ///
    /// First, the `trailing` bytes must correspond to the index and `spilled`
    /// bytes must be read from the following one stream, as the value will be
    /// fixed and never updated after being set. Second, concurrent clones of
    /// `BlockStreams` must be protected by a lock when calling this function,
    /// such that the buffers are updated before the clone is done, or the
    /// clone is done before the buffer is set. Third, this function must be
    /// called before syncing the underlying block streams, so that the buffers
    /// can be accessed safely through reads.
    ///
    /// # Panics
    ///
    /// Panics if one of trailing or spilled bytes is empty, while another
    /// is not. Also, if there is no valid buffer at the given index or if the
    /// index is of the last buffer or larger.
    #[inline(always)]
    #[must_use]
    unsafe fn set_buffer(&self, index: usize, trailing: &[u8], spilled: &[u8]) -> bool {
        let pos = index - self.removed.0;
        // Buffers track streams in length, but it does not make sense to set
        // the buffer at the last position.
        assert!(
            pos < self.streams.len().saturating_sub(1),
            "index too large"
        );
        let buffer = make_span_buffer(trailing, spilled)
            .expect("trailing and spilled should be either empty or non-empty");
        if buffer.is_none() {
            return false;
        }
        self.buffers.compare_update_default(
            |current| current[pos].is_none(),
            move |current, buffers| {
                buffers.clone_from(current);
                buffers[pos] = buffer.map(Arc::new);
            },
        )
    }
}

impl<B: Blocks> BlockStreams<B> {
    /// Returns the stats about these block streams.
    #[inline(always)]
    #[must_use]
    fn stats(&self, offset: usize) -> Stats {
        let ending = self.pick_ending().map(|index| index - self.removed.0);
        let mut cumulative = self.removed.1;
        let mut stats: Vec<BlockStreamStats> = Vec::with_capacity(self.streams.len());
        for (pos, stream) in self.streams.iter().enumerate() {
            let capacity = stream.capacity();
            let length = stream.len();

            cumulative += length;
            // A special case, where the first stream has a spilled section,
            // which is not accessible precisely because it's the first stream.
            if pos == 0 {
                cumulative -= stream.spilled().len();
            }

            stats.push(BlockStreamStats {
                block_count: stream.block_count(),
                block_shift: stream.block_shift(),
                data_capacity: capacity,
                data_used: length,
                data_wasted: length.saturating_sub(cumulative.saturating_sub(offset)),
                data_available: 0,
            });
            // Bytes are available only when the stream is at or after the
            // ending stream, otherwise it will never be appended to, and
            // therefore is wasted.
            if ending.is_some_and(|ending| pos >= ending) {
                stats.last_mut().unwrap().data_available = capacity - length;
            } else {
                stats.last_mut().unwrap().data_wasted += capacity - length;
            }
        }
        Stats(stats)
    }

    /// Returns the offset within block streams given the comparison function,
    /// which takes current and the last data block as input. The search is
    /// done via binary search per block, and the match within the left-most
    /// block is returned. The return value of the function controls the
    /// direction of binary search.
    ///
    /// The search starts from the first block stream. This function is used
    /// in the context, where if offset has been advanced, the block streams
    /// would likely be removed, hence it is likely that the position we are
    /// searching for is in the beginning of the block streams.
    ///
    /// The returned offset is cumulative. If there was no exact match via
    /// comparison function, `Err` is returned with the offset at which the
    /// search has ended.
    fn offset_at(&self, compare: impl Fn(&[u8], &[u8]) -> SearchControl) -> Result<usize, usize> {
        if self.streams.is_empty() {
            return Err(self.removed.1);
        }

        // Get the last non-empty index, which is the one before empty, or the
        // last block stream otherwise.
        let last_index = self.pick_empty().map_or_else(
            || self.removed.0 + self.streams.len() - 1,
            |index| index.saturating_sub(1),
        );

        let buffers = self.buffers.read();
        // Get the bytes of the last block of the last non-empty stream.
        // This is going to be passed to comparison function along the way.
        // Technically, this can be data spilled over multiple blocks, or data
        // that spans block streams, which is stored in buffer.
        let last_bytes = if let Some(stream) = self.get(last_index) {
            if stream.data().is_empty() {
                let buffer = buffers[last_index.saturating_sub(1) - self.removed.0].as_deref();
                if let Some(buffer) = buffer {
                    buffer
                } else {
                    return Err(self.removed.1);
                }
            } else {
                let range = stream
                    .data_range_for(*stream.data_block_range().end())
                    .expect("end block range should be contain valid data range");
                &stream.data()[range]
            }
        } else {
            return Err(self.removed.1);
        };

        let mut offset = self.removed.1;
        for index in self.removed.0..=last_index {
            let stream = self.get(index).unwrap();
            let data = stream.data();

            if !data.is_empty() {
                let block_range = stream.data_block_range();
                let mut position = None;
                let mut start_block = *block_range.start();
                let mut end_block = *block_range.end();
                while start_block != end_block {
                    let middle_block = (start_block + end_block) / 2;
                    let (block, range) = match stream.data_range_for(middle_block) {
                        Ok(range) => (middle_block, range),
                        Err(block) => (
                            block,
                            stream
                                .data_range_for(block)
                                .expect("block should be smaller than total number of blocks"),
                        ),
                    };
                    assert_ne!(block as u64, stream.block_count());

                    let range_start = range.start;
                    match compare(&data[range], last_bytes) {
                        SearchControl::Match(pos) => {
                            position = Some(range_start + pos);
                            end_block = middle_block;
                        }
                        SearchControl::SearchLeft => end_block = middle_block,
                        SearchControl::SearchRight => start_block = block + 1,
                    };
                }
                // Check the final block on which the loop has converged, since
                // we are interested in the actual position within that block.
                match stream.data_range_for(end_block) {
                    Ok(range) => {
                        let range_start = range.start;
                        match compare(&data[range], last_bytes) {
                            SearchControl::Match(pos) => {
                                position = Some(range_start + pos);
                            }
                            SearchControl::SearchLeft | SearchControl::SearchRight => (),
                        }
                    }
                    Err(_) => unreachable!("end block should always be within data range"),
                };
                // If there was a match within this stream, pick that, which is
                // the going to be the position within the left-most block if
                // the search matched multiple times.
                if let Some(pos) = position {
                    return Ok(offset + pos);
                }
                // Otherwise, if haven't reached the end of this block stream,
                // there is nothing else left to search for. Stop searching further
                // and return the offset at the start of this block.
                if end_block != *block_range.end() {
                    return Err(offset);
                }
                offset += data.len();
            }

            if let Some(buffer) = buffers[index - self.removed.0].as_deref() {
                match compare(buffer, last_bytes) {
                    SearchControl::Match(pos) => return Ok(offset + pos),
                    SearchControl::SearchLeft => return Err(offset),
                    SearchControl::SearchRight => offset += buffer.len(),
                };
            }
        }
        Err(offset)
    }
}

impl<B> Clone for BlockStreams<B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            streams: self.streams.clone(),
            buffers: self.buffers.clone(),
            removed: self.removed,
        }
    }

    #[inline(always)]
    fn clone_from(&mut self, source: &Self) {
        self.streams.clone_from(&source.streams);
        self.buffers.clone_from(&source.buffers);
        self.removed = source.removed;
    }
}

impl<B> Default for BlockStreams<B> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            streams: VecDeque::default(),
            buffers: VecDeque::default().into(),
            removed: (0, 0),
        }
    }
}

impl<B> TryFrom<VecDeque<block::Stream<B>>> for BlockStreams<B> {
    type Error = VecDeque<block::Stream<B>>;

    fn try_from(mut value: VecDeque<block::Stream<B>>) -> Result<Self, Self::Error> {
        let mut streams = Self::new();
        while let Some(stream) = value.pop_front() {
            if let Err(stream) = streams.append(Arc::new(stream)) {
                value.push_front(Arc::into_inner(stream).unwrap());
                return Err(value);
            }
        }
        Ok(streams)
    }
}

/// A convenience function to create a buffer from two chunks of bytes. Both
/// chunks must be either empty or non-empty, if only one of them is then it
/// will return an error.
#[inline(always)]
fn make_span_buffer(trail: &[u8], spill: &[u8]) -> Result<Option<Vec<u8>>, ()> {
    if trail.is_empty() ^ spill.is_empty() {
        Err(())
    } else if !trail.is_empty() {
        let mut buffer = vec![0; trail.len() + spill.len()];
        buffer[..trail.len()].copy_from_slice(trail);
        buffer[trail.len()..].copy_from_slice(spill);
        Ok(Some(buffer))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use core::{mem, time};
    use std::{
        io::{Read, Seek, Write},
        sync::mpsc,
        thread,
    };

    use super::*;

    macro_rules! assert_waits {
        ($what:expr, $until:expr, $($args:tt)+) => {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                tx.send(false).unwrap();
                $what;
                tx.send(true).unwrap();
            });
            assert!(!rx.recv().unwrap(), $($args)+);
            thread::sleep(time::Duration::from_millis(100));
            assert!(rx.try_recv().is_err(), $($args)+);
            $until;
            assert!(rx.recv().unwrap(), $($args)+);
        }
    }

    macro_rules! blockstreams {
        ($($kind:tt),+) => {{
            let mut streams = VecDeque::new();
            let mut iter = (1..).peekable();
            $(
                streams.push_back(blockstreams!(@$kind &mut iter));
            )+
            streams
        }};
        // The kind is composed of two letters:
        // 1: f | s - whether the start is fresh or spilling.
        // 2: f | t | p | e - full, with trail, partial or empty.
        (@ff $iter:expr) => {
            test_generate_blockstream($iter, 255, TestGenerateFlags::Empty as u8)
        };
        (@ft $iter:expr) => {
            test_generate_blockstream($iter, 144, TestGenerateFlags::Trailing as u8)
        };
        (@fp $iter:expr) => {
            test_generate_blockstream($iter, 144, TestGenerateFlags::Empty as u8)
        };
        (@fe $iter:expr) => {
            test_generate_blockstream($iter, 0, TestGenerateFlags::Empty as u8)
        };
        (@sf $iter:expr) => {
            test_generate_blockstream($iter, 255, TestGenerateFlags::Spilled as u8)
        };
        (@st $iter:expr) => {
            test_generate_blockstream($iter, 142,
                TestGenerateFlags::Spilled as u8 | TestGenerateFlags::Trailing as u8)
        };
        (@sp $iter:expr) => {
            test_generate_blockstream($iter, 142, TestGenerateFlags::Spilled as u8)
        };
        (@se $iter:expr) => {
            test_generate_blockstream($iter, 0, TestGenerateFlags::Spilled as u8)
        };
    }

    #[test]
    fn stream_set_offset() {
        // Search is tested in `blockstreams_offset_at`, here we test the glue.
        macro_rules! read_word {
            ($iter:expr) => {{
                let chunk = $iter.next().unwrap();
                let bytes = chunk.bytes().unwrap();
                u64::from_le_bytes(bytes[..8].try_into().unwrap())
            }};
        }

        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, ff, fe).into());
        stream.load().unwrap();

        let case = "offset within first block stream";
        let result = stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 42));
        assert!(result, "{case}");
        assert_eq!(read_word!(stream.iter()), 42, "{case}");

        let case = "offset within second block stream";
        let result = stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 198));
        assert!(result, "{case}");
        assert_eq!(read_word!(stream.iter()), 198, "{case}");

        let case = "offset before current offset is not set";
        let result = stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 42));
        assert!(!result, "{case}");
        assert_eq!(read_word!(stream.iter()), 198, "{case}");

        let case = "offset of unmatched value is not set";
        let result = stream.set_offset(|_, _| SearchControl::SearchRight);
        assert!(!result, "{case}");
        assert_eq!(read_word!(stream.iter()), 198, "{case}");
    }

    #[test]
    fn stream_block_streams() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, fe).into());
        stream.load().unwrap();
        let block_streams = stream.block_streams();
        assert_eq!(block_streams.len(), 2);
        assert!(block_streams[0].verify(|_| ()));
        assert!(block_streams[1].verify(|_| ()));
    }

    #[test]
    fn stream_stats() {
        // More specific cases are checked in `blockstreams_stats`.
        let case = "basic";
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, ff, fe).into());
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 64)));
        let stats = stream.stats();
        assert_eq!(stats.blocks_size(), 3 * 2048, "{case}");
        assert_eq!(stats.data_capacity(), 3 * 1280, "{case}");
        assert_eq!(stats.data_used(), 1152 + 1280 + 0, "{case}");
        assert_eq!(stats.data_wasted(), 63 * 8 + 128, "{case}");
        assert_eq!(stats.data_available(), 1280, "{case}");
    }

    #[test]
    fn stream_iter() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(
            blockstreams!(fe, fp, ft, sp, ff, fe, fp, fe).into(),
        );
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 42)));
        assert_eq!(stream.maybe_shrink(), (1, 0));

        let case = "first chunk";
        let mut iter = stream.iter();
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), (144 - 42) * 8 + 8, "{case}");

        let case = "second chunk";
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 144 * 8, "{case}");

        let case = "rewind before spanned chunk";
        iter.rewind(64);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 64, "{case}");

        let case = "third chunk";
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 16 * 8 + 16, "{case}");

        let case = "rewind spanned chunk";
        iter.rewind(48);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 48, "{case}");

        let case = "fourth chunk";
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 142 * 8, "{case}");

        let case = "fifth chunk";
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 160 * 8, "{case}");

        let case = "sixths chunk";
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 144 * 8, "{case}");

        let case = "rewind last chunk";
        iter.rewind(32);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 32, "{case}");

        let case = "rewind two chunks";
        iter.rewind(728 + 144 * 8);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 728, "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 144 * 8, "{case}");

        let case = "drained";
        assert!(iter.next().is_none(), "{case}");
        assert!(iter.next().is_none(), "{case}");

        let case = "consistent returned data";
        let mut value = 42;
        for (n, chunk) in stream.iter().enumerate() {
            let bytes = chunk.bytes().expect(case);
            for i in 0..bytes.len() >> 3 {
                let word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
                // Regular data is always incrementing, spanned data is always
                // repeating the same value. It's sufficient to check only the
                // incrementing data, skipping repeated, because it is checked
                // implicitly above.
                if word != TEST_WORDS_REPEATING[0] {
                    assert_eq!(word, value, "{case}: chunk={n} word={i}");
                    value += 1;
                }
            }
        }
    }

    #[test]
    fn stream_iter_advance() {
        let mut stream =
            Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(ft, sp, ff, fe).into());
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 114)));

        // One iterator is used for advancing the offset, one behind and one
        // ahead of it. One that falls behind will be skipping chunks, one that
        // is ahead should not be affected by advancing the offset.
        let mut iter = stream.iter();
        let mut behind = stream.iter();
        let mut ahead = stream.iter();

        let case = "advance within first data chunk";
        iter.next().unwrap();
        iter.rewind(188);
        ahead.next().unwrap();
        assert!(stream.advance(iter.advance_context()), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 188, "{case}");
        assert!(behind.next().expect(case).is_skipped(), "{case}");
        let chunk = behind.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 188, "{case}");
        let chunk = ahead.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 16 * 8 + 16, "{case}");

        let case = "advance within spanned chunk";
        iter.next().unwrap();
        iter.rewind(64);
        assert!(stream.advance(iter.advance_context()), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 64, "{case}");
        assert!(behind.next().expect(case).is_skipped(), "{case}");
        let chunk = behind.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 64, "{case}");
        let chunk = ahead.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 142 * 8, "{case}");

        let case = "advance until the end skipping few chunks";
        while iter.next().is_some() {}
        assert!(stream.advance(iter.advance_context()), "{case}");
        assert!(iter.next().is_none(), "{case}");
        assert!(behind.next().expect(case).is_skipped(), "{case}");
        assert!(behind.next().is_none(), "{case}");
        assert!(ahead.next().expect(case).is_skipped(), "{case}");
        assert!(ahead.next().is_none(), "{case}");
    }

    #[test]
    fn stream_iter_rewind() {
        let mut stream =
            Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(ft, sp, ff, fe).into());
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 114)));
        let mut iter = stream.iter();
        let mut behind = stream.iter();
        let mut ahead = stream.iter();
        iter.next().unwrap();
        iter.rewind(128);
        ahead.next().unwrap();

        let case = "rewind after advancing";
        assert!(stream.advance(iter.advance_context()), "{case}");
        assert!(iter.rewind(1024), "{case}");
        assert!(!behind.rewind(1024), "{case}");
        assert!(ahead.rewind(1024), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");
        let chunk = behind.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");
        let chunk = ahead.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");
    }

    #[test]
    fn stream_iter_shrink() {
        let mut stream =
            Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, fp, ff, fe).into());
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 114)));

        let case = "shrink while reading";
        let mut iter = stream.iter();
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        let mut behind = iter.clone();
        behind.rewind(64);
        assert!(stream.advance(iter.advance_context()), "{case}");
        assert_eq!(stream.maybe_shrink(), (1, 1152), "{case}");
        assert_eq!(bytes.len(), 1152 - 912 + 8, "{case}");

        let case = "read after shrink";
        assert!(behind.next().expect(case).is_skipped(), "{case}");
        let size: usize = iter.map(|chunk| chunk.bytes().expect(case).len()).sum();
        assert_eq!(size, 1152 + 1280, "{case}");
        let size: usize = behind.map(|chunk| chunk.bytes().expect(case).len()).sum();
        assert_eq!(size, 1152 + 1280, "{case}");
    }

    #[test]
    fn stream_advance() {
        // Most cases are in `stream_iter_advance`.
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, ff, fe).into());
        stream.load().unwrap();

        let case = "advance before offset";
        let ctx = stream.iter().advance_context();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 114)));
        assert!(!stream.advance(ctx), "{case}");

        let case = "advance at offset";
        assert!(!stream.advance(stream.iter().advance_context()), "{case}");
    }

    #[test]
    #[should_panic(expected = "advance past the end of the stream")]
    fn stream_advance_large() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, ff, fe).into());
        stream.load().unwrap();
        assert!(stream.set_offset(|bytes, _| test_find_word_in_stream(bytes, 114)));
        let ctx = AdvanceContext(*stream.offset.get_mut() + 1152 - 904 + 1280 + 1);
        stream.advance(ctx);
    }

    #[test]
    fn stream_load() {
        // Mostly tested implicitly in other functions.
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, ff, fe).into());

        let case = "load basic";
        stream.load().expect(case);
        assert_eq!(stream.stats().iter().count(), 3, "{case}");

        let case = "load repeated";
        let mut iter = stream.iter();
        iter.next();
        assert!(stream.advance(iter.advance_context()));
        assert_ne!(*stream.offset.get_mut(), 0, "{case}");
        stream.load().expect(case);
        assert_eq!(*stream.offset.get_mut(), 0, "{case}");
        assert_eq!(stream.stats().iter().count(), 3, "{case}");

        let case = "load from empty";
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(VecDeque::new().into());
        stream.load().expect(case);
        assert_eq!(stream.stats().iter().count(), 0, "{case}");
    }

    #[test]
    fn stream_append() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(fp, fe, fe).into());
        stream.load().unwrap();
        let mut iter = stream.iter();
        while iter.next().is_some() {}

        let case = "append larger than the capacity of the following stream fails";
        let err = stream.append(&TEST_BYTES_REPEATING[..1282]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: append exceeds block stream capacity of 1280 bytes",
            "{case}"
        );

        let case = "empty append";
        let ctx = stream.append(&[]).expect(case);
        assert!(ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");

        let case = "aligned to the end of a block stream";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..128]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");

        let case = "append larger than the capacity of an empty stream fails";
        let err = stream.append(&TEST_BYTES_REPEATING[..1282]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: append exceeds block stream capacity of 1280 bytes",
            "{case}"
        );

        let case = "append to an empty block stream";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..1000]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 1000, "{case}");

        let case = "append to the next block stream due to size";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..500]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 500, "{case}");

        let case = "append within a block stream";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..250]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..250]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        ctx.sync().expect(case);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 500, "{case}");

        let case = "append fails because no block streams available";
        let err = stream.append(&TEST_BYTES_REPEATING[..1000]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: no block streams available",
            "{case}"
        );

        let case = "append aligned to the end of the stream";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..280]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 280, "{case}");

        stream.grow().unwrap();
        let case = "append at capacity succeeds after growing the stream";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..1280]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 1280, "{case}");
    }

    #[test]
    fn stream_append_sized_span_behavior() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::with_span_behavior(
            blockstreams!(fp, fe, fe).into(),
            SpanBehavior::Sized(130),
        );
        stream.load().unwrap();
        let mut iter = stream.iter();
        while iter.next().is_some() {}

        let case = "no span with the append larger than the limit";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..132]).expect(case);
        assert!(stream.streams.read().buffers.read()[0].is_none(), "{case}");
        assert!(ctx.left.is_none(), "{case}");
        assert!(ctx.right.is_some(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 132, "{case}");

        let case = "spanned append";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..1086]).unwrap();
        ctx.sync().unwrap();
        iter.next();
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..128]).expect(case);
        assert!(stream.streams.read().buffers.read()[1].is_some(), "{case}");
        assert!(ctx.left.is_some(), "{case}");
        assert!(ctx.right.is_some(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");

        let case = "append after spanned append";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..64]).expect(case);
        assert!(!ctx.is_synced(), "{case}");
        assert!(iter.next().is_none(), "{case}");
        ctx.sync().expect(case);
        assert!(ctx.is_synced(), "{case}");
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 64, "{case}");
    }

    #[test]
    fn stream_append_concurrent() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::with_span_behavior(
            blockstreams!(ff, fp, fe).into(),
            SpanBehavior::Sized(130),
        );
        stream.load().unwrap();
        let streams = stream.streams.read();

        let case = "append to a locked block stream fails";
        let locked = streams.get(1).unwrap().lock();
        let err = stream.append(&TEST_BYTES_REPEATING[..64]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: blockstream: stream is busy with another operation",
            "{case}"
        );
        drop(locked);

        let case = "append to dirty block stream fails";
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..64]).expect(case);
        let err = stream.append(&TEST_BYTES_REPEATING[..64]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: previous sync has not completed",
            "{case}"
        );
        ctx.sync().unwrap();

        let case = "spanned append with locked buffers fails";
        let locked = Stream::<TestMemoryBlocksAllocator>::lock_buffers(&streams).unwrap();
        let err = stream.append(&TEST_BYTES_REPEATING[..128]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: blockstream: stream is busy with another operation",
            "{case}"
        );
        drop(locked);

        let case = "append during partial sync of a spanned append fails";
        let mut iter = stream.iter();
        while iter.next().is_some() {}
        let mut ctx = stream.append(&TEST_BYTES_REPEATING[..128]).expect(case);
        let err = stream.append(&TEST_BYTES_REPEATING[..64]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: previous sync has not completed",
            "{case}"
        );
        assert!(iter.next().is_none(), "{case}");
        // Make a partial sync to emulate a state during IO errors.
        streams
            .get(ctx.right.take().unwrap())
            .unwrap()
            .sync()
            .unwrap();
        let err = stream.append(&TEST_BYTES_REPEATING[..64]).err();
        assert_eq!(
            err.expect(case).to_string(),
            "endlessstream: previous sync has not completed",
            "{case}"
        );
        assert!(iter.next().is_none(), "{case}");

        let case = "append after completing partial sync succeeds";
        ctx.sync().expect(case);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 128, "{case}");
        // This one indirectly tests an API when the context is dropped.
        stream.append(&TEST_BYTES_REPEATING[..64]).expect(case);
        let chunk = iter.next().expect(case);
        let bytes = chunk.bytes().expect(case);
        assert_eq!(bytes.len(), 64, "{case}");
    }

    #[test]
    fn stream_grow() {
        let mut stream = Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(ff, fe).into());
        stream.load().unwrap();
        let stream = Arc::new(stream);
        // Hold onto a chunk to simulate concurrent reads.
        let mut iter = stream.iter();
        let chunk = iter.next().unwrap();
        assert_eq!(stream.stats().data_capacity(), 2 * 1280);

        let case = "grow while reading";
        stream.grow().expect(case);
        assert_eq!(stream.stats().data_capacity(), 3 * 1280, "{case}");

        let case = "growing waits for readers";
        let stream_clone = Arc::clone(&stream);
        assert_waits!(stream_clone.grow().expect(case), drop(chunk), "{case}");
        assert_eq!(stream.stats().data_capacity(), 4 * 1280, "{case}");

        let case = "growing waits for locked buffers";
        let streams = stream.streams.read();
        let locked = Stream::<TestMemoryBlocksAllocator>::lock_buffers(&streams).unwrap();
        let stream_clone = Arc::clone(&stream);
        assert_waits!(stream_clone.grow().expect(case), drop(locked), "{case}");
        assert_eq!(stream.stats().data_capacity(), 5 * 1280, "{case}");
    }

    #[test]
    fn stream_maybe_shrink() {
        let mut stream =
            Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(ft, sp, fp, fp, fe, fe).into());
        stream.load().unwrap();
        let stream = Arc::new(stream);
        // Hold onto a chunk to simulate concurrent reads.
        let mut iter = stream.iter();
        let chunk = iter.next().unwrap();

        let case = "no shrink when referenced";
        assert_eq!(stream.maybe_shrink(), (0, 0), "{case}");
        assert_eq!(stream.stats().data_capacity(), 6 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 0, "{case}");

        let case = "no shrink when buffer referenced";
        assert!(stream.advance(iter.advance_context()));
        assert_eq!(stream.maybe_shrink(), (0, 0), "{case}");
        assert_eq!(stream.stats().data_capacity(), 6 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 0, "{case}");

        let case = "shrink first block stream";
        iter.next().unwrap();
        assert!(stream.advance(iter.advance_context()));
        assert_eq!(stream.maybe_shrink(), (1, 1296), "{case}");
        assert_eq!(stream.stats().data_capacity(), 5 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 1, "{case}");

        let case = "shrinking waits for readers and removes two block streams";
        while iter.next().is_some() {}
        assert!(stream.advance(iter.advance_context()));
        let stream_clone = Arc::clone(&stream);
        assert_waits!(
            assert_eq!(stream_clone.maybe_shrink(), (2, 2 * 1152 - 16), "{case}"),
            drop(chunk),
            "{case}"
        );
        assert_eq!(stream.stats().data_capacity(), 3 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 3, "{case}");

        let case = "last non-full stream is not removed";
        assert!(iter.next().is_none(), "{case}");
        assert_eq!(stream.maybe_shrink(), (0, 0), "{case}");
        assert_eq!(stream.stats().data_capacity(), 3 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 3, "{case}");

        let case = "shrinking waits for locked buffers and removes last full stream";
        stream.append(&TEST_BYTES_REPEATING[..128]).unwrap();
        iter.next().unwrap();
        assert!(stream.advance(iter.advance_context()));
        assert!(iter.next().is_none(), "{case}");
        let streams = stream.streams.read();
        let locked = Stream::<TestMemoryBlocksAllocator>::lock_buffers(&streams).unwrap();
        let stream_clone = Arc::clone(&stream);
        assert_waits!(
            assert_eq!(stream_clone.maybe_shrink(), (1, 1280), "{case}"),
            drop(locked),
            "{case}"
        );
        assert_eq!(stream.stats().data_capacity(), 2 * 1280, "{case}");
        assert_eq!(stream.releasables.lock().unwrap().len(), 4, "{case}");
    }

    #[test]
    fn stream_release() {
        let mut stream =
            Stream::<TestMemoryBlocksAllocator>::new(blockstreams!(ft, sp, fp, fp, fe, fe).into());
        stream.load().unwrap();
        let stream = Arc::new(stream);
        let mut iter = stream.iter();
        iter.next().unwrap();
        iter.next().unwrap();

        let case = "recently removed are not released";
        assert!(stream.advance(iter.advance_context()));
        assert_eq!(stream.maybe_shrink(), (1, 1296), "{case}");
        // FUTURE: Use assert_matches once stable
        assert!(matches!(stream.try_release(), (0, 1, None)), "{case}");

        let case = "removed after changing streams are released";
        stream.grow().unwrap();
        assert!(matches!(stream.try_release(), (1, 0, None)), "{case}");

        let case = "recently removed are released immediately if forced";
        iter.next().unwrap();
        assert!(stream.advance(iter.advance_context()));
        assert_eq!(stream.maybe_shrink(), (1, 1152 - 16), "{case}");
        assert!(matches!(stream.force_release(), (1, 0, None)), "{case}");
    }

    #[test]
    fn blockstreams_len() {
        let case = "with spilled at front";
        let streams = BlockStreams::try_from(blockstreams!(sp, fe, ff, fe)).unwrap();
        assert_eq!(streams.len(), (4, 1136 + 0 + 1280 + 0), "{case}");

        let case = "with spanned data";
        let streams = BlockStreams::try_from(blockstreams!(sp, ft, sf, fe)).unwrap();
        assert_eq!(streams.len(), (4, 1136 + 1296 + 1264 + 0), "{case}");

        let case = "without spilled at front";
        let streams = BlockStreams::try_from(blockstreams!(fp, ft, sf, fe)).unwrap();
        assert_eq!(streams.len(), (4, 1152 + 1296 + 1264 + 0), "{case}");
    }

    #[test]
    fn blockstreams_stats() {
        let case = "with spilled at front";
        let streams = BlockStreams::try_from(blockstreams!(sp, fe, ft, sp, fe)).unwrap();
        let stats = streams.stats(88);
        assert_eq!(stats.blocks_size(), 5 * 2048, "{case}");
        assert_eq!(stats.data_capacity(), 5 * 1280, "{case}");
        assert_eq!(stats.data_used(), 1152 + 0 + 1280 + 1152 + 0, "{case}");
        assert_eq!(stats.data_wasted(), 16 + 88 + 128 + 1280 + 0, "{case}");
        assert_eq!(stats.data_available(), 1280 - 1152 + 1280, "{case}");

        let case = "with offset from the third block stream";
        let streams = BlockStreams::try_from(blockstreams!(sp, fe, ft, sp, fe)).unwrap();
        let stats = streams.stats(1168);
        assert_eq!(stats.blocks_size(), 5 * 2048, "{case}");
        assert_eq!(stats.data_capacity(), 5 * 1280, "{case}");
        assert_eq!(stats.data_used(), 1152 + 0 + 1280 + 1152 + 0, "{case}");
        assert_eq!(stats.data_wasted(), 16 + 128 + 1280 + 1168 + 0, "{case}");
        assert_eq!(stats.data_available(), 1280 - 1152 + 1280, "{case}");

        let case = "without spilled at front";
        let streams = BlockStreams::try_from(blockstreams!(fp, fe, ft, sp, fe)).unwrap();
        let stats = streams.stats(88);
        assert_eq!(stats.blocks_size(), 5 * 2048, "{case}");
        assert_eq!(stats.data_capacity(), 5 * 1280, "{case}");
        assert_eq!(stats.data_used(), 1152 + 0 + 1280 + 1152 + 0, "{case}");
        assert_eq!(stats.data_wasted(), 88 + 128 + 1280 + 0, "{case}");
        assert_eq!(stats.data_available(), 1280 - 1152 + 1280, "{case}");

        let case = "with removed block streams";
        let mut streams = BlockStreams::try_from(blockstreams!(fe, fp, fe, ft, sp, fe)).unwrap();
        streams.remove();
        let stats = streams.stats(88);
        assert_eq!(stats.blocks_size(), 5 * 2048, "{case}");
        assert_eq!(stats.data_capacity(), 5 * 1280, "{case}");
        assert_eq!(stats.data_used(), 1152 + 0 + 1280 + 1152 + 0, "{case}");
        assert_eq!(stats.data_wasted(), 88 + 128 + 1280 + 0, "{case}");
        assert_eq!(stats.data_available(), 1280 - 1152 + 1280, "{case}");
    }

    #[test]
    fn blockstreams_get() {
        let mut streams = BlockStreams::try_from(blockstreams!(fe, ff, fe)).unwrap();

        let case = "within new block streams";
        assert_eq!(streams.get(0).map(|s| s.is_empty()), Some(true), "{case}");
        assert_eq!(streams.get(1).map(|s| s.is_empty()), Some(false), "{case}");
        assert_eq!(streams.get(2).map(|s| s.is_empty()), Some(true), "{case}");
        assert_eq!(streams.get(3).map(|s| s.is_empty()), None, "{case}");

        let case = "after removing first block stream";
        streams.remove();
        assert_eq!(streams.get(0).map(|s| s.is_empty()), None, "{case}");
        assert_eq!(streams.get(1).map(|s| s.is_empty()), Some(false), "{case}");
        assert_eq!(streams.get(2).map(|s| s.is_empty()), Some(true), "{case}");
        assert_eq!(streams.get(3).map(|s| s.is_empty()), None, "{case}");

        let case = "after removing all block streams";
        streams.remove();
        streams.remove();
        assert_eq!(streams.get(0).map(|s| s.is_empty()), None, "{case}");
        assert_eq!(streams.get(1).map(|s| s.is_empty()), None, "{case}");
        assert_eq!(streams.get(2).map(|s| s.is_empty()), None, "{case}");
        assert_eq!(streams.get(3).map(|s| s.is_empty()), None, "{case}");
    }

    #[test]
    fn blockstreams_append() {
        // This function is tested indirectly via blockstreams macro.
        // Checking only for uncommon cases here, including errors.
        let case = "spill on a first append succeeds";
        let spilled =
            test_generate_blockstream(&mut (1..).peekable(), 255, TestGenerateFlags::Spilled as u8);
        let mut streams = BlockStreams::new();
        streams.append(spilled.into()).expect(case);

        let case = "spill following blockstream without trail fails";
        let partial =
            test_generate_blockstream(&mut (1..).peekable(), 255, TestGenerateFlags::Empty as u8);
        let spilled =
            test_generate_blockstream(&mut (1..).peekable(), 255, TestGenerateFlags::Spilled as u8);
        let mut streams = BlockStreams::new();
        streams.append(partial.into()).unwrap();
        streams.append(spilled.into()).expect_err(case);
    }

    #[test]
    #[should_panic(expected = "append uninitialized, or with trail too large")]
    fn blockstreams_append_uninitialized() {
        let mut streams = BlockStreams::new();
        streams
            .append(Arc::new(block::Stream::new(TestMemoryBlocks::new())))
            .unwrap();
    }

    #[test]
    fn blockstreams_remove() {
        let case = "remove basic";
        let mut streams = BlockStreams::try_from(blockstreams!(ff, ft, sp, fe, fe)).unwrap();
        assert_eq!(streams.removed(), (0, 0), "{case}");
        streams.remove().expect(case);
        assert_eq!(streams.removed(), (1, 1280), "{case}");
        streams.remove().expect(case);
        assert_eq!(streams.removed(), (2, 1280 + 1280 + 16), "{case}");
        streams.remove().expect(case);
        assert_eq!(streams.removed(), (3, 1280 + 1280 + 16 + 1136), "{case}");
        streams.remove().expect(case);
        assert_eq!(streams.removed(), (4, 1280 + 1280 + 16 + 1136), "{case}");
        streams.remove().expect(case);
        assert_eq!(streams.removed(), (5, 1280 + 1280 + 16 + 1136), "{case}");

        let case = "remove nothing";
        assert!(streams.remove().is_none(), "{case}");
        assert_eq!(streams.removed(), (5, 1280 + 1280 + 16 + 1136), "{case}");
    }

    #[test]
    fn blockstreams_pick_empty() {
        let case = "single empty should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(0), "{case}");

        let case = "empty following partial stream should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, fp, fe, fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(4), "{case}");

        let case = "empty following full stream should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, ff, fe, fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(4), "{case}");

        let case = "empty streams in the front should be skipped";
        let streams = BlockStreams::try_from(blockstreams!(fe, fe, ff, fe, fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(3), "{case}");

        let case = "empty streams in the middle should be skipped";
        let streams = BlockStreams::try_from(blockstreams!(ff, fe, fe, ff, fp, fe, fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(5), "{case}");

        let case = "no streams returns nothing";
        let streams = BlockStreams::<TestMemoryBlocks>::new();
        assert_eq!(streams.pick_empty(), None, "{case}");

        let case = "single non-empty stream returns nothing";
        let streams = BlockStreams::try_from(blockstreams!(fp)).unwrap();
        assert_eq!(streams.pick_empty(), None, "{case}");
        let streams = BlockStreams::try_from(blockstreams!(ff)).unwrap();
        assert_eq!(streams.pick_empty(), None, "{case}");

        let case = "no empty streams following non-empty returns nothing";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fe, fe, fp)).unwrap();
        assert_eq!(streams.pick_empty(), None, "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fe, fe, ff)).unwrap();
        assert_eq!(streams.pick_empty(), None, "{case}");

        let case = "removing streams keeps the index stable";
        let mut streams = BlockStreams::try_from(blockstreams!(ff, fp, ff, fp, fe)).unwrap();
        assert_eq!(streams.pick_empty(), Some(4), "{case}");
        streams.remove();
        assert_eq!(streams.pick_empty(), Some(4), "{case}");
    }

    #[test]
    fn blockstreams_pick_ending() {
        let case = "single empty should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(0), "{case}");

        let case = "single non-empty should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fp)).unwrap();
        assert_eq!(streams.pick_ending(), Some(0), "{case}");

        let case = "last non-empty should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, fp)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fe, fp)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, fp, fe, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");

        let case = "first empty should be picked";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, ff, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(4), "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, ff, fe, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(4), "{case}");

        let case = "empty streams in the front should be skipped";
        let streams = BlockStreams::try_from(blockstreams!(fe, fe, ff, fe, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fe, fe, ff, fp, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");

        let case = "empty streams in the middle should be skipped";
        let streams = BlockStreams::try_from(blockstreams!(ff, fp, fe, fe, ff, fp, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(5), "{case}");

        let case = "empty streams returns nothing";
        let streams = BlockStreams::<TestMemoryBlocks>::new();
        assert_eq!(streams.pick_ending(), None, "{case}");

        let case = "single full stream returns nothing";
        let streams = BlockStreams::try_from(blockstreams!(ff)).unwrap();
        assert_eq!(streams.pick_ending(), None, "{case}");

        let case = "full last stream returns nothing";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fp, ff)).unwrap();
        assert_eq!(streams.pick_ending(), None, "{case}");
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fe, ff)).unwrap();
        assert_eq!(streams.pick_ending(), None, "{case}");

        let case = "removing streams keeps the index stable";
        let mut streams = BlockStreams::try_from(blockstreams!(ff, fp, ff, fp, fe)).unwrap();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");
        streams.remove();
        assert_eq!(streams.pick_ending(), Some(3), "{case}");
    }

    #[test]
    fn blockstreams_stream_at() {
        let streams =
            BlockStreams::try_from(blockstreams!(ft, sp, fp, fe, fe, ft, sp, fe)).unwrap();

        let case = "zero offset";
        assert_eq!(streams.stream_at(0), Some((0, 0)), "{case}");

        let case = "offset within the first stream";
        assert_eq!(streams.stream_at(1120), Some((0, 1120)), "{case}");

        let case = "offset within the first stream span";
        assert_eq!(streams.get(0).unwrap().data().len(), 1152, "{case}");
        assert_eq!(streams.stream_at(1192), Some((0, 1192)), "{case}");

        let case = "offset within the second stream";
        assert_eq!(streams.stream_at(1296 + 16), Some((1, 16)), "{case}");

        let case = "offset aligned to the start of the third stream";
        assert_eq!(streams.stream_at(1296 + 1136), Some((2, 0)), "{case}");

        let case = "offset skipping empty streams";
        assert_eq!(streams.stream_at(3584 + 16), Some((5, 16)), "{case}");

        let case = "offset within the sixths stream span";
        assert_eq!(streams.get(5).unwrap().data().len(), 1152, "{case}");
        assert_eq!(streams.stream_at(3584 + 1192), Some((5, 1192)), "{case}");

        let case = "offset aligned to the end of the last stream";
        assert_eq!(streams.get(6).unwrap().data().len(), 1136, "{case}");
        assert_eq!(streams.stream_at(4880 + 1136), Some((6, 1136)), "{case}");

        let case = "offset past the end of the last stream";
        assert_eq!(streams.get(6).unwrap().data().len(), 1136, "{case}");
        assert_eq!(streams.stream_at(4880 + 1280), None, "{case}");
    }

    #[test]
    fn blockstreams_stream_at_end_aligned() {
        let case = "end of a partial stream";
        let streams = BlockStreams::try_from(blockstreams!(fp, fp, fe)).unwrap();
        assert_eq!(streams.stream_at(1152 + 1152), Some((1, 1152)), "{case}");

        let case = "end of a full stream with empty";
        let streams = BlockStreams::try_from(blockstreams!(fp, ff, fe)).unwrap();
        assert_eq!(streams.stream_at(1152 + 1280), Some((2, 0)), "{case}");

        let case = "end of a full stream without empty";
        let streams = BlockStreams::try_from(blockstreams!(fp, fp, ff)).unwrap();
        assert_eq!(streams.stream_at(2304 + 1280), Some((3, 0)), "{case}");

        let case = "start of a spilled stream";
        let streams = BlockStreams::try_from(blockstreams!(fp, ft, se)).unwrap();
        assert_eq!(streams.stream_at(2304 + 144), Some((2, 0)), "{case}");
    }

    #[test]
    fn blockstreams_stream_at_removed() {
        let mut streams =
            BlockStreams::try_from(blockstreams!(ft, sp, fp, fe, fe, ft, sp, fe)).unwrap();
        streams.remove();

        let case = "offset within removed stream";
        assert_eq!(streams.stream_at(0), None, "{case}");
        assert_eq!(streams.stream_at(16), None, "{case}");

        let case = "offset at the start of the first stream";
        assert_eq!(streams.stream_at(1296), Some((1, 0)), "{case}");

        let case = "offset within the first stream";
        assert_eq!(streams.stream_at(1296 + 16), Some((1, 16)), "{case}");

        let case = "offset aligned to the start of the second stream";
        assert_eq!(streams.stream_at(1296 + 1136), Some((2, 0)), "{case}");

        let case = "offset skipping empty streams";
        assert_eq!(streams.stream_at(3584 + 16), Some((5, 16)), "{case}");

        streams.remove();
        streams.remove();
        let case = "zero offset with empty streams at the start";
        assert_eq!(streams.stream_at(3583), None, "{case}");
        assert_eq!(streams.stream_at(3584), Some((5, 0)), "{case}");
    }

    #[test]
    fn blockstreams_offset_at() {
        // Lengths of different kinds of block streams, where TS is the buffer.
        const FT: usize = 1152;
        const FP: usize = 1152;
        const SF: usize = 1280 - 16;
        const SP: usize = 1152 - 16;
        const TS: usize = 128 + 16;

        let mut streams =
            BlockStreams::try_from(blockstreams!(fp, ft, sp, fp, fe, ft, sf, fe)).unwrap();
        streams.remove();
        let case = "offset of monotonically increasing numbers";
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 144));
        assert_eq!(offset, Err(streams.removed().1), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 145));
        assert_eq!(offset, Ok(FP), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 146));
        assert_eq!(offset, Ok(FP + 8), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 200));
        assert_eq!(offset, Ok(FP + 440), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 288));
        assert_eq!(offset, Ok(FP + FT - 8), "{case}");

        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 289));
        assert_eq!(offset, Ok(FP + FT + TS), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 290));
        assert_eq!(offset, Ok(FP + FT + TS + 8), "{case}");

        let expected = FP + FT + TS + SP + FP - 16;
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 573));
        assert_eq!(offset, Ok(expected), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 574));
        assert_eq!(offset, Ok(expected + 8), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 575));
        assert_eq!(offset, Ok(expected + 16), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 576));
        assert_eq!(offset, Ok(expected + 24), "{case}");

        let expected = FP + FT + TS + SP + FP + FT - 8;
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 718));
        assert_eq!(offset, Ok(expected), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 719));
        assert_eq!(offset, Ok(expected + 8 + TS), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 720));
        assert_eq!(offset, Ok(expected + 8 + TS + 8), "{case}");

        let expected = FP + FT + TS + SP + FP + FT + TS + SF - 16;
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 875));
        assert_eq!(offset, Ok(expected), "{case}");
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 876));
        assert_eq!(offset, Ok(expected + 8), "{case}");

        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 877));
        assert_eq!(offset, Err(streams.removed().1 + streams.len().1), "{case}");

        let case = "offset from the last block";
        let offset = streams.offset_at(|bytes, last| {
            let word = u64::from_le_bytes(last[..8].try_into().unwrap());
            test_find_word_in_stream(bytes, word)
        });
        let expected = FP + FT + TS + SP + FP + FT + TS + SF - 8 * 10;
        assert_eq!(offset, Ok(expected), "{case}");
        let offset = streams.offset_at(|bytes, last| {
            let word = u64::from_le_bytes(last[last.len() - 8..last.len()].try_into().unwrap());
            test_find_word_in_stream(bytes, word)
        });
        let expected = FP + FT + TS + SP + FP + FT + TS + SF - 8;
        assert_eq!(offset, Ok(expected), "{case}");

        let case = "offset of a repeating word";
        let offset =
            streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, TEST_WORDS_REPEATING[0]));
        assert_eq!(offset, Ok(FP + FT), "{case}");

        let case = "offset from the last block from a buffer";
        let streams = BlockStreams::try_from(blockstreams!(fp, ft, se)).unwrap();
        let offset = streams.offset_at(|bytes, last| {
            let word = u64::from_le_bytes(last[..8].try_into().unwrap());
            test_find_word_in_stream(bytes, word)
        });
        assert_eq!(offset, Ok(FP + FT), "{case}");

        // NOTE: This also indirectly tests unaligned appends.
        let case = "offset of repeating increasing numbers";
        let streams = BlockStreams::try_from(blockstreams!(fe, fe)).unwrap();
        let stream = streams.get(0).unwrap();
        stream.append(test_words_as_bytes(&[1_u64; 8])).unwrap();
        stream.append(test_words_as_bytes(&[2_u64; 8])).unwrap();
        stream.append(test_words_as_bytes(&[3_u64; 8])).unwrap();
        stream.append(test_words_as_bytes(&[4_u64; 8])).unwrap();
        stream.sync().unwrap();
        let offset = streams.offset_at(|bytes, last| {
            let word = u64::from_le_bytes(last[last.len() - 8..last.len()].try_into().unwrap());
            test_find_word_in_stream(bytes, word)
        });
        assert_eq!(offset, Ok(3 * 64), "{case}");

        let case = "offset within block stream with empty data";
        let mut streams = BlockStreams::new();
        let mut stream = block::Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        stream
            .append_with_opts(test_words_as_bytes(&[1_u64; 8]), true)
            .unwrap();
        stream.append(test_words_as_bytes(&[2_u64; 162])).unwrap();
        stream.sync().unwrap();
        assert!(stream.data().is_empty(), "{case}");
        streams.append(stream.into()).unwrap();
        let mut stream = block::Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        stream
            .append_with_opts(test_words_as_bytes(&[2_u64; 10]), true)
            .unwrap();
        stream.append(test_words_as_bytes(&[3_u64; 8])).unwrap();
        stream.append(test_words_as_bytes(&[4_u64; 8])).unwrap();
        stream.sync().unwrap();
        streams.append(stream.into()).unwrap();
        let offset = streams.offset_at(|bytes, _| test_find_word_in_stream(bytes, 4));
        assert_eq!(offset, Ok(162 * 8 + 8 * 8), "{case}");

        let case = "no block streams";
        let mut streams = BlockStreams::try_from(blockstreams!(fp)).unwrap();
        streams.remove();
        let offset = streams.offset_at(|_, _| unreachable!());
        assert_eq!(offset, Err(FP), "{case}");

        let case = "empty block streams";
        let mut streams = BlockStreams::try_from(blockstreams!(fp, fe)).unwrap();
        streams.remove();
        let offset = streams.offset_at(|_, _| unreachable!());
        assert_eq!(offset, Err(FP), "{case}");

        let case = "always search left";
        let streams = BlockStreams::try_from(blockstreams!(ft, se, ff)).unwrap();
        let offset = streams.offset_at(|_, _| SearchControl::SearchLeft);
        assert_eq!(offset, Err(0), "{case}");

        let case = "search left within buffer";
        let streams = BlockStreams::try_from(blockstreams!(ft, se, ff)).unwrap();
        let offset = streams.offset_at(|bytes, _| {
            if u64::from_le_bytes(bytes[..8].try_into().unwrap()) > 144 {
                SearchControl::SearchLeft
            } else {
                SearchControl::SearchRight
            }
        });
        assert_eq!(offset, Err(FT), "{case}");

        let case = "search left within non-first block stream";
        let streams = BlockStreams::try_from(blockstreams!(ft, se, ff)).unwrap();
        let offset = streams.offset_at(|bytes, _| {
            let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            if word == TEST_WORDS_REPEATING[0] || word <= 164 {
                SearchControl::SearchRight
            } else {
                SearchControl::SearchLeft
            }
        });
        assert_eq!(offset, Err(FT + TS), "{case}");
    }

    #[test]
    fn blockstreams_set_buffer() {
        macro_rules! get_buffer {
            ($streams:expr, $pos:expr) => {{
                $streams.buffers.get_mut().get_mut($pos).unwrap().as_ref()
            }};
        }

        let mut streams = BlockStreams::try_from(blockstreams!(fp, ft, st, sp, fp)).unwrap();

        let case = "set with empty data";
        let updated = unsafe { streams.set_buffer(0, &[], &[]) };
        assert!(!updated, "{case}");
        let buffer = get_buffer!(streams, 0);
        assert!(buffer.is_none(), "{case}");

        let case = "set on an empty index";
        let updated = unsafe { streams.set_buffer(0, &[0x01, 0x02], &[0x03, 0x04]) };
        assert!(updated, "{case}");
        let buffer = get_buffer!(streams, 0).expect(case);
        assert_eq!(buffer.as_slice(), &[0x01, 0x02, 0x03, 0x04], "{case}");

        let case = "no overwrite on non-empty index";
        let updated = unsafe { streams.set_buffer(1, &[0x01, 0x02], &[0x03, 0x04]) };
        assert!(!updated, "{case}");
        let buffer = get_buffer!(streams, 1).expect(case);
        assert_eq!(buffer.as_slice(), &TEST_BYTES_REPEATING[..144], "{case}");

        let case = "set after append";
        let mut stream = block::Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        streams.append(Arc::new(stream)).unwrap();
        let updated = unsafe { streams.set_buffer(3, &[0x05, 0x06], &[0x07, 0x08]) };
        assert!(updated, "{case}");
        let buffer = get_buffer!(streams, 0).expect(case);
        assert_eq!(buffer.as_slice(), &[0x01, 0x02, 0x03, 0x04], "{case}");
        let buffer = get_buffer!(streams, 1).expect(case);
        assert_eq!(buffer.as_slice(), &TEST_BYTES_REPEATING[..144], "{case}");
        let buffer = get_buffer!(streams, 2).expect(case);
        assert_eq!(buffer.as_slice(), &TEST_BYTES_REPEATING[..144], "{case}");
        let buffer = get_buffer!(streams, 3).expect(case);
        assert_eq!(buffer.as_slice(), &[0x05, 0x06, 0x07, 0x08], "{case}");
        let buffer = get_buffer!(streams, 4);
        assert!(buffer.is_none(), "{case}");

        let case = "set after remove";
        streams.remove();
        let updated = unsafe { streams.set_buffer(4, &[0x09, 0x10], &[0x11, 0x12]) };
        assert!(updated, "{case}");
        let buffer = get_buffer!(streams, 0).expect(case);
        assert_eq!(buffer.as_slice(), &TEST_BYTES_REPEATING[..144], "{case}");
        let buffer = get_buffer!(streams, 1).expect(case);
        assert_eq!(buffer.as_slice(), &TEST_BYTES_REPEATING[..144], "{case}");
        let buffer = get_buffer!(streams, 2).expect(case);
        assert_eq!(buffer.as_slice(), &[0x05, 0x06, 0x07, 0x08], "{case}");
        let buffer = get_buffer!(streams, 3).expect(case);
        assert_eq!(buffer.as_slice(), &[0x09, 0x10, 0x11, 0x12], "{case}");
        let buffer = get_buffer!(streams, 4);
        assert!(buffer.is_none(), "{case}");
    }

    #[test]
    #[should_panic(expected = "attempt to subtract with overflow")]
    fn blockstreams_set_buffer_index_too_small() {
        let mut streams = BlockStreams::try_from(blockstreams!(fp, fp, fp)).unwrap();
        streams.remove();
        unsafe { streams.set_buffer(0, &[], &[]) };
    }

    #[test]
    #[should_panic(expected = "index too large")]
    fn blockstreams_set_buffer_index_too_large() {
        let streams = BlockStreams::try_from(blockstreams!(fp, fp, fp)).unwrap();
        unsafe { streams.set_buffer(2, &[], &[]) };
    }

    #[test]
    #[should_panic(expected = "trailing and spilled should be either empty or non-empty")]
    fn blockstreams_set_buffer_no_spilled() {
        let streams = BlockStreams::try_from(blockstreams!(fp, fp, fp)).unwrap();
        unsafe { streams.set_buffer(0, &[0x01], &[]) };
    }

    #[test]
    #[should_panic(expected = "trailing and spilled should be either empty or non-empty")]
    fn blockstreams_set_buffer_no_trailing() {
        let streams = BlockStreams::try_from(blockstreams!(fp, fp, fp)).unwrap();
        unsafe { streams.set_buffer(0, &[], &[0x01]) };
    }

    #[test]
    fn make_span_buffer_fn() {
        let case = "empty span makes no buffer";
        assert_eq!(make_span_buffer(&[], &[]).expect(case), None, "{case}");

        let case = "span makes a buffer";
        let bytes = make_span_buffer(&[0x01, 0x02], &[0x03]).expect(case);
        assert_eq!(bytes.as_deref().expect(case), &[0x01, 0x02, 0x03], "{case}");

        let case = "partial span results in error";
        make_span_buffer(&[0x01, 0x02], &[]).expect_err(case);
        make_span_buffer(&[], &[0x01, 0x02]).expect_err(case);
    }

    static TEST_WORDS_REPEATING: [u64; 170] = [0xcccccccc_u64.to_le(); 170];
    static TEST_BYTES_REPEATING: &'static [u8] = test_words_as_bytes(&TEST_WORDS_REPEATING);

    #[inline(always)]
    const fn test_words_as_bytes(words: &[u64]) -> &[u8] {
        unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 8) }
    }

    #[repr(u8)]
    enum TestGenerateFlags {
        Empty = 0b0000,
        Spilled = 0b0001,
        Trailing = 0b0010,
    }

    fn test_generate_blockstream(
        iter: &mut core::iter::Peekable<impl Iterator<Item = u64>>,
        take: usize,
        flags: u8,
    ) -> block::Stream<TestMemoryBlocks> {
        let mut stream = block::Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        if flags & TestGenerateFlags::Spilled as u8 != 0 {
            stream
                .append_with_opts(&TEST_BYTES_REPEATING[..16], /*spilled=*/ true)
                .unwrap();
        }
        for _ in 0..take {
            let word = iter.peek().unwrap();
            match stream.append(&word.to_le_bytes()) {
                Ok(_) => {
                    iter.next();
                    continue;
                }
                Err(block::StreamError::Full) => break,
                Err(_) => unreachable!(),
            }
        }
        if flags & TestGenerateFlags::Trailing as u8 != 0 {
            stream.append(&TEST_BYTES_REPEATING).unwrap();
        }
        stream.sync().unwrap();
        stream
    }

    fn test_find_word_in_stream(bytes: &[u8], value: u64) -> SearchControl {
        assert!(!bytes.is_empty());
        for i in 0..bytes.len() >> 3 {
            let word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            if word == value {
                return SearchControl::Match(i * 8);
            } else if word == TEST_WORDS_REPEATING[0] {
                return SearchControl::SearchRight;
            } else if word > value {
                return SearchControl::SearchLeft;
            }
        }
        SearchControl::SearchRight
    }

    struct TestMemoryBlocksAllocator(Mutex<Vec<Option<TestMemoryBlocks>>>);

    unsafe impl BlocksAllocator for TestMemoryBlocksAllocator {
        type Blocks = TestMemoryBlocks;

        fn alloc(&self) -> io::Result<Self::Blocks> {
            let mut allocated = self.0.lock().unwrap();
            allocated.push(Some(TestMemoryBlocks::new()));
            Ok(unsafe { allocated.last_mut().unwrap().as_mut().unwrap().refclone() })
        }

        fn release(&self, blocks: Self::Blocks) -> Result<(), (Self::Blocks, io::Error)> {
            let bytes = blocks.0.get_ref();
            for blocks in self.0.lock().unwrap().iter_mut() {
                if blocks
                    .as_ref()
                    .is_some_and(|blocks| blocks.0.get_ref().as_ptr() == bytes.as_ptr())
                {
                    blocks.take();
                    return Ok(());
                }
            }
            Err((blocks, io::Error::from(io::ErrorKind::Other)))
        }

        fn retrieve(&self, mut f: impl FnMut(Self::Blocks)) -> io::Result<()> {
            for blocks in self.0.lock().unwrap().iter_mut() {
                if blocks.is_some() {
                    f(unsafe { blocks.as_mut().unwrap().refclone() });
                }
            }
            Ok(())
        }
    }

    impl From<VecDeque<block::Stream<TestMemoryBlocks>>> for TestMemoryBlocksAllocator {
        fn from(mut value: VecDeque<block::Stream<TestMemoryBlocks>>) -> Self {
            let mut blocks = Vec::with_capacity(value.len());
            for stream in value.drain(..) {
                blocks.push(Some(stream.into_inner()));
            }
            Self(blocks.into())
        }
    }

    struct TestMemoryBlocks(TestMaybeDrop<io::Cursor<Vec<u8>>>);

    impl TestMemoryBlocks {
        const BLOCK_COUNT: usize = 16;
        const BLOCK_SHIFT: u32 = 7;
        const BLOCK_SIZE: usize = 1 << Self::BLOCK_SHIFT;
        const SIZE: usize = Self::BLOCK_COUNT * Self::BLOCK_SIZE;

        #[inline(always)]
        fn new() -> Self {
            Self(TestMaybeDrop::Normal(io::Cursor::new(vec![0; Self::SIZE])))
        }

        unsafe fn refclone(&mut self) -> Self {
            let bytes = self.0.get_mut();
            Self(TestMaybeDrop::Manual(mem::ManuallyDrop::new(
                io::Cursor::new(Vec::from_raw_parts(
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    bytes.capacity(),
                )),
            )))
        }
    }

    impl Blocks for TestMemoryBlocks {
        #[inline(always)]
        fn block_count(&self) -> u64 {
            Self::BLOCK_COUNT as u64
        }

        #[inline(always)]
        fn block_shift(&self) -> u32 {
            Self::BLOCK_SHIFT
        }

        #[inline(always)]
        fn load_from(&mut self, block: u64, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<()> {
            self.0
                .seek(io::SeekFrom::Start(block << Self::BLOCK_SHIFT))?;
            for buf in bufs {
                self.0.read_exact(buf)?;
            }
            Ok(())
        }

        #[inline(always)]
        fn store_at(&mut self, block: u64, bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
            self.0
                .seek(io::SeekFrom::Start(block << Self::BLOCK_SHIFT))?;
            for buf in bufs {
                self.0.write_all(buf)?;
            }
            Ok(())
        }
    }

    impl Drop for TestMemoryBlocks {
        #[inline(always)]
        fn drop(&mut self) {
            if let TestMaybeDrop::Manual(ref mut cursor) = self.0 {
                // Drop the cursor, but not the underlying vector.
                let cursor = unsafe { mem::ManuallyDrop::take(cursor) };
                mem::forget(cursor.into_inner());
            }
        }
    }

    enum TestMaybeDrop<T> {
        Normal(T),
        Manual(mem::ManuallyDrop<T>),
    }

    impl<T> core::ops::Deref for TestMaybeDrop<T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &T {
            match self {
                Self::Normal(value) => value,
                Self::Manual(value) => value,
            }
        }
    }

    impl<T> core::ops::DerefMut for TestMaybeDrop<T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut T {
            match self {
                Self::Normal(value) => value,
                Self::Manual(value) => value,
            }
        }
    }
}
