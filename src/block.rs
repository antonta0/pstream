//! A stream over a fixed number of fixed-sized blocks.
//!
//! The primary problem that [`Stream`] solves is small and variable-sized
//! record writes over larger fixed-sized blocks, described by [`Blocks`] trait.
//! The rest is more or less a consequence of that.
//!
//! This implementation uses each block to hold metadata. This adds an extra
//! overhead of in-memory copy during writes, yet provides better integrity,
//! allows for more efficient navigation, and keeps the IO part as basic as
//! possible, which is easier to reason about. The layout can be used for a
//! binary search, or may be even parallel computations over data, as it is
//! possible to find the start of a write for any given block.
//!
//! This abstraction is generic enough to work over block devices without
//! a filesystem, or with conventional files.
//!
//! # Things to note
//!
//! The whole stream is loaded as a contiguous memory. There is no page caching.
//!
//! Metadata consumes 48 bytes from every block, consider this when calculating
//! the effective capacity of the stream.
//!
//! I haven't thought of using this with large records much. It may not be the
//! best choice for that scenario and it may be more wise to use files or
//! another more specialized structure after a certain point, yet this stream
//! can still hold the metadata. What is large specifically? Depends on the
//! hardware and block size, but just to throw some random number, everything
//! larger than 8M per record should be considered carefully.

// In most cases the cast is from u64 to usize, which is ensured to not truncate
// by a cfg below. Exception is `AboutBlock`, where the casts are meant to
// truncate the values.
#![allow(clippy::cast_possible_truncation)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("block stream works only on 64-bit platforms");

use core::{cell::UnsafeCell, cmp, ops::Range, ops::RangeInclusive, sync::atomic};

use std::io;

/// Something that can be subdivided into blocks, each being a single unit of
/// operation.
pub trait Blocks {
    /// The number of blocks available. The returned value must be fixed for
    /// the whole lifetime of [`Stream`].
    fn block_count(&self) -> u64;

    /// The size of a single block expressed as a power of two. This value must
    /// be fixed for the whole lifetime of [`Stream`].
    fn block_shift(&self) -> u32;

    /// Loads the contents of blocks into `bufs` starting from `block`.
    ///
    /// This function must fill the `bufs` completely and otherwise return an
    /// error. The contents of `bufs` are not specified and must not be checked
    /// by the implementation.
    ///
    /// The current [`Stream`] implementation guarantees that the total length
    /// of all `bufs` is divisible by block size.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    fn load_from(&mut self, block: u64, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<()>;

    /// Stores the contents of `bufs` to blocks starting at `block`.
    ///
    /// This function must ensure that the `bufs` actually made it to the
    /// underlying blocks. That is, every store call must sync data to disk.
    ///
    /// The current [`Stream`] implementation guarantees the following:
    /// -   There is an even number of `bufs`.
    /// -   Every pair within `bufs` adds up to block size, except for the last
    ///     pair, where it can be smaller.
    /// -   It is never called for the same block after that block has been
    ///     written in full.
    /// -   Incomplete blocks are always overwritten from the start until
    ///     complete.
    ///
    /// If store fails for any reason, it could be retried with exactly the
    /// same state.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    fn store_at(&mut self, block: u64, bufs: &mut [io::IoSlice<'_>]) -> io::Result<()>;
}

/// Size of the meta section of each block, with some room for future
/// use, aligned to 8-byte word.
///
/// First 8 bytes is [`AboutBlock`], followed by 4 bytes of parity of
/// `AboutBlock` and 4 bytes of CRC32 of the written data section of the block.
const META_SECTION_SIZE_BYTES: u64 = 48;

macro_rules! meta_offset {
    ($block:expr) => {
        // Not sure if compiler is optimizing around the fact that 48 = 32 + 16
        ($block << 5) + ($block << 4)
    };
}

macro_rules! meta_about_section {
    ($block:expr) => {
        meta_offset!($block)..meta_offset!($block) + 8
    };
}

macro_rules! meta_parity_section {
    ($block:expr) => {
        meta_offset!($block) + 8..meta_offset!($block) + 12
    };
}

macro_rules! meta_crc32_section {
    ($block:expr) => {
        meta_offset!($block) + 12..meta_offset!($block) + 16
    };
}

macro_rules! meta_section {
    ($block:expr) => {
        meta_offset!($block)..meta_offset!($block + 1)
    };
}

macro_rules! data_offset {
    ($shift:expr, $block:expr) => {
        // This is about 2x faster than $block * data_section_size and
        // is frequently used, so worth the hassle.
        ($block << $shift) - meta_offset!($block)
    };
}

macro_rules! data_section {
    ($shift:expr, $block:expr) => {
        data_offset!($shift, $block)..data_offset!($shift, $block + 1)
    };
}

/// A fixed-size stream implemented on top of [`Blocks`] with serial writes and
/// wait-free read access.
///
/// Memory-wise, the current implementation allocates memory once when created
/// and maps it directly to the underlying blocks. Every block takes 48 bytes
/// to keep metadata, which is necessary to verify data integrity, recover
/// after restarts, and efficiently navigate the stream. Data buffer is exposed
/// as a byte slice to readers and is append-only. Metadata, however, may be
/// rewritten on a block that has not been filled yet.
///
/// The implementation is not optimized for more than one writer, although
/// appends are quite efficient. The strategy for multiple writers could be
/// to have a writer per `Stream` and then merge the results via an iterator.
/// That depends on the type of the data written to the stream, however. Reads
/// are guaranteed to return only the data that has been successfully synced
/// to blocks.
///
/// `Blocks` must be zero-initialized.
#[doc(alias = "blockstream")]
#[derive()] // manual: Debug
pub struct Stream<B> {
    /// A series of blocks.
    blocks: UnsafeCell<B>,
    /// Size of the data section of each block.
    data_section_size: u64,

    // A buffer of meta and data sections of each block are represented as a
    // contiguous memory each. On the block device side, however, each block
    // has a meta and a data section both of which sum up to
    // `1 << B::block_shift`.
    //
    // Let's say '.' is 1 byte, 'M' is meta and 'D' is data, block size 4.
    //
    // Blocks: |....|....|....|....|....|
    //         |MDDD|MDDD|MDDD|MDDD|MDDD|
    //          *    *    *    *    *
    //          | +--|    |    |    |
    //          | | +-----+    |    |
    //          | | | +--------+    |
    //          | | | | +-----------+
    //          | | | | |
    //          ^ ^ ^ ^ ^
    // Memory: |M|M|M|M|M|
    //         |DDD|DDD|DDD|DDD|DDD|
    //
    // Alternative is to keep memory and disk layout identical:
    // 1.   Keep all meta sections in first blocks, data following later:
    //      Blocks: |MMMM|M...|DDDD|DDDD|DDDD|DDD.|
    // 2.   In-memory layout to be a 1:1 map of what's in blocks:
    //      Memory: |MDDD|MDDD|MDDD|MDDD|MDDD|
    //
    // In the first case, each sync to disk requires two IO operations at least.
    // In the second case, reading data may require stitching it in-memory on
    // demand if it is split between blocks.
    //
    // The selected layout is kind-of a hybrid between the two. The only downside
    // is that it requires to make a copy when loading or flushing data to
    // disk, be it through OS page cache or a direct IO implementation.
    //
    // The copy is generally faster, than doing two IO writes if meta and data
    // are split on disk. A random IO write on NVMe takes somewhere around 1ms
    // (based on a fast consumer grade NVMe), while an in-memory copy with
    // equivalent latency is for buffers between 16M and 32M large (see bench
    // results below). I expect that small writes will be predominant,
    // so the selected layout seems to be a better choice than alternative 1.
    //
    // Comparing to alternative 2, the selected layout organizes data in-memory
    // for more efficient reads when loading from disk, which is a one-time
    // operation. After loading, no allocations are needed for accessing data,
    // making access time predictable and stable in the expense of an extra
    // memory copy during writes, which is not that slow when comparing to disk
    // access times anyway.
    //
    // A very simple micro-benchmark of a buffer copy with `std::vec::Vec` on
    // a fairly fast x86_64 desktop. 100 iterations each, so divide by 100 to
    // get an average per iteration (some numbers are funny, I don't know why,
    // could be due to alignment or some cache optimizations..).
    // test copy_1k_linear   ... bench:         724 ns/iter (+/- 9)
    // test copy_2k_linear   ... bench:       1,505 ns/iter (+/- 4)
    // test copy_4k_linear   ... bench:      92,429 ns/iter (+/- 460)
    // test copy_8k_linear   ... bench:     239,741 ns/iter (+/- 2,145)
    // test copy_16k_linear  ... bench:     359,037 ns/iter (+/- 6,456)
    // test copy_32k_linear  ... bench:     770,195 ns/iter (+/- 13,656)
    // test copy_64k_linear  ... bench:   1,498,241 ns/iter (+/- 9,152)
    // test copy_128k_linear ... bench:     175,126 ns/iter (+/- 463)
    // test copy_256k_linear ... bench:   5,953,646 ns/iter (+/- 24,969)
    // test copy_512k_linear ... bench:     949,082 ns/iter (+/- 9,166)
    // test copy_1m_linear   ... bench:   1,821,326 ns/iter (+/- 22,021)
    // test copy_2m_linear   ... bench:   3,633,795 ns/iter (+/- 14,292)
    // test copy_4m_linear   ... bench:   7,652,839 ns/iter (+/- 361,232)
    // test copy_8m_linear   ... bench:  15,176,197 ns/iter (+/- 74,131)
    // test copy_16m_linear  ... bench:  57,419,645 ns/iter (+/- 5,056,722)
    // test copy_32m_linear  ... bench: 161,909,575 ns/iter (+/- 1,913,318)
    /// A contiguous buffer of meta sections of `blocks`.
    meta_buffer: UnsafeCell<Vec<u8>>,
    /// A contiguous buffer of data sections of `blocks`.
    data_buffer: UnsafeCell<Vec<u8>>,

    /// The current position in `data_buffer`, indicating the offset up to
    /// which the data has been appended to the buffer.
    current: atomic::AtomicUsize,
    /// The synced position in `data_buffer`, indicating the offset up to
    /// which the data has been synced to the underlying blocks.
    synced: atomic::AtomicUsize,
    /// The trail position in `data_buffer`, which points to the start of the
    /// append that was not fully written, because the end of the stream was
    /// reached.
    trail: atomic::AtomicUsize,

    /// Whether the stream is locked. Used to synchronize appends, syncs and
    /// lock writes during reads where concurrent writes would be unsafe.
    locked: atomic::AtomicBool,
}

impl<B: Blocks> Stream<B> {
    /// Creates a new uninitialized instance of `Stream` from `Blocks`.
    /// Uninitialized stream cannot be appended to. [`Stream::initialize`]
    /// has to complete without errors before appending.
    ///
    /// # Panics
    ///
    /// Panics if `B` has zero `block_count` or when `block_shift` is less
    /// than 6 or greater than or equal to 28.
    #[must_use]
    pub fn new(blocks: B) -> Self {
        assert_ne!(blocks.block_count(), 0, "blocks have no blocks?");
        assert!(blocks.block_shift() >= 6, "block shift too small");
        assert!(
            blocks.block_shift() < AboutBlock::TRAIL_BITS,
            "block shift too big"
        );
        let data_section_size = (1 << blocks.block_shift()) - META_SECTION_SIZE_BYTES;
        let meta_buffer_len = (blocks.block_count() * META_SECTION_SIZE_BYTES) as usize;
        let data_buffer_len =
            (blocks.block_count() << blocks.block_shift()) as usize - meta_buffer_len;
        Self {
            blocks: blocks.into(),
            data_section_size,
            meta_buffer: vec![0; meta_buffer_len].into(),
            data_buffer: vec![0; data_buffer_len].into(),
            current: data_buffer_len.into(),
            synced: data_buffer_len.into(),
            trail: 0.into(),
            locked: false.into(),
        }
    }

    /// Returns the number of blocks of the underlying `Blocks`.
    #[inline(always)]
    #[must_use]
    pub fn block_count(&self) -> u64 {
        // SAFETY: Trait requires implementations that `block_count` is fixed.
        unsafe { &*self.blocks.get() }.block_count()
    }

    /// Returns the shift of the underlying `Blocks`.
    #[inline(always)]
    #[must_use]
    pub fn block_shift(&self) -> u32 {
        // SAFETY: Trait requires implementations that `block_shift` is fixed.
        unsafe { &*self.blocks.get() }.block_shift()
    }

    /// Initializes a stream from the current buffer state. This should be
    /// called after `load` to set the position for subsequent appends.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::MetaSectionCorrupted`] if metadata is
    /// inconsistent. In this case, more detailed status can be retrieved by
    /// running a more expensive [`Stream::verify`].
    ///
    /// Note, that the verification of the stream is not complete, and if some
    /// blocks in the middle are damaged it won't be noticed via this path.
    pub fn initialize(&mut self) -> Result<(), StreamError> {
        // SAFETY: Exclusive access is guaranteed by the borrow checker.
        let block = unsafe { self.find_ending_block() };
        // SAFETY: Exclusive access is guaranteed by the borrow checker.
        let about = unsafe { self.meta_about(block) };

        if about.is_complete() {
            if block != self.block_count() as usize - 1 {
                return Err(StreamError::MetaSectionCorrupted(
                    "complete block is not the last block",
                ));
            }
            let position = self.capacity();
            *self.current.get_mut() = position;
            *self.synced.get_mut() = position;
            *self.trail.get_mut() = position - self.trail_size(block, about)?;
            return Ok(());
        }

        // SAFETY: Exclusive access is guaranteed by the borrow checker.
        let parity = unsafe { self.meta_parity(block) };
        if (parity != 0 && about.is_empty() || !about.is_empty()) && !about.verify(parity) {
            return Err(StreamError::MetaSectionCorrupted("parity mismatch"));
        }

        if about.trail_bytes() >= self.data_section_size
            || about.spilled_bytes() >= self.data_section_size
        {
            return Err(StreamError::MetaSectionCorrupted(
                "offset larger than data section size",
            ));
        }
        if about.trail_bytes() != 0 && about.trail_bytes() <= about.spilled_bytes() {
            return Err(StreamError::MetaSectionCorrupted("inconsistent offsets"));
        }

        let position = data_offset!(self.block_shift(), block)
            + cmp::max(about.trail_bytes(), about.spilled_bytes()) as usize;

        // SAFETY: Exclusive access is guaranteed by the borrow checker.
        let crc32 = unsafe { self.meta_crc32(block) };
        // SAFETY: Exclusive access is guaranteed by the borrow checker.
        let crc32_data = unsafe {
            &(*self.data_buffer.get())[data_offset!(self.block_shift(), block)..position]
        };
        if crc32fast::hash(crc32_data) != crc32 {
            return Err(StreamError::MetaSectionCorrupted("crc32 mismatch"));
        }

        // Do a quick check of previous blocks, which are supposed to be
        // complete. Check the first block just before this one, and the
        // very first block of the stream, if there's one. This is a
        // lightweight verification to keep initialization quicker.
        // For a full verification `verify` should be used.
        if block != 0 {
            // Yes, this could check block 0 twice, but that's alright.
            for block in [0, block - 1] {
                // SAFETY: Exclusive access is guaranteed by the borrow checker.
                let about = unsafe { self.meta_about(block) };
                // SAFETY: Exclusive access is guaranteed by the borrow checker.
                if !about.is_complete() || !about.verify(unsafe { self.meta_parity(block) }) {
                    return Err(StreamError::MetaSectionCorrupted(
                        "preceding block(s) malformed",
                    ));
                }
            }
        }

        *self.current.get_mut() = position;
        *self.synced.get_mut() = position;
        *self.trail.get_mut() = self.capacity();
        Ok(())
    }

    /// Returns the block numbers where the data has been appended to, where
    /// the end block is the start of the last append. The returned range is
    /// safe to pass to [`Stream::data_range_for`] to get the actual data.
    #[must_use]
    pub fn data_block_range(&self) -> RangeInclusive<usize> {
        let data_range = self.data_range();
        let start_block = data_range.start / self.data_section_size as usize;
        let end_block = {
            let block = data_range.end / self.data_section_size as usize;
            if data_range.end % self.data_section_size as usize == 0 {
                block.saturating_sub(1)
            } else {
                block
            }
        };

        // SAFETY: Spilled bytes are guaranteed to be written correctly at
        // synced position, because it is written only once before syncing.
        let spilled = unsafe { self.meta_about(end_block) }.spilled_bytes();
        // If spilled bytes are covering only part of the capacity of this
        // block, or there are no spilled bytes, return this block.
        if spilled == 0
            || spilled != (data_range.end - data_offset!(self.block_shift(), end_block)) as u64
        {
            return start_block..=end_block;
        }
        // Otherwise search up until we find where the spill has started from.
        for block in (start_block..end_block).rev() {
            // SAFETY: These blocks have been written in full and are not
            // expected to be modified anymore, since they precede the block,
            // which was acquired via synced position.
            let about = unsafe { self.meta_about(block) };
            debug_assert!(about.is_complete(), "multiple synced incomplete blocks");
            // Since this is a complete block, first encounter of trail bytes
            // is the actual start of the trail from which the spill has started.
            if about.trail_bytes() != 0 {
                return start_block..=block;
            }
        }
        // If there is no trail section up there, just return the starting block.
        start_block..=start_block
    }

    /// Returns the offset range relative to the slice returned by `data` that
    /// maps to the provided block number. It is guaranteed that the first
    /// item at the start offset will correspond to the start of an append to the
    /// stream and the end to the end of the last append to that block. The end
    /// offset may be outside this block, if that block has a trail.
    ///
    /// If items appended to the stream are sequentially numbered, this
    /// function can be used to do a binary search within the stream by blocks.
    /// Keep in mind, however, when accessing data in concurrent setting, that
    /// either clamping the offsets to the actual size of previously returned
    /// data slice or calling this function before reading data slice is
    /// necessary.
    ///
    /// If data is empty or block is an empty block, a zero-sized offset range
    /// is returned.
    ///
    /// # Errors
    ///
    /// If this block is being spilled from previous blocks in full, then the
    /// error is returned, which includes the block number where the next append
    /// can be found, which is equal to `block_count` if the spill continues
    /// past this stream.
    ///
    /// If block is greater or equals to `block_count` an error with that block
    /// number is returned.
    pub fn data_range_for(&self, block: usize) -> Result<Range<usize>, usize> {
        if block >= self.block_count() as usize {
            return Err(block);
        }

        let range = self.data_range();
        // This also covers the case when there's a spill on the first block
        // and the block stream has not been synced.
        if range.is_empty() {
            return Ok(0..0);
        }

        // SAFETY: If block happens to be empty, it's all good. If not, the
        // complete flag and spilled bytes are guaranteed to be written
        // correctly, as they are never updated after advancing the synced
        // position. That is, if the synced position is at this block, the
        // spilled bytes has been written already. If the synced position is
        // past this block, this block has been marked as complete. Due to
        // concurrent access, it may happen, however, that the synced position
        // is at this block and it is marked as complete.
        let about = unsafe { self.meta_about(block) };
        if about.is_empty() {
            let offset = range.end - range.start;
            return Ok(offset..offset);
        }

        let spilled = about.spilled_bytes();
        if spilled > self.data_section_size {
            debug_assert!(about.is_complete(), "incomplete spilled block");
            return Err(cmp::min(
                self.block_count() as usize,
                block + (spilled / self.data_section_size) as usize,
            ));
        }

        let start = data_offset!(self.block_shift(), block) + spilled as usize;
        let block_end = data_offset!(self.block_shift(), block + 1);
        let end = if range.end > block_end {
            // If synced position is after the end of this block, then some
            // block after has to be incomplete, and therefore the current one
            // is complete. If not, something went terribly wrong.
            debug_assert!(about.is_complete(), "multiple synced incomplete blocks");

            // Exploit the invariants a little, and just extend the end of the
            // block with the spilled bytes from the following block. This works
            // because whether the next block has spilled bytes is tied directly
            // to the trail bytes value of this complete block. The literal edge
            // case is when the spill is past this block stream, so clamp the
            // end to the end of the data range.
            //
            // SAFETY: The synced position from which the data range was set
            // is past the end of this block, meaning that the spilled bytes
            // on the following block were written - this happens exactly once
            // during an append that spills to another block.
            let about = unsafe { self.meta_about(block + 1) };
            cmp::min(range.end, block_end + about.spilled_bytes() as usize)
        } else {
            range.end
        };

        Ok(start - range.start..end - range.start)
    }

    /// Appends `bytes` to the in-memory buffer of this block stream and
    /// returns the number of bytes written.
    ///
    /// If there's not enough memory left to write the whole `bytes`, a partial
    /// write will be made, which can be retrieved via [`Stream::trailing`].
    ///
    /// The `spilled` flag allows marking the first append as a spill, e.g.
    /// a continuation of an incomplete append of some other `Stream`, and it
    /// can be retrieved via [`Stream::spilled`].
    ///
    /// Note, that the size of a single append is limited by `1 GiB - 1` bytes
    /// due to the data layout on `Blocks`.
    ///
    /// # Errors
    ///
    /// Returns one of the following errors:
    ///
    /// -   [`StreamError::AppendTooLarge`] if the size of `bytes` exceeds the
    ///     limit mentioned above.
    /// -   [`StreamError::Busy`] if attempt to acquire a lock fails.
    /// -   [`StreamError::SpilledAfterFirstAppend`] if `spilled` is set, yet
    ///     the stream is not empty.
    /// -   [`StreamError::Full`] if the stream is at capacity, even for
    ///     zero-sized appends.
    ///
    /// In case of an error, the underlying buffer is left unchanged.
    #[inline(always)]
    pub fn append_with_opts(&self, bytes: &[u8], spilled: bool) -> Result<usize, StreamError> {
        Self::verify_append(bytes)?;
        // Try to acquire a lock to serialize appends. Do not block, instead
        // let the caller decide how to handle concurrent calls.
        let lock = Lock::try_acquire(&self.locked).ok_or(StreamError::Busy)?;
        self.append_with_opts_locked(bytes, spilled, &lock)
    }

    /// Pre-locked version of [`Stream::append_with_opts`].
    ///
    /// # Panics
    ///
    /// If the lock belongs to another block stream.
    fn append_with_opts_locked(
        &self,
        bytes: &[u8],
        spilled: bool,
        lock: &Lock,
    ) -> Result<usize, StreamError> {
        Self::verify_append(bytes)?;
        assert_eq!(
            lock.0 as *const atomic::AtomicBool,
            core::ptr::addr_of!(self.locked),
            "unrelated lock",
        );

        // Current position is updated only during the append, which is
        // protected by the lock above, so using Relaxed here.
        let current = self.current.load(atomic::Ordering::Relaxed);

        if current != 0 && spilled {
            return Err(StreamError::SpilledAfterFirstAppend);
        }
        if current == self.capacity() {
            return Err(StreamError::Full);
        }
        if bytes.is_empty() {
            return Ok(0);
        }

        // Append the data. It's just a simple copy.
        let ending_position = cmp::min(current + bytes.len(), self.capacity());
        let written = ending_position - current;
        // SAFETY: This is the only place where writes to data_buffer
        // are happening, and it is protected by the lock. The write is
        // append-only and does not conflict with concurrent reads.
        unsafe { &mut (*self.data_buffer.get())[current..ending_position] }
            .copy_from_slice(&bytes[..written]);

        // Update metadata of the starting block. Get the current meta, or
        // create a new one if exactly at the start of the block.
        let starting_block = current / self.data_section_size as usize;
        let ending_block = (ending_position - 1) / self.data_section_size as usize;
        let mut about = if current % self.data_section_size as usize == 0 {
            AboutBlock::new()
        } else {
            // SAFETY: Concurrent mutable access is protected by the lock.
            unsafe { self.meta_about(starting_block) }
        };
        // This should never happen, as this invariant is maintained during
        // initialization, and after that the stream is updated one way.
        assert!(!about.is_complete(), "append to a complete block");

        // If no new blocks were written, record the ending position relative
        // to this block in the trail_bytes. If the data lands exactly at the
        // end of this block, set it as complete. Additionally, if that's a
        // partial write, set trail_bytes to the actual trail of a complete block.
        //
        // Otherwise, if have written to more blocks, mark the block as
        // complete and record the size of the trail, which is reconstructed
        // from subsequent block(s).
        let mut spilled_bytes = if starting_block == ending_block {
            about.set_trail_bytes(ending_position as u64 % self.data_section_size);
            if ending_position % self.data_section_size as usize == 0 {
                about.set_complete();
                if written < bytes.len() {
                    about.set_trail_bytes(written as u64);
                }
            }
            0
        } else {
            about.set_complete();
            about.set_trail_bytes(self.data_section_size - current as u64 % self.data_section_size);
            bytes.len() as u64 - about.trail_bytes()
        };

        // If that's a spill from somewhere else, set the length of spilled
        // bytes and reset the trail_bytes.
        if spilled {
            about.set_spilled_bytes(bytes.len() as u64);
            about.set_trail_bytes(0);
        }

        // Initialize CRC32 hasher and update it with the data written to
        // the starting block.
        // SAFETY: Concurrent mutable access is protected by the lock.
        let mut crc32 =
            crc32fast::Hasher::new_with_initial(unsafe { self.meta_crc32(starting_block) });
        let end = cmp::min(
            ending_position,
            data_offset!(self.block_shift(), starting_block + 1),
        );
        // SAFETY: Concurrent mutable access is protected by the lock.
        crc32.update(unsafe { &(*self.data_buffer.get())[current..end] });

        // Write the metadata of the starting block.
        // SAFETY: This is the only function where writes to meta_buffer are
        // happening and it is protected by the lock. The overwrites are common,
        // and concurrent reads of meta sections may happen, which is handled
        // depending on the context.
        let meta_buffer = unsafe { &mut *self.meta_buffer.get() };
        meta_buffer[meta_about_section!(starting_block)].copy_from_slice(&<[u8; 8]>::from(about));
        meta_buffer[meta_parity_section!(starting_block)]
            .copy_from_slice(&about.parity().to_le_bytes());
        meta_buffer[meta_crc32_section!(starting_block)]
            .copy_from_slice(&crc32.finalize().to_le_bytes());

        // Write metadata to all subsequent blocks, where "middle" blocks are
        // additionally set as complete. The final block in this append is also
        // set as complete if the data lands at the end of that block.
        // Position is not saved here, as it is redundant - spilled_bytes
        // gives the offset within the block.
        for block in starting_block + 1..=ending_block {
            let mut about = AboutBlock::new();
            if block != ending_block || ending_position % self.data_section_size as usize == 0 {
                about.set_complete();
            }
            about.set_spilled_bytes(spilled_bytes);
            meta_buffer[meta_about_section!(block)].copy_from_slice(&<[u8; 8]>::from(about));
            meta_buffer[meta_parity_section!(block)].copy_from_slice(&about.parity().to_le_bytes());
            let start = data_offset!(self.block_shift(), block);
            let end = cmp::min(ending_position, data_offset!(self.block_shift(), block + 1));
            // SAFETY: Concurrent mutable access is protected by the lock.
            let crc32 = crc32fast::hash(unsafe { &(*self.data_buffer.get())[start..end] });
            meta_buffer[meta_crc32_section!(block)].copy_from_slice(&crc32.to_le_bytes());
            // Saturating sub is to handle the case when block == ending_block.
            spilled_bytes = spilled_bytes.saturating_sub(self.data_section_size);
        }

        if written < bytes.len() {
            // Readers of trail are interested only in atomic operations,
            // hence Relaxed. This value is used for buffer access, but it
            // is checked against synced value, which will ensure the order.
            self.trail.store(current, atomic::Ordering::Relaxed);
        }
        // Relaxed is fine, this position is not used for buffer access.
        // Release would be necessary for dirty reads, for example.
        self.current
            .store(ending_position, atomic::Ordering::Relaxed);
        Ok(written)
    }

    /// A shorter alternative to [`Stream::append_with_opts`], which has
    /// `spilled` flag unset.
    ///
    /// # Errors
    ///
    /// Returns same errors, as [`Stream::append_with_opts`] with the exception
    /// of [`StreamError::SpilledAfterFirstAppend`].
    #[inline(always)]
    pub fn append(&self, bytes: &[u8]) -> Result<usize, StreamError> {
        self.append_with_opts(bytes, /*spilled=*/ false)
    }

    /// Verifies integrity of the all blocks and calls `error` for every error
    /// status.
    ///
    /// There can be no more than one unique error per block, although a block
    /// could have multiple errors. See [`Inconsistency`] for possible
    /// inconsistencies.
    ///
    /// This call will spin wait if the lock is held by something else.
    pub fn verify(&self, error: impl FnMut(Inconsistency)) -> bool {
        let lock = Lock::acquire(&self.locked);
        self.verify_locked(error, &lock)
    }

    /// Pre-locked version of [`Stream::verify`].
    ///
    /// # Panics
    ///
    /// If the lock belongs to another block stream.
    #[allow(clippy::too_many_lines)]
    fn verify_locked(&self, mut error: impl FnMut(Inconsistency), lock: &Lock<'_>) -> bool {
        // The scan is happening under the lock, because the integrity of
        // both meta and data sections are checked. It should be possible
        // to grab the lock only after the last complete block was checked,
        // but I don't want to bother with that. This function is supposed to
        // be slow anyway and I expect it to be called before initializing.
        // When appending to the stream, the integrity is guaranteed by the code.
        assert_eq!(
            lock.0 as *const atomic::AtomicBool,
            core::ptr::addr_of!(self.locked),
            "unrelated lock",
        );

        // Flipped to false if at least one check fails.
        let mut result = true;
        // Tracks the first encountered incomplete block.
        let mut incomplete_block = None;
        // Tracks the first encountered empty block.
        let mut empty_block = None;
        // The block at which the spill has started and the value of the spilled
        // bytes from the previous block in the sequence.
        let mut spill_ctx: (Option<usize>, u64) = (None, 0);

        // The decision tree is slightly complicated, but seems complete.
        // Who could have known that blocks have that many entanglements!
        //
        // First make distinction between non-empty and empty blocks, then
        // divide non-empty into complete and incomplete. For every group,
        // first check integrity, then update the state variables. The
        // the exception is CRC32, which does its own thing at the end.
        for block in 0..self.block_count() as usize {
            // SAFETY: Concurrent mutable access is protected by the lock.
            let about = unsafe { self.meta_about(block) };

            #[allow(clippy::if_not_else)]
            if !about.is_empty() {
                // SAFETY: Concurrent mutable access is protected by the lock.
                if !about.verify(unsafe { self.meta_parity(block) }) {
                    result = false;
                    error(Inconsistency::MetaParityMismatch(block));
                }

                if let Some(empty_block) = empty_block {
                    result = false;
                    error(Inconsistency::EmptySequenceBroken(block, empty_block));
                }

                if let (Some(spill_block), prev_spilled_bytes) = spill_ctx {
                    if prev_spilled_bytes > 0
                        && about.spilled_bytes()
                            != prev_spilled_bytes.saturating_sub(self.data_section_size)
                    {
                        result = false;
                        error(Inconsistency::SpilledBytesUneven(block, spill_block));
                    }
                }

                if about.is_complete() {
                    if let Some(incomplete_block) = incomplete_block {
                        result = false;
                        error(Inconsistency::CompleteSequenceBroken(
                            block,
                            incomplete_block,
                        ));
                    }

                    if about.trail_bytes() > self.data_section_size {
                        result = false;
                        error(Inconsistency::TrailBytesTooLarge(block));
                    }

                    if about.spilled_bytes() <= self.data_section_size
                        && about.trail_bytes() + about.spilled_bytes() > self.data_section_size
                    {
                        result = false;
                        error(Inconsistency::TrailSpilledBytesTooLarge(block));
                    }

                    if about.spilled_bytes() > self.data_section_size {
                        if about.trail_bytes() != 0 {
                            result = false;
                            error(Inconsistency::TrailBytesUnexpected(block));
                        }

                        spill_ctx.1 = about.spilled_bytes();
                    }

                    if about.trail_bytes() != 0 {
                        spill_ctx.0 = Some(block);
                    }
                } else {
                    if let Some(incomplete_block) = incomplete_block {
                        result = false;
                        error(Inconsistency::IncompleteRepeated(block, incomplete_block));
                    }

                    if about.trail_bytes() != 0 && about.trail_bytes() <= about.spilled_bytes() {
                        result = false;
                        error(Inconsistency::TrailBytesTooSmall(block));
                    } else if about.trail_bytes() >= self.data_section_size {
                        result = false;
                        error(Inconsistency::TrailBytesTooLarge(block));
                    }

                    if about.spilled_bytes() > self.data_section_size {
                        result = false;
                        error(Inconsistency::SpilledBytesTooLarge(block));
                    }

                    if incomplete_block.is_none() {
                        incomplete_block = Some(block);
                    }

                    spill_ctx.0 = None;
                }
            } else {
                // SAFETY: Concurrent mutable access is protected by the lock.
                if unsafe { self.meta_parity(block) } != 0 {
                    result = false;
                    error(Inconsistency::EmptyNonZeroParity(block));
                }

                if empty_block.is_none() {
                    empty_block = Some(block);
                }

                spill_ctx.0 = None;
            }

            if about.spilled_bytes() <= self.data_section_size {
                spill_ctx.1 = 0;
            }

            let end = if about.is_complete() {
                data_offset!(self.block_shift(), block + 1)
            } else {
                data_offset!(self.block_shift(), block)
                    + cmp::min(
                        cmp::max(about.trail_bytes(), about.spilled_bytes()),
                        self.data_section_size,
                    ) as usize
            };
            // SAFETY: Concurrent mutable access is protected by the lock.
            let crc32 = unsafe { self.meta_crc32(block) };
            // SAFETY: Concurrent mutable access is protected by the lock.
            let crc32_data =
                unsafe { &(*self.data_buffer.get())[data_offset!(self.block_shift(), block)..end] };
            if crc32fast::hash(crc32_data) != crc32 {
                result = false;
                error(Inconsistency::DataChecksumMismatch(block));
            }
        }
        result
    }

    /// Loads all data from the underlying blocks into memory. Blocks should
    /// support vectored IO for efficiency.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    pub fn load(&mut self) -> io::Result<()> {
        let block_count = self.block_count() as usize;
        let mut iovec = Vec::with_capacity(2 * block_count);
        for block in 0..block_count {
            // SAFETY: Exclusive access to buffer is guaranteed by the borrow
            // checker. iovec contains non-overlapping slices.
            iovec.push(io::IoSliceMut::new(unsafe {
                &mut (*self.meta_buffer.get())[meta_section!(block)]
            }));
            // SAFETY: See the comment above.
            iovec.push(io::IoSliceMut::new(unsafe {
                &mut (*self.data_buffer.get())[data_section!(self.block_shift(), block)]
            }));
        }
        self.blocks.get_mut().load_from(0, &mut iovec)
    }

    /// Syncs dirty data in buffers into the underlying blocks. Blocks should
    /// support vectored IO for efficiency.
    ///
    /// It is OK to retry in case an error is returned. Retry may attempt
    /// syncing blocks that has been synced again. For example, block 1 and 2
    /// have dirty data, block 1 has been written, writing block 2 failed.
    /// A retry will write blocks 1 and 2 again.
    ///
    /// This call will spin wait if the lock is held by something else.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    pub fn sync(&self) -> io::Result<()> {
        // Lock is required to prevent syncs of incomplete writes.
        let lock = Lock::acquire(&self.locked);
        self.sync_locked(&lock)
    }

    /// Pre-locked version of [`Stream::sync`].
    ///
    /// # Panics
    ///
    /// If the lock belongs to another block stream.
    fn sync_locked(&self, lock: &Lock) -> io::Result<()> {
        assert_eq!(
            lock.0 as *const atomic::AtomicBool,
            core::ptr::addr_of!(self.locked),
            "unrelated lock",
        );

        // It is OK to load with relaxed, thanks to a lock.
        let current = self.current.load(atomic::Ordering::Relaxed);
        let synced = self.synced.load(atomic::Ordering::Relaxed);

        if current == synced {
            return Ok(());
        }

        let start = synced / self.data_section_size as usize;
        let end = (current - 1) / self.data_section_size as usize;
        let mut iovec = Vec::with_capacity((end - start + 1) * 2);
        for block in start..=end {
            // SAFETY: Concurrent mutable access is protected by the lock.
            iovec.push(io::IoSlice::new(unsafe {
                &(*self.meta_buffer.get())[meta_section!(block)]
            }));
            let data_range = if block == end {
                data_offset!(self.block_shift(), block)..current
            } else {
                data_section!(self.block_shift(), block)
            };
            // SAFETY: Concurrent mutable access is protected by the lock.
            iovec.push(io::IoSlice::new(unsafe {
                &(*self.data_buffer.get())[data_range]
            }));
        }

        // SAFETY: Concurrent mutable access is protected by the lock.
        let blocks = unsafe { &mut *self.blocks.get() };
        blocks.store_at(start as u64, &mut iovec)?;
        // I thought of doing Release here, but it seems that Relaxed also
        // works. Readers read from buffers, which has been written fully
        // up to the state that is stored here because of the lock shared with
        // append_with_opts
        self.synced.store(current, atomic::Ordering::Relaxed);
        Ok(())
    }
}

impl<B> Stream<B> {
    /// Verifies whether the byte slice passed to [`Stream::append`] is correct
    /// and returns an error if it is not.
    #[inline(always)]
    pub(crate) fn verify_append(bytes: &[u8]) -> Result<(), StreamError> {
        if bytes.len() > AboutBlock::SPILLED_MAX as usize {
            Err(StreamError::AppendTooLarge(
                AboutBlock::SPILLED_MAX as usize,
            ))
        } else {
            Ok(())
        }
    }

    /// Consumes this block stream and returns the underlying `Blocks`.
    #[inline(always)]
    #[must_use = "do you want to drop instead?"]
    pub fn into_inner(self) -> B {
        self.blocks.into_inner()
    }

    /// Returns the total number of bytes appended and synced to this stream.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.synced.load(atomic::Ordering::Relaxed)
    }

    /// Returns the total number of bytes this stream can hold.
    #[inline(always)]
    #[must_use]
    pub fn capacity(&self) -> usize {
        // SAFETY: Data buffer length is fixed after stream has been created.
        unsafe { &*self.data_buffer.get() }.len()
    }

    /// Whether this stream is empty. Unsynced stream is not empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current.load(atomic::Ordering::Relaxed) == 0
    }

    /// Whether this stream has unsynced changes.
    #[inline(always)]
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.current.load(atomic::Ordering::Relaxed) != self.synced.load(atomic::Ordering::Relaxed)
    }

    /// Whether this stream is full and cannot accept any more appends.
    /// Calling append on a full stream will return an error.
    #[inline(always)]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.current.load(atomic::Ordering::Relaxed) == self.capacity()
    }

    /// Returns the view into the appended data of this block stream, excluding
    /// any spilled data at the start or trailing data at the end. If the
    /// stream is uninitialized, then this will always return an empty slice.
    #[inline(always)]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        // SAFETY: Data until synced position is guaranteed to be written
        // and not change. Trail is used as an end only when the append that
        // has created the trail has been synced. See also the comment in
        // `data_range`.
        unsafe { &(*self.data_buffer.get())[self.data_range()] }
    }

    /// Returns the view into the spilled data part at the start of this block
    /// stream. Does not depend on a stream being initialized.
    #[inline(always)]
    #[must_use]
    pub fn spilled(&self) -> &[u8] {
        // Relaxed, because writes to buffers and a sync are synchronized by
        // a shared lock. Therefore, data up to synced position is guaranteed
        // to be correct.
        let synced = self.synced.load(atomic::Ordering::Relaxed);
        // SAFETY: First, spilled_size is only on the first block, so unless
        // the buffer was synced, synced position will be 0. After a sync,
        // it will contain the correct value. Second, spilled_bytes are not
        // updated after the initial append. Data after a sync is guaranteed
        // not to change.
        unsafe { &(*self.data_buffer.get())[..cmp::min(self.spilled_size(), synced)] }
    }

    /// Returns the view into the trailing data part at the end of this block
    /// stream. On uninitialized streams, this is always the whole data buffer.
    #[inline(always)]
    #[must_use]
    pub fn trailing(&self) -> &[u8] {
        // Relaxed, because data_buffer is append-only and synced position
        // is guaranteed to contain the data fully written to it.
        let synced = self.synced.load(atomic::Ordering::Relaxed);
        // Relaxed, because this atomic is always updated before synced.
        let trail = self.trail.load(atomic::Ordering::Relaxed);
        // At the time of the append when trail is updated, synced will point to
        // the same or earlier position, allowing to prevent dirty changes to
        // appear by a simple comparison. Trail is updated only once and is
        // fixed after.
        if synced <= trail {
            self.data_buffer_end()
        } else {
            // SAFETY: Trail is smaller than synced if and only if the append
            // that has created the trail has been synced. Synced data is
            // guaranteed to not change.
            unsafe { &(*self.data_buffer.get())[trail..] }
        }
    }

    /// Lock this block stream, spinning if the lock is held by something else.
    /// Dropping the result unlocks the stream.
    #[inline(always)]
    pub fn lock(&self) -> LockedStream<'_, B> {
        LockedStream(self, Lock::acquire(&self.locked))
    }

    /// Attempts to lock this block stream, returning the locked block stream.
    /// if successful. Dropping the result unlocks the stream.
    ///
    /// # Errors
    ///
    /// On failure returns [`StreamError::Busy`].
    #[inline(always)]
    pub fn try_lock(&self) -> Result<LockedStream<'_, B>, StreamError> {
        Lock::try_acquire(&self.locked)
            .map(|lock| LockedStream(self, lock))
            .ok_or(StreamError::Busy)
    }
}

impl<B: Blocks> From<B> for Stream<B> {
    #[inline(always)]
    #[must_use]
    fn from(value: B) -> Self {
        Self::new(value)
    }
}

impl<B> core::fmt::Debug for Stream<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: Meta buffer length is fixed after stream has been created.
        let meta_buffer_len = unsafe { &(*self.meta_buffer.get()) }.len();
        f.debug_struct("blockstream")
            .field("data_section_size", &self.data_section_size)
            .field("meta_buffer_len", &meta_buffer_len)
            .field("data_buffer_len", &self.capacity())
            .field("current", &self.current)
            .field("synced", &self.synced)
            .field("trail", &self.trail)
            .field("locked", &self.locked)
            .finish_non_exhaustive()
    }
}

unsafe impl<B: Send> Send for Stream<B> {}
unsafe impl<B: Send> Sync for Stream<B> {}

/// An error specific to the [`Stream`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StreamError {
    /// An attempt to append from a buffer that is too large. The inner value
    /// is the maximum allowed size of a buffer.
    AppendTooLarge(usize),
    /// The stream is busy with another operation that requires exclusive write
    /// access.
    Busy,
    /// The stream is full and cannot be appended to.
    Full,
    /// The meta section of a block is inconsistent.
    MetaSectionCorrupted(&'static str),
    /// An attempt to append spilled data in the middle of the stream.
    SpilledAfterFirstAppend,
}

impl std::error::Error for StreamError {}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AppendTooLarge(limit) => {
                write!(f, "blockstream: append exceeds the limit of {limit} bytes")
            }
            Self::Busy => {
                write!(f, "blockstream: stream is busy with another operation")
            }
            Self::Full => {
                write!(f, "blockstream: stream is full")
            }
            Self::MetaSectionCorrupted(msg) => {
                write!(f, "blockstream: {msg}: meta section corrupted?")
            }
            Self::SpilledAfterFirstAppend => {
                write!(
                    f,
                    "blockstream: spilled option is allowed only on the first append"
                )
            }
        }
    }
}

/// An error status returned during [`Stream::verify`] call for every block.
///
/// First enclosed value is always a block number, at which inconsistency was
/// encountered.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Inconsistency {
    /// Meta section parity of a non-empty block is not consistent with the
    /// value. This usually means the meta section is corrupted. If that
    /// has happened during normal operation, there is a chance to recover.
    MetaParityMismatch(usize),
    /// A block is marked as complete, yet it is not expected, because the
    /// last contiguous sequence of complete blocks has ended on block number
    /// in the second value. This could mean that blocks were not
    /// zero-initialized and contain some garbage.
    CompleteSequenceBroken(usize, usize),
    /// An incomplete block following last incomplete block with a block
    /// number in the second value. Under normal operation, there is only
    /// one incomplete block after every append.
    IncompleteRepeated(usize, usize),
    /// A non-empty block (complete or incomplete) found after an empty block
    /// with a block number in the second value. This could mean that blocks
    /// were not zero-initialized and contain some garbage. Once an empty block
    /// has been found, all future blocks must be empty as well.
    EmptySequenceBroken(usize, usize),
    /// Trail size in bytes on an incomplete block, if non-zero, is smaller
    /// than the spilled size. On incomplete blocks, trail shows the next
    /// position to append stream data to, and it cannot overlap with data
    /// spilled from the previous block.
    TrailBytesTooSmall(usize),
    /// For complete blocks, trail bytes exceed the size of the block data
    /// section, for incomplete blocks, the trail bytes is greater or equal
    /// to the block data section size.
    TrailBytesTooLarge(usize),
    /// Trail bytes are set, but not expected. This is for blocks which are
    /// in the middle of a spill.
    TrailBytesUnexpected(usize),
    /// A complete block with the sum of trail and spilled bytes larger than
    /// the size of the block data section.
    TrailSpilledBytesTooLarge(usize),
    /// An incomplete block with spilled bytes exceeding the size of the
    /// block data section.
    SpilledBytesTooLarge(usize),
    /// Spilled bytes from previous block(s) that spill further are not equal
    /// to spilled bytes of a previous block minus the size of the data section
    /// of a block. The second value indicates a block number at which spill
    /// has started.
    SpilledBytesUneven(usize, usize),
    /// Data CRC32 checksum is not consistent with a checksum written in the
    /// meta section.
    DataChecksumMismatch(usize),
    /// A block appears to be empty, but has non-zero parity.
    EmptyNonZeroParity(usize),
}

impl<B> Stream<B> {
    /// Returns the about part of the meta section.
    ///
    /// # Safety
    ///
    /// Take into consideration that `meta_buffer` may be modified concurrently.
    ///
    /// # Panics
    ///
    /// Panics if `block` is equal to or greater than `block_count`
    #[inline(always)]
    #[must_use]
    unsafe fn meta_about(&self, block: usize) -> AboutBlock {
        AboutBlock::try_from(&(*self.meta_buffer.get())[meta_about_section!(block)])
            .unwrap_unchecked()
    }

    /// Returns the parity part of the meta section.
    ///
    /// # Safety
    ///
    /// Take into consideration that `meta_buffer` may be modified concurrently.
    ///
    /// # Panics
    ///
    /// Panics if `block` is equal to or greater than `block_count`
    #[inline(always)]
    #[must_use]
    unsafe fn meta_parity(&self, block: usize) -> u32 {
        u32::from_le_bytes(
            (*self.meta_buffer.get())[meta_parity_section!(block)]
                .try_into()
                .unwrap_unchecked(),
        )
    }

    /// Returns the CRC32 part of the meta section.
    ///
    /// # Safety
    ///
    /// Take into consideration that `meta_buffer` may be modified concurrently.
    ///
    /// # Panics
    ///
    /// Panics if `block` is equal to or greater than `block_count`
    #[inline(always)]
    #[must_use]
    unsafe fn meta_crc32(&self, block: usize) -> u32 {
        u32::from_le_bytes(
            (*self.meta_buffer.get())[meta_crc32_section!(block)]
                .try_into()
                .unwrap_unchecked(),
        )
    }

    /// A safe convenience wrapper that returns an empty slice from data buffer.
    #[inline(always)]
    #[must_use]
    fn data_buffer_end(&self) -> &[u8] {
        // SAFETY: The returned slice is always an empty slice.
        unsafe { &(*self.data_buffer.get())[self.capacity()..] }
    }

    /// Returns the range of the appended data within `data_buffer`, excluding
    /// any spilled data at the start or trailing data at the end.
    #[inline(always)]
    #[must_use]
    fn data_range(&self) -> Range<usize> {
        let end = cmp::min(
            // Relaxed, because data_buffer is append-only and synced position
            // is guaranteed to contain the data fully written to it.
            self.synced.load(atomic::Ordering::Relaxed),
            // Relaxed, because this atomic is always updated before synced.
            self.trail.load(atomic::Ordering::Relaxed),
        );
        // SAFETY: First, spilled_size is only on the first block, so unless
        // the buffer was synced, end will be 0. After sync, it will contain
        // the correct value. Second, spilled_bytes are not updated after the
        // initial append.
        let start = cmp::min(unsafe { self.spilled_size() }, end);
        start..end
    }

    /// Returns the size in bytes of the append that has spilled into this
    /// block stream.
    ///
    /// # Safety
    ///
    /// This function may potentially return some garbage, which should be
    /// accounted for when using the result, or `meta_buffer` must not be
    /// modified concurrently.
    ///
    /// Note, that `spilled_bytes` are written only once. If one of the
    /// synced or current positions advances past that, it will always return
    /// the correct value.
    #[inline(always)]
    #[must_use]
    unsafe fn spilled_size(&self) -> usize {
        cmp::min(self.meta_about(0).spilled_bytes() as usize, self.capacity())
    }
}

impl<B: Blocks> Stream<B> {
    /// Finds the ending block within the stream, which if incomplete, is
    /// supposed to be the only incomplete block. Because the stream requires
    /// that there is only one incomplete block, complete blocks to always come
    /// before, and empty blocks to come after, the implementation is a simple
    /// binary search where complete blocks are "smaller" and empty blocks are
    /// "bigger". Returns the last complete block if the stream is full and
    /// first empty block if there are no incomplete blocks.
    ///
    /// Running this function on corrupted structure is undefined, as in it
    /// may or may not return an ending block, and if it does return an
    /// empty or incomplete block, it can be an arbitrary one.
    ///
    /// # Safety
    ///
    /// `meta_buffer` must not be modified concurrently.
    #[inline(always)]
    #[must_use]
    unsafe fn find_ending_block(&self) -> usize {
        let mut start = 0;
        let mut end = self.block_count() as usize - 1;
        while start != end {
            let middle = (start + end) / 2;
            let about = self.meta_about(middle);
            if about.is_complete() {
                // A tiny optimization for large spills. Not that I expect
                // this case to be frequent, but it's supposed to be cheap
                // operation and does not complicate the algorithm much,
                // only adds a division and calls to min and max.
                let skip = cmp::max((about.spilled_bytes() / self.data_section_size) as usize, 1);
                start = cmp::min(middle + skip, end);
                continue;
            }
            if about.is_empty() {
                end = middle;
                continue;
            }
            return middle;
        }
        end
    }

    /// Returns the size in bytes of the append that has spilled outside this
    /// block stream.
    ///
    /// # Panics
    ///
    /// Panics if `block` is equal to or greater than `block_count`
    #[inline(always)]
    fn trail_size(&self, block: usize, about: AboutBlock) -> Result<usize, StreamError> {
        assert!(
            block < self.block_count() as usize,
            "block out of bounds: the count is {} but the block is {block}",
            self.block_count()
        );

        // Incomplete block, means that we haven't reached the end yet.
        if !about.is_complete() {
            return Ok(0);
        }
        // If this block has trail bytes, that's exactly what we want, unless
        // it is too large.
        let trail = about.trail_bytes();
        if trail > self.data_section_size {
            return Err(StreamError::MetaSectionCorrupted("trail too large"));
        }
        if trail > 0 {
            return Ok(trail as usize);
        }
        // If spilled bytes are partially covering this block, then the remaining
        // items nicely fit into the remaining space. Similar case when spilled
        // bytes cover this block precisely.
        if about.spilled_bytes() <= self.data_section_size {
            return Ok(0);
        }
        // Otherwise just keep searching backwards until we find a trail.
        let mut spilled = self.data_section_size;
        for block in (0..block).rev() {
            // SAFETY: The current block is complete, therefore all previous
            // blocks are also complete, implying that meta is not changed.
            let about = unsafe { self.meta_about(block) };
            if !about.is_complete() {
                return Err(StreamError::MetaSectionCorrupted("block not complete"));
            }
            let trail = about.trail_bytes();
            if trail > self.data_section_size {
                return Err(StreamError::MetaSectionCorrupted("trail too large"));
            }
            if trail > 0 {
                return Ok((trail + spilled) as usize);
            }
            if about.spilled_bytes() <= self.data_section_size {
                return Err(StreamError::MetaSectionCorrupted("missing trail"));
            }
            spilled += self.data_section_size;
        }
        // If there's no trail meta, the whole stream is just a trail.
        Ok(spilled as usize)
    }
}

/// A locked [`Stream`]. Returned by [`Stream::try_lock`] or [`Stream::lock`].
#[derive(Debug)]
#[must_use = "locking without holding a lock does not make sense?"]
pub struct LockedStream<'a, B>(&'a Stream<B>, Lock<'a>);

impl<'a, B> LockedStream<'a, B> {
    /// Releases the lock and returns the underlying `Stream`.
    #[inline(always)]
    #[must_use = "do you want to drop instead?"]
    pub fn unlock(this: Self) -> &'a Stream<B> {
        drop(this.1);
        this.0
    }
}

impl<B: Blocks> LockedStream<'_, B> {
    /// Same as [`Stream::append_with_opts`], except that it is pre-locked.
    ///
    /// # Errors
    ///
    /// See [`Stream::append_with_opts`] for details about errors.
    #[inline(always)]
    pub fn append_with_opts(
        this: &Self,
        bytes: &[u8],
        spilled: bool,
    ) -> Result<usize, StreamError> {
        this.0.append_with_opts_locked(bytes, spilled, &this.1)
    }

    /// Same as [`Stream::append`], except that it is pre-locked.
    ///
    /// # Errors
    ///
    /// See [`Stream::append`] for details about errors.
    #[inline(always)]
    pub fn append(this: &Self, bytes: &[u8]) -> Result<usize, StreamError> {
        Self::append_with_opts(this, bytes, /*spilled=*/ false)
    }

    /// Same as [`Stream::verify`], except that it is pre-locked.
    #[inline(always)]
    pub fn verify<F: FnMut(Inconsistency)>(this: &Self, error: F) -> bool {
        this.0.verify_locked(error, &this.1)
    }

    /// Same as [`Stream::sync`], except that it is pre-locked.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    #[inline(always)]
    pub fn sync(this: &Self) -> io::Result<()> {
        this.0.sync_locked(&this.1)
    }
}

impl<B> core::ops::Deref for LockedStream<'_, B> {
    type Target = Stream<B>;

    #[inline(always)]
    fn deref(&self) -> &Stream<B> {
        self.0
    }
}

/// A simple RAII wrapper around an atomic bool used for locking.
#[derive(Debug)]
#[must_use = "locking without holding a lock does not make sense?"]
struct Lock<'a>(&'a atomic::AtomicBool);

impl<'a> Lock<'a> {
    /// Loops in the most basic way in an attempt to acquire a lock.
    #[inline(always)]
    fn acquire(locked: &'a atomic::AtomicBool) -> Self {
        loop {
            if let Some(lock) = Self::try_acquire(locked) {
                return lock;
            }
            core::hint::spin_loop();
        }
    }

    /// Attempts to acquire a lock, returning `None` if the attempt has failed.
    #[inline(always)]
    fn try_acquire(locked: &'a atomic::AtomicBool) -> Option<Self> {
        locked
            .compare_exchange_weak(
                false,
                true,
                atomic::Ordering::Acquire,
                atomic::Ordering::Relaxed,
            )
            .ok()
            .map(|_| Lock(locked))
    }
}

impl Drop for Lock<'_> {
    /// Releases the underlying lock.
    ///
    /// # Panics
    ///
    /// If the lock has been released earlier.
    #[inline(always)]
    fn drop(&mut self) {
        loop {
            match self.0.compare_exchange_weak(
                true,
                false,
                atomic::Ordering::Release,
                atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(curr) => assert!(!curr),
            }
        }
    }
}

/// A special entry in each block of the stream, describing the type of that
/// block. Represented as a single u64.
///
/// Blocks may have data spilled from the previous block and / or spill data
/// to future block(s). Storing this state allows navigating the buffer
/// more efficiently, as well as recovering after restarts or crashes.
///
/// The state is encoded in the following way:
/// ```text
///   00  04  08  0c  10  14  18  1c  20  24  28  2c  30  34  38  3c  40
/// ++---+---------------------------+-+------------------------------++
/// || A |            B              |C|              D               ||
/// ++---+---------------------------+-+------------------------------++
///
/// A (4 bits) = Only 1 bit is used to indicate whether the block is complete.
/// B (28 bits) = If block is complete, this is the size in bytes of the
///   partial write to this block, i.e. the trailing size. If block is not
///   complete, this is the offset from the start of the data section of the
///   block to the end of the last write to that block. Hence, the block size
///   is limited roughly to 256 MiB.
/// C (2 bits) = Unused.
/// D (30 bits) = The remaining size in bytes of the data spilling from the
///   previous block(s). From that it is possible to derive how many blocks
///   the spill continues for. That also implies that the largest size of
///   a single write is roughly 1 GiB.
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AboutBlock(u64);

impl AboutBlock {
    const FLAGS_BITS: u32 = 4;
    const TRAIL_BITS: u32 = 28;
    const UNUSED_BITS: u32 = 2;
    const SPILLED_BITS: u32 = 30;

    const SPILLED_SHIFT: u32 = Self::FLAGS_BITS + Self::TRAIL_BITS + Self::UNUSED_BITS;

    const TRAIL_MAX: u64 = (1 << Self::TRAIL_BITS) - 1;
    const SPILLED_MAX: u64 = (1 << Self::SPILLED_BITS) - 1;

    const TRAIL_MASK: u64 = !(Self::TRAIL_MAX << Self::FLAGS_BITS);
    const SPILLED_MASK: u64 = !(Self::SPILLED_MAX << Self::SPILLED_SHIFT);

    const FLAG_COMPLETE: u64 = 0b0001;

    /// Initializes a new empty instance of `AboutBlock`.
    #[inline(always)]
    #[must_use]
    fn new() -> AboutBlock {
        AboutBlock(0)
    }

    /// Whether this block is empty.
    #[inline(always)]
    #[must_use]
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether this block is complete.
    #[inline(always)]
    #[must_use]
    fn is_complete(self) -> bool {
        self.0 & Self::FLAG_COMPLETE == 1
    }

    /// Marks this block as complete.
    #[inline(always)]
    fn set_complete(&mut self) {
        self.0 |= Self::FLAG_COMPLETE;
    }

    /// Returns the amount of data in the trail part of this block in bytes.
    /// Maximum is capped at 2<sup>28</sup> &minus; 1 bytes.
    #[inline(always)]
    #[must_use]
    fn trail_bytes(self) -> u64 {
        u64::from(self.0 as u32 >> Self::FLAGS_BITS)
    }

    /// Sets the amount of trail bytes.
    ///
    /// # Panics
    ///
    /// Panics if the value is greater than 2<sup>28</sup> &minus; 1.
    #[inline(always)]
    fn set_trail_bytes(&mut self, bytes: u64) {
        assert!(bytes < 1 << Self::TRAIL_BITS, "too much bytes");
        self.0 &= Self::TRAIL_MASK;
        self.0 |= bytes << Self::FLAGS_BITS;
    }

    /// Returns the amount of data spilled from the previous block to this and
    /// potentially future block(s) in bytes. Maximum is capped at
    /// 2<sup>30</sup> &minus; 1 bytes.
    #[inline(always)]
    #[must_use]
    fn spilled_bytes(self) -> u64 {
        self.0 >> Self::SPILLED_SHIFT
    }

    /// Sets the amount of spilled bytes.
    ///
    /// # Panics
    ///
    /// Panics if the value is greater than 2<sup>30</sup> &minus; 1.
    #[inline(always)]
    fn set_spilled_bytes(&mut self, bytes: u64) {
        assert!(bytes < 1 << Self::SPILLED_BITS, "too much bytes");
        self.0 &= Self::SPILLED_MASK;
        self.0 |= bytes << Self::SPILLED_SHIFT;
    }

    /// Computes parity bits of the two halves of the state.
    #[inline(always)]
    #[must_use]
    fn parity(self) -> u32 {
        !((self.0 >> 32) as u32 ^ self.0 as u32)
    }

    /// Verifies correctness against parity bits.
    #[inline(always)]
    #[must_use]
    fn verify(self, parity: u32) -> bool {
        let upper = (self.0 >> 32) as u32;
        let lower = self.0 as u32;
        upper ^ !parity == lower && lower ^ !parity == upper
    }
}

impl From<[u8; 8]> for AboutBlock {
    #[inline(always)]
    #[must_use]
    fn from(value: [u8; 8]) -> AboutBlock {
        AboutBlock(u64::from_le_bytes(value))
    }
}

impl From<AboutBlock> for [u8; 8] {
    #[inline(always)]
    #[must_use]
    fn from(value: AboutBlock) -> [u8; 8] {
        value.0.to_le_bytes()
    }
}

impl TryFrom<&[u8]> for AboutBlock {
    type Error = core::array::TryFromSliceError;

    #[inline(always)]
    fn try_from(slice: &[u8]) -> Result<AboutBlock, Self::Error> {
        Ok(<[u8; 8]>::try_from(slice)?.into())
    }
}

impl Default for AboutBlock {
    #[inline(always)]
    #[must_use]
    fn default() -> AboutBlock {
        AboutBlock::new()
    }
}

#[cfg(test)]
mod tests {
    use core::array;
    use std::io::{Read, Seek, Write};

    use super::*;

    macro_rules! assert_meta {
        (@ $block:literal $about:ident) => {
            assert!($about.is_complete(), "block {}: incomplete", $block);
        };
        (! $block:literal $about:ident) => {
            assert!(!$about.is_complete(), "block {}: complete", $block);
        };

        ($blocks:expr; $($flag:tt$block:literal $spilled:literal $trail:literal),+) => {
            $(
                let about = $blocks.read_meta_about($block);
                assert_meta!($flag $block about);
                assert!(about.verify($blocks.read_meta_parity($block)), "block {}: parity mismatch", $block);
                assert_eq!(about.spilled_bytes(), $spilled, "block {}: spilled bytes", $block);
                assert_eq!(about.trail_bytes(), $trail, "block {}: trail bytes", $block);
            )+
        }
    }

    macro_rules! assert_positions {
        ($stream:expr, $values:expr, $($args:tt)+) => {
            let current = $stream.current.load(atomic::Ordering::Relaxed);
            let synced = $stream.synced.load(atomic::Ordering::Relaxed);
            let trail = $stream.trail.load(atomic::Ordering::Relaxed);
            assert_eq!((current, synced, trail), $values, $($args)+);
        };
    }

    macro_rules! append_stream {
        ($stream:expr, $bytes:expr, $length:expr) => {
            assert_eq!($stream.append($bytes).unwrap(), $length, "stream append");
            $stream.sync().unwrap();
            verify_stream!($stream);
        };

        ($stream:expr, $bytes:expr) => {
            append_stream!($stream, $bytes, $bytes.len());
        };
    }

    macro_rules! verify_stream {
        ($stream:expr) => {
            assert!($stream.verify(|status| eprintln!("{:?}", status)), "stream verify");
        };

        ($stream:expr, $($pattern:pat),+) => {
            let mut unmatched = false;
            $stream.verify(|status| match status {
                $(
                    $pattern => (),
                )+
                _ => {
                    unmatched = true;
                    eprintln!("{:?}", status);
                }
            });
            assert!(!unmatched, "stream verify: unmatched inconsistencies");
        }
    }

    macro_rules! write_meta {
        (@ $about:ident) => {
            $about.set_complete();
        };
        (! $about:ident) => {};

        ($blocks:expr; $($flag:tt$block:literal $spilled:literal $trail:literal $size:literal),+) => {
            $(
                let mut about = AboutBlock::new();
                write_meta!($flag about);
                about.set_spilled_bytes($spilled);
                about.set_trail_bytes($trail);
                $blocks.write_meta_block($block, about, $size);
            )+
        };
    }

    #[test]
    #[should_panic(expected = "blocks have no blocks?")]
    fn stream_new_empty() {
        struct EmptyBlocks;
        impl Blocks for EmptyBlocks {
            fn block_count(&self) -> u64 {
                0
            }
            fn block_shift(&self) -> u32 {
                10
            }
            fn load_from(
                &mut self,
                _block: u64,
                _bufs: &mut [io::IoSliceMut<'_>],
            ) -> io::Result<()> {
                unimplemented!()
            }
            fn store_at(&mut self, _block: u64, _bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
                unimplemented!()
            }
        }
        let _ = Stream::new(EmptyBlocks);
    }

    #[test]
    #[should_panic(expected = "block shift too small")]
    fn stream_new_small() {
        struct SmallBlocks;
        impl Blocks for SmallBlocks {
            fn block_count(&self) -> u64 {
                1
            }
            fn block_shift(&self) -> u32 {
                5
            }
            fn load_from(
                &mut self,
                _block: u64,
                _bufs: &mut [io::IoSliceMut<'_>],
            ) -> io::Result<()> {
                unimplemented!()
            }
            fn store_at(&mut self, _block: u64, _bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
                unimplemented!()
            }
        }
        let _ = Stream::new(SmallBlocks);
    }

    #[test]
    #[should_panic(expected = "block shift too big")]
    fn stream_new_large() {
        struct LargeBlocks;
        impl Blocks for LargeBlocks {
            fn block_count(&self) -> u64 {
                1
            }
            fn block_shift(&self) -> u32 {
                28
            }
            fn load_from(
                &mut self,
                _block: u64,
                _bufs: &mut [io::IoSliceMut<'_>],
            ) -> io::Result<()> {
                unimplemented!()
            }
            fn store_at(&mut self, _block: u64, _bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
                unimplemented!()
            }
        }
        let _ = Stream::new(LargeBlocks);
    }

    #[test]
    fn stream_len() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty uninitialized is max-length";
        assert_eq!(stream.len(), 1280, "{case}");

        let case = "empty initialized is zero-length";
        stream.initialize().unwrap();
        assert_eq!(stream.len(), 0, "{case}");

        let case = "unsynced is zero-length";
        assert_eq!(stream.append(&TEST_BYTES_REPEATING[..2]).unwrap(), 2);
        verify_stream!(stream);
        assert_eq!(stream.len(), 0, "{case}");

        let case = "synced is synced-length";
        stream.sync().unwrap();
        assert_eq!(stream.len(), 2, "{case}");
    }

    #[test]
    fn stream_data_block_range() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty uninitialized";
        assert_eq!(stream.data_block_range(), 0..=0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "empty initialized";
        stream.initialize().unwrap();
        assert_eq!(stream.data_block_range(), 0..=0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "unsynced";
        stream
            .append_with_opts(&TEST_BYTES_REPEATING[..88], true)
            .unwrap();
        assert_eq!(stream.data_block_range(), 0..=0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "no data with spilled section";
        stream.sync().unwrap();
        assert_eq!(stream.data_block_range(), 1..=1, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..0, "{case}");

        let case = "data before the block boundary";
        append_stream!(stream, &TEST_BYTES_REPEATING[..70]);
        assert_eq!(stream.data_block_range(), 1..=1, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..70, "{case}");

        let case = "data at the block boundary";
        append_stream!(stream, &TEST_BYTES_REPEATING[..2]);
        assert_eq!(stream.data_block_range(), 1..=1, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..72, "{case}");

        let case = "data spilled to another block";
        append_stream!(stream, &TEST_BYTES_REPEATING[..88]);
        assert_eq!(stream.data_block_range(), 1..=2, "{case}");
        assert_eq!(stream.data_range_for(2).expect(case), 72..160, "{case}");

        let case = "data after a spill";
        append_stream!(stream, &TEST_BYTES_REPEATING[..2]);
        assert_eq!(stream.data_block_range(), 1..=3, "{case}");
        assert_eq!(stream.data_range_for(3).expect(case), 160..162, "{case}");

        let case = "data spilled till the block boundary";
        append_stream!(stream, &TEST_BYTES_REPEATING[..70 + 80]);
        assert_eq!(stream.data_block_range(), 1..=3, "{case}");
        assert_eq!(stream.data_range_for(3).expect(case), 160..312, "{case}");

        // Retrieve blocks to test two branches.
        let blocks = stream.into_inner();

        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        stream.initialize().unwrap();
        let case = "data after a partial append following a spill";
        append_stream!(stream, TEST_BYTES_REPEATING, 880);
        assert_eq!(stream.data_block_range(), 1..=3, "{case}");
        assert_eq!(stream.data_range_for(3).expect(case), 160..312, "{case}");

        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        stream.initialize().unwrap();
        let case = "data spilled over 2 blocks at block boundary";
        append_stream!(stream, &TEST_BYTES_REPEATING[..160]);
        assert_eq!(stream.data_block_range(), 1..=5, "{case}");
        assert_eq!(stream.data_range_for(5).expect(case), 312..472, "{case}");

        let case = "data at the block boundary following a spill";
        append_stream!(stream, &TEST_BYTES_REPEATING[..80]);
        assert_eq!(stream.data_block_range(), 1..=7, "{case}");
        assert_eq!(stream.data_range_for(7).expect(case), 472..552, "{case}");

        let case = "data after partial append following normal append";
        append_stream!(stream, &TEST_BYTES_REPEATING, 640);
        assert_eq!(stream.data_block_range(), 1..=7, "{case}");
        assert_eq!(stream.data_range_for(7).expect(case), 472..552, "{case}");
    }

    #[test]
    fn stream_is_empty() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty uninitialized is not empty";
        assert!(!stream.is_empty(), "{case}");

        let case = "empty initialized is empty";
        stream.initialize().unwrap();
        assert!(stream.is_empty(), "{case}");

        let case = "non-empty is not empty";
        append_stream!(stream, &TEST_BYTES_REPEATING[..2]);
        assert!(!stream.is_empty(), "{case}");
    }

    #[test]
    fn stream_is_dirty() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty uninitialized is not dirty";
        assert!(!stream.is_dirty(), "{case}");

        let case = "empty initialized is not dirty";
        stream.initialize().unwrap();
        assert!(!stream.is_dirty(), "{case}");

        let case = "unsynced is dirty";
        assert_eq!(stream.append(&TEST_BYTES_REPEATING[..2]).unwrap(), 2);
        verify_stream!(stream);
        assert!(stream.is_dirty(), "{case}");

        let case = "synced is not dirty";
        stream.sync().unwrap();
        assert!(!stream.is_dirty(), "{case}");
    }

    #[test]
    fn stream_is_full() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty uninitialized is full";
        assert!(stream.is_full(), "{case}");

        let case = "empty initialized is not full";
        stream.initialize().unwrap();
        assert!(!stream.is_full(), "{case}");

        let case = "appended is not full";
        append_stream!(stream, &TEST_BYTES_REPEATING[..2]);
        assert!(!stream.is_full(), "{case}");

        let case = "full unsynced is full";
        assert_eq!(stream.append(&TEST_BYTES_REPEATING).unwrap(), 1278);
        verify_stream!(stream);
        assert!(stream.is_full(), "{case}");

        let case = "full synced is full";
        stream.sync().unwrap();
        assert!(stream.is_full(), "{case}");
    }

    #[test]
    fn stream_read() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "uninitialized empty";
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 1280, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "uninitialized spilled";
        write_meta!(stream.blocks.get_mut(); @0 88 0 80);
        stream.load().unwrap();
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 88, "{case}");
        assert_eq!(stream.trailing().len(), 1280, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "initialized spilled empty";
        write_meta!(stream.blocks.get_mut(); @0 88 0 80, !1 8 0 8);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 88, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..0, "{case}");

        let case = "initialized spilled nonempty";
        write_meta!(stream.blocks.get_mut(); @0 88 0 80, !1 8 24 24);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 16, "{case}");
        assert_eq!(stream.spilled().len(), 88, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect_err(case), 1, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..16, "{case}");

        let case = "initialized nonempty";
        write_meta!(stream.blocks.get_mut(); @0 0 34 80, !1 8 48 48);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 128, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..88, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 88..128, "{case}");

        let case = "initialized spilled full no trail";
        let mut about = AboutBlock::new();
        about.set_complete();
        for i in 1..=TestMemoryBlocks::BLOCK_COUNT {
            about.set_spilled_bytes(i * 80);
            stream
                .blocks
                .get_mut()
                .write_meta_block(TestMemoryBlocks::BLOCK_COUNT - i, about, 80);
        }
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 1280, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(14).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(15).expect(case), 0..0, "{case}");

        let case = "initialized spilled full with trail";
        let mut about = AboutBlock::new();
        about.set_complete();
        for i in 1..=TestMemoryBlocks::BLOCK_COUNT {
            about.set_spilled_bytes(i * 80 + 8);
            stream
                .blocks
                .get_mut()
                .write_meta_block(TestMemoryBlocks::BLOCK_COUNT - i, about, 80);
        }
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 1280, "{case}");
        assert_eq!(stream.trailing().len(), 1280, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(14).expect(case), 0..0, "{case}");
        assert_eq!(stream.data_range_for(15).expect(case), 0..0, "{case}");

        let case = "initialized full no trail";
        let mut about = AboutBlock::new();
        about.set_complete();
        for i in 0..TestMemoryBlocks::BLOCK_COUNT {
            stream.blocks.get_mut().write_meta_block(i, about, 80);
        }
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 1280, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..80, "{case}");
        assert_eq!(stream.data_range_for(1).expect(case), 80..160, "{case}");
        assert_eq!(stream.data_range_for(14).expect(case), 1120..1200, "{case}");
        assert_eq!(stream.data_range_for(15).expect(case), 1200..1280, "{case}");

        let case = "initialized full with trail";
        write_meta!(stream.blocks.get_mut(); @13 0 8 80, @14 172 0 80, @15 92 0 80);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        assert_eq!(stream.data().len(), 1112, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 168, "{case}");

        let case = "data at range with unspilled blocks";
        let range = stream.data_range_for(12).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 80 + 80 - 8, "{case}");
        assert_eq!(stream.data()[range].len(), 80, "{case}");
        let range = stream.data_range_for(13).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 80 - 8, "{case}");
        assert_eq!(stream.data()[range].len(), 80 - 8, "{case}");

        let case = "data range at trail block is not available";
        assert_eq!(stream.data_range_for(14).expect_err(case), 16, "{case}");
        assert_eq!(stream.data_range_for(15).expect_err(case), 16, "{case}");

        let case = "data at range with spilled blocks";
        write_meta!(stream.blocks.get_mut(); @13 0 8 80, @14 32 32 80, @15 16 10 80);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        let range = stream.data_range_for(12).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 4 * 80 - 10, "{case}");
        assert_eq!(stream.data()[range].len(), 80, "{case}");
        let range = stream.data_range_for(13).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 3 * 80 - 10, "{case}");
        assert_eq!(stream.data()[range].len(), 80 + 32, "{case}");
        let range = stream.data_range_for(14).expect(case);
        let expected = 80 - 32 + 80 - 10;
        assert_eq!(stream.data()[range.start..].len(), expected, "{case}");
        assert_eq!(stream.data()[range].len(), 80 - 32 + 16, "{case}");
        let range = stream.data_range_for(15).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 80 - 16 - 10, "{case}");
        assert_eq!(stream.data()[range].len(), 80 - 16 - 10, "{case}");

        let case = "data at range with incomplete blocks";
        write_meta!(stream.blocks.get_mut(); @13 0 8 80, !14 32 40 40, !15 0 0 0);
        stream.load().unwrap();
        verify_stream!(stream);
        stream.initialize().unwrap();
        let range = stream.data_range_for(12).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 80 + 80 + 40, "{case}");
        assert_eq!(stream.data()[range].len(), 80, "{case}");
        let range = stream.data_range_for(13).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 80 + 40, "{case}");
        assert_eq!(stream.data()[range].len(), 80 + 32, "{case}");
        let range = stream.data_range_for(14).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 40 - 32, "{case}");
        assert_eq!(stream.data()[range].len(), 40 - 32, "{case}");
        let range = stream.data_range_for(15).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 0, "{case}");
        assert_eq!(stream.data()[range].len(), 0, "{case}");

        let case = "data range at block beyond stream size is not available";
        assert_eq!(stream.data_range_for(99).expect_err(case), 99, "{case}");
    }

    #[test]
    fn stream_read_dirty() {
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();

        let case = "initial append";
        let written = stream.append(&TEST_BYTES_REPEATING[..88]);
        assert_eq!(written.unwrap(), 88);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        stream.sync().unwrap();
        assert_meta!(stream.blocks.get_mut(); @0 0 80, !1 8 0);

        let case = "append after sync";
        let written = stream.append(&TEST_BYTES_REPEATING[..88]);
        assert_eq!(written.unwrap(), 88);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 88, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        let range = stream.data_range_for(0).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 88, "{case}");
        assert_eq!(stream.data()[range].len(), 88, "{case}");
        let range = stream.data_range_for(1).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 0, "{case}");
        assert_eq!(stream.data()[range].len(), 0, "{case}");

        stream.sync().unwrap();
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 8 72, !2 16 0);

        let case = "append until the end with trail after sync";
        let written = stream.append(&TEST_BYTES_REPEATING[..1280]);
        assert_eq!(written.unwrap(), 1280 - 88 - 88);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 88 + 88, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        let range = stream.data_range_for(0).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 88 + 88, "{case}");
        assert_eq!(stream.data()[range].len(), 88, "{case}");
        let range = stream.data_range_for(1).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 88, "{case}");
        assert_eq!(stream.data()[range].len(), 88, "{case}");
        let range = stream.data_range_for(2).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 0, "{case}");
        assert_eq!(stream.data()[range].len(), 0, "{case}");

        stream.sync().unwrap();
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 8 72, @2 16 64, @3 1216 0, @15 256 0);

        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();

        let case = "spilled first append";
        let written = stream.append_with_opts(&TEST_BYTES_REPEATING[..88], true);
        assert_eq!(written.unwrap(), 88);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        let case = "spilled second append";
        let written = stream.append(&TEST_BYTES_REPEATING[..16]);
        assert_eq!(written.unwrap(), 16);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 0, "{case}");
        assert_eq!(stream.spilled().len(), 0, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect(case), 0..0, "{case}");

        stream.sync().unwrap();
        assert_meta!(stream.blocks.get_mut(); @0 88 0, !1 8 24);

        let case = "spilled append after sync";
        let written = stream.append(&TEST_BYTES_REPEATING[..16]);
        assert_eq!(written.unwrap(), 16);
        verify_stream!(stream);
        assert_eq!(stream.data().len(), 16, "{case}");
        assert_eq!(stream.spilled().len(), 88, "{case}");
        assert_eq!(stream.trailing().len(), 0, "{case}");
        assert_eq!(stream.data_range_for(0).expect_err(case), 1, "{case}");
        let range = stream.data_range_for(1).expect(case);
        assert_eq!(stream.data()[range.start..].len(), 16, "{case}");
        assert_eq!(stream.data()[range].len(), 16, "{case}");

        stream.sync().unwrap();
        assert_meta!(stream.blocks.get_mut(); @0 88 0, !1 8 40);
    }

    #[test]
    fn stream_lock() {
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        let locked = stream.lock();

        let case = "lock is busy";
        let err = stream.try_lock().expect_err(case);
        assert_eq!(err, StreamError::Busy, "{case}");

        let case = "append on stream is busy";
        let err = stream.append(&[]).expect_err(case);
        assert_eq!(err, StreamError::Busy, "{case}");

        let case = "append on locked succeeds";
        let written = LockedStream::append(&locked, &TEST_BYTES_REPEATING[..16]).expect(case);
        assert_eq!(written, 16, "{case}");
    }

    #[test]
    #[should_panic(expected = "unrelated lock")]
    fn stream_lock_mismatch() {
        let mut stream_a = Stream::new(TestMemoryBlocks::new());
        let mut stream_b = Stream::new(TestMemoryBlocks::new());
        stream_a.initialize().unwrap();
        stream_b.initialize().unwrap();

        let locked_b = stream_b.lock();
        stream_a
            .append_with_opts_locked(&[], false, &locked_b.1)
            .unwrap();
    }

    #[test]
    fn stream_trail_size() {
        macro_rules! assert_trail_size {
            ($stream:expr, $size:expr, $case:expr) => {
                let about = $stream.blocks.get_mut().read_meta_about(15);
                assert_eq!(
                    $stream.trail_size(15, about).expect($case),
                    $size,
                    "{}",
                    $case
                );
            };
        }
        macro_rules! assert_trail_size_err {
            ($stream:expr, $err:expr, $case:expr) => {
                let about = $stream.blocks.get_mut().read_meta_about(15);
                let err = $stream.trail_size(15, about).expect_err($case);
                assert_eq!(err.to_string(), $err, "{}", $case);
            };
        }

        // Write all blocks as complete to avoid any verification failures.
        let mut blocks = TestMemoryBlocks::new();
        let mut about = AboutBlock::new();
        about.set_complete();
        for i in 0..TestMemoryBlocks::BLOCK_COUNT {
            blocks.write_meta_block(i, about, 80);
        }

        let case = "incomplete block has no trail";
        write_meta!(blocks; !15 0 16 16);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 0, case);

        let case = "final block has a trail";
        write_meta!(blocks; @15 0 16 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 16, case);

        let case = "final block trail larger than data section size fails";
        write_meta!(blocks; @15 0 81 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(
            stream,
            Inconsistency::TrailBytesTooLarge(15),
            Inconsistency::TrailSpilledBytesTooLarge(15)
        );
        assert_trail_size_err!(
            stream,
            "blockstream: trail too large: meta section corrupted?",
            case
        );

        let case = "multi-block append returns a trail";
        write_meta!(blocks; @12 0 16 80, @13 248 0 80, @14 168 0 80, @15 88 0 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 256, case);

        let case = "multi-block append trail larger than data section size fails";
        write_meta!(blocks; @12 0 81 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(
            stream,
            Inconsistency::TrailBytesTooLarge(12),
            Inconsistency::TrailSpilledBytesTooLarge(12)
        );
        assert_trail_size_err!(
            stream,
            "blockstream: trail too large: meta section corrupted?",
            case
        );

        let case = "multi-block append aligned on the final block returns no trail";
        write_meta!(blocks; @12 0 16 80, @13 240 0 80, @14 160 0 80, @15 80 0 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 0, case);

        let case = "spill followed by an aligned append returns no trail";
        write_meta!(blocks; @12 0 16 80, @13 168 0 80, @14 88 0 80, @15 8 0 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 0, case);

        let case = "all blocks are spilling returns a trail";
        let mut about = AboutBlock::new();
        about.set_complete();
        for i in 1..=TestMemoryBlocks::BLOCK_COUNT {
            about.set_spilled_bytes(i * 80 + 8);
            blocks.write_meta_block(TestMemoryBlocks::BLOCK_COUNT - i, about, 80);
        }
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream);
        assert_trail_size!(stream, 1280, case);

        let case = "non-contiguous spilled sequence fails";
        write_meta!(blocks; @14 0 0 80, @15 88 0 80);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::SpilledBytesUneven(14, 0));
        assert_trail_size_err!(
            stream,
            "blockstream: missing trail: meta section corrupted?",
            case
        );

        let case = "incomplete block before spilled block fails";
        write_meta!(blocks; !14 0 16 16);
        let mut stream = Stream::new(blocks.clone());
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::CompleteSequenceBroken(15, 14));
        assert_trail_size_err!(
            stream,
            "blockstream: block not complete: meta section corrupted?",
            case
        );
    }

    #[test]
    fn stream_initialize() {
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty stream should succeed";
        stream.initialize().expect(case);
        assert_positions!(stream, (0, 0, 1280), "{case}");

        let case = "first incomplete block should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40]);
        assert_meta!(stream.blocks.get_mut(); !0 0 40);
        stream.initialize().expect(case);
        assert_positions!(stream, (40, 40, 1280), "{case}");

        let case = "empty block following complete should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40]);
        assert_meta!(stream.blocks.get_mut(); @0 0 0);
        stream.initialize().expect(case);
        assert_positions!(stream, (80, 80, 1280), "{case}");

        let case = "spilled block should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..88]);
        assert_meta!(stream.blocks.get_mut(); !2 8 0);
        stream.initialize().expect(case);
        assert_positions!(stream, (168, 168, 1280), "{case}");

        let case = "spilled written block should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..32]);
        assert_meta!(stream.blocks.get_mut(); !2 8 40);
        stream.initialize().expect(case);
        assert_positions!(stream, (200, 200, 1280), "{case}");

        let case = "last block should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..8 * 70]);
        append_stream!(stream, &TEST_BYTES_REPEATING[..500]);
        assert_meta!(stream.blocks.get_mut(); !15 60 0);
        stream.initialize().expect(case);
        assert_positions!(stream, (1260, 1260, 1280), "{case}");

        let case = "full stream should succeed";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40], 20);
        assert_meta!(stream.blocks.get_mut(); @15 60 20);
        stream.initialize().expect(case);
        assert_positions!(stream, (1280, 1280, 1260), "{case}");
    }

    #[test]
    fn stream_initialize_corrupted() {
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();

        append_stream!(stream, &TEST_BYTES_REPEATING[..8 * 30]);
        assert_meta!(stream.blocks.get_mut(); @2 80 0);

        let case = "nonzero parity on empty block should fail";
        stream.blocks.get_mut().write_meta_parity(3, 0x01010101);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::EmptyNonZeroParity(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: parity mismatch: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "zero parity on incomplete block should fail";
        write_meta!(stream.blocks.get_mut(); !3 8 0 8);
        stream.blocks.get_mut().write_meta_parity(3, 0x00000000);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::MetaParityMismatch(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: parity mismatch: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "bad crc32 on incomplete block should fail";
        write_meta!(stream.blocks.get_mut(); !3 8 0 0);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::DataChecksumMismatch(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: crc32 mismatch: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "spilled bytes larger or equal to data section size should fail";
        write_meta!(stream.blocks.get_mut(); !3 80 0 80);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::SpilledBytesTooLarge(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: offset larger than data section size: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "trail bytes larger or equal to data section size should fail";
        write_meta!(stream.blocks.get_mut(); !3 8 80 80);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::TrailBytesTooLarge(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: offset larger than data section size: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "trail bytes smaller or equal to spilled bytes should fail";
        write_meta!(stream.blocks.get_mut(); !3 60 60 60);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::TrailBytesTooSmall(3));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: inconsistent offsets: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);

        let case = "preceding middle block meta corruption should not fail";
        stream.blocks.get_mut().write_meta_parity(1, 0x01010101);
        write_meta!(stream.blocks.get_mut(); !3 8 0 8);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::MetaParityMismatch(1));
        stream.initialize().expect(case);
        assert_positions!(stream, (248, 248, 1280), "{case}");

        let case = "preceding block meta corruption should fail";
        stream.blocks.get_mut().write_meta_parity(2, 0x01010101);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::MetaParityMismatch(1 | 2));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: preceding block(s) malformed: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "first block meta corruption should fail";
        stream.blocks.get_mut().write_meta_parity(0, 0x01010101);
        write_meta!(stream.blocks.get_mut(); @2 80 0 80);
        stream.load().unwrap();
        verify_stream!(stream, Inconsistency::MetaParityMismatch(0..=2));
        let err = stream.initialize().expect_err(case);
        let msg = "blockstream: preceding block(s) malformed: meta section corrupted?";
        assert_eq!(err.to_string(), msg, "{case}",);
    }

    #[test]
    fn stream_find_ending_block() {
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();

        let case = "empty stream";
        assert_eq!(unsafe { stream.find_ending_block() }, 0, "{case}");

        let case = "block 0 incomplete";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40]);
        assert_meta!(stream.blocks.get_mut(); !0 0 40);
        assert_eq!(unsafe { stream.find_ending_block() }, 0, "{case}");

        let case = "block 1 empty after aligned append";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40]);
        assert_meta!(stream.blocks.get_mut(); @0 0 0);
        assert_eq!(unsafe { stream.find_ending_block() }, 1, "{case}");

        let case = "block 2 incomplete after a spilled append";
        append_stream!(stream, &TEST_BYTES_REPEATING[..88]);
        assert_meta!(stream.blocks.get_mut(); @1 0 80, !2 8 0);
        assert_eq!(unsafe { stream.find_ending_block() }, 2, "{case}");

        let case = "stream is half -1 block full";
        append_stream!(stream, &TEST_BYTES_REPEATING[..352]);
        assert_meta!(stream.blocks.get_mut(); !6 40 0);
        assert_eq!(unsafe { stream.find_ending_block() }, 6, "{case}");

        let case = "stream is half full aligned";
        append_stream!(stream, &TEST_BYTES_REPEATING[..40]);
        assert_meta!(stream.blocks.get_mut(); @6 40 0);
        assert_eq!(unsafe { stream.find_ending_block() }, 7, "{case}");

        let case = "stream is half +1 block full spilled";
        append_stream!(stream, &TEST_BYTES_REPEATING[..88]);
        assert_meta!(stream.blocks.get_mut(); @7 0 80, !8 8 0);
        assert_eq!(unsafe { stream.find_ending_block() }, 8, "{case}");

        let case = "stream is full";
        append_stream!(stream, &TEST_BYTES_REPEATING[..560]);
        append_stream!(stream, &TEST_BYTES_REPEATING[..560], 72);
        assert_meta!(stream.blocks.get_mut(); @15 8 72);
        assert_eq!(unsafe { stream.find_ending_block() }, 15, "{case}");
    }

    #[test]
    fn stream_verify() {
        // The strategy here is to test only the inconsistencies. Verification
        // is present in other tests, including those that call append, which
        // ensures that there are no inconsistencies introduced during normal use.
        let mut stream = Stream::new(TestMemoryBlocks::new());

        let case = "empty stream should succeed";
        assert!(stream.verify(|status| eprintln!("{:?}", status)), "{case}");

        // Have block 0-4 which are correct so we can get references later.
        let case = "blocks 0-4 should succeed";
        write_meta!(stream.blocks.get_mut(); @0 0 0 80, @1 0 32 80, @2 88 0 80, @3 8 10 80, !4 30 79 79);
        stream.load().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)), "{case}");

        // Mangle previous blocks:
        // 3:  set trail bytes to value larger than data section size and
        //     make spilled bytes uneven
        // 4:  mangle parity, set trail bytes larger than the data section size
        write_meta!(stream.blocks.get_mut(); @3 18 73 80, !4 30 81 79);
        stream.blocks.get_mut().write_meta_parity(4, 0x01010101);
        // Create more blocks with inconsistencies introduced intentionally:
        // 5:  empty, but has some parity set
        // 6:  complete and spilling to block 7, 8, but has a bad CRC32
        // 7:  complete, fully spilling, but has trail bytes
        // 8:  complete with spilling and trail bytes too large
        // 9:  incomplete with a spill larger than data section size and
        //     trail bytes smaller than the spill size.
        // 10: incomplete with a spill and trail bytes equal to the size
        //     of the spill.
        stream.blocks.get_mut().write_meta_parity(5, 0x10101010);
        write_meta!(stream.blocks.get_mut(); @6 0 14 13, @7 90 14 80, @8 40 41 80, !9 81 40 80, !10 40 40 40);
        stream.load().unwrap();

        // Catch matches via flags, so it is easier to find the failed one.
        // The leftmost flag, if set, indicates that there are unmatched records.
        let mut match_flags = 0b0000_0000_0000_0000_0000_0000;
        stream.verify(|status| {
            macro_rules! handle_match {
                ($expression:expr; $($pattern:pat => $shift:literal),+) => {
                    match $expression {
                        $(
                            $pattern => match_flags |= 1 << $shift,
                        )+
                        _ => {
                            match_flags |= 1 << 23;
                            eprintln!("{:?}", status);
                        }
                    }
                };
            }
            use Inconsistency::*;
            match status {
                MetaParityMismatch(block) => handle_match!(block; 4 => 0),
                CompleteSequenceBroken(block, end) => {
                    handle_match!((block, end); (6, 4) => 1, (7, 4) => 2, (8, 4) => 3)
                }
                IncompleteRepeated(block, last) => {
                    handle_match!((block, last); (9, 4) => 4, (10, 4) => 5)
                }
                EmptySequenceBroken(block, start) => {
                    handle_match!((block, start); (6, 5) => 6, (7, 5) => 7, (8, 5) => 8,
                                                  (9, 5) => 9, (10, 5) => 10)
                }
                TrailBytesTooSmall(block) => handle_match!(block; 9 => 11, 10 => 12),
                TrailBytesTooLarge(block) => handle_match!(block; 4 => 13),
                TrailBytesUnexpected(block) => handle_match!(block; 7 => 14),
                TrailSpilledBytesTooLarge(block) => handle_match!(block; 3 => 15, 8 => 16),
                SpilledBytesTooLarge(block) => handle_match!(block; 9 => 17),
                SpilledBytesUneven(block, start) => {
                    handle_match!((block, start); (3, 1) => 18, (8, 7) => 19)
                }
                DataChecksumMismatch(block) => handle_match!(block; 4 => 20, 6 => 21),
                EmptyNonZeroParity(block) => handle_match!(block; 5 => 22),
            };
        });
        assert_eq!(
            match_flags, 0b0111_1111_1111_1111_1111_1111,
            "unmatched inconsistency matches: {match_flags:#026b}"
        );
    }

    #[test]
    fn stream_append_small() {
        // Appending 64 bytes each time.
        let data: [[u64; 8]; 6] = [
            array::from_fn(|p| (p as u64 + 8 * 0 + 1).to_le()),
            array::from_fn(|p| (p as u64 + 8 * 1 + 1).to_le()),
            array::from_fn(|p| (p as u64 + 8 * 2 + 1).to_le()),
            array::from_fn(|p| (p as u64 + 8 * 3 + 1).to_le()),
            array::from_fn(|p| (p as u64 + 8 * 4 + 1).to_le()),
            array::from_fn(|p| (p as u64 + 8 * 5 + 1).to_le()),
        ];

        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();

        // Append 1. Block 0 is partially written.
        let bytes = test_words_as_bytes(&data[0]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); !0 0 64);
        // Append 2. Spilling to block 1.
        let bytes = test_words_as_bytes(&data[1]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, !1 48 0);
        // Append 3. Spilling to block 2.
        let bytes = test_words_as_bytes(&data[2]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, !2 32 0);
        // Append 4. Spilling to block 3.
        let bytes = test_words_as_bytes(&data[3]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, @2 32 48, !3 16 0);
        // Append 5. Complete block 3.
        let bytes = test_words_as_bytes(&data[4]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, @2 32 48, @3 16 0);
        // Append 6. Wrapping the sequence. This one is similar to Append 1.
        let bytes = test_words_as_bytes(&data[5]);
        append_stream!(stream, bytes);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, @2 32 48, @3 16 0, !4 0 64);
        // Append 7. Zero-size append writes 0 bytes.
        append_stream!(stream, &bytes[..0]);
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, @2 32 48, @3 16 0, !4 0 64);
        // Append 8. Append fewer bytes, than the trail of incomplete block.
        append_stream!(stream, &255_u64.to_le_bytes());
        assert_meta!(stream.blocks.get_mut(); @0 0 16, @1 48 32, @2 32 48, @3 16 0, !4 0 72);

        // Check the data. CRC32 values were generated with python3, e.g.:
        // `hex(binascii.crc32(struct.pack("<" + "q"*10, *range(1, 11))))`
        let blocks = stream.blocks.get_mut();
        assert_eq!(blocks.read_meta_crc32(0), 0x91351cb9);
        assert_eq!(&blocks.read_data_block(0)[..8], &data[0][..]);
        assert_eq!(&blocks.read_data_block(0)[8..], &data[1][..2]);
        assert_eq!(blocks.read_meta_crc32(1), 0x3c2e2d64);
        assert_eq!(&blocks.read_data_block(1)[..6], &data[1][2..]);
        assert_eq!(&blocks.read_data_block(1)[6..], &data[2][..4]);
        assert_eq!(blocks.read_meta_crc32(2), 0x3a6f3c8e);
        assert_eq!(&blocks.read_data_block(2)[..4], &data[2][4..]);
        assert_eq!(&blocks.read_data_block(2)[4..], &data[3][..6]);
        assert_eq!(blocks.read_meta_crc32(3), 0x251e5fda);
        assert_eq!(&blocks.read_data_block(3)[..2], &data[3][6..]);
        assert_eq!(&blocks.read_data_block(3)[2..], &data[4][..]);
        assert_eq!(blocks.read_meta_crc32(4), 0xf25e0261);
        assert_eq!(&blocks.read_data_block(4)[..8], &data[5][..]);
        assert_eq!(&blocks.read_data_block(4)[8..], &[255, 0]);
        assert_positions!(stream, (392, 392, 1280), "stream_append_small");
    }

    #[test]
    fn stream_append_large() {
        macro_rules! assert_data {
            ($blocks:expr, $data:expr, $($args:tt)+) => {
                // CRC32 here and below were generated with python3, e.g.:
                // `hex(binascii.crc32(struct.pack("<" + "q"*10, *[0xcccccccc]*10)))`
                for block in 0..TestMemoryBlocks::BLOCK_COUNT {
                    assert_eq!($blocks.read_meta_crc32(block), 0x59fe31d4, $($args)+);
                    assert_eq!($blocks.read_data_block(block), &$data[..10], $($args)+);
                }
            };
        }

        let case = "whole stream in one aligned append";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        append_stream!(stream, &TEST_BYTES_REPEATING[..8 * 160]);
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 1200 0, @2 1120 0, @3 1040 0, @15 80 0);
        assert_data!(stream.blocks.get_mut(), &TEST_WORDS_REPEATING, "{case}");
        assert_positions!(stream, (1280, 1280, 1280), "{case}");

        let case = "whole stream in one append with spill";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        append_stream!(stream, &TEST_BYTES_REPEATING[..8 * 161], 8 * 160);
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 1208 0, @2 1128 0, @3 1048 0, @15 88 0);
        assert_data!(stream.blocks.get_mut(), &TEST_WORDS_REPEATING, "{case}");
        assert_positions!(stream, (1280, 1280, 0), "{case}");

        let case = "whole stream in two uneven appends with spill";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        append_stream!(stream, &TEST_BYTES_REPEATING[..4 * 161], 4 * 161);
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 564 0, !8 4 0);
        append_stream!(stream, &TEST_BYTES_REPEATING[4 * 161..8 * 161], 4 * 161 - 8);
        assert_meta!(stream.blocks.get_mut(); @0 0 80, @1 564 0, @8 4 76, @9 568 0, @15 88 0);
        assert_data!(stream.blocks.get_mut(), &TEST_WORDS_REPEATING, "{case}");
        assert_positions!(stream, (1280, 1280, 4 * 161), "{case}");
    }

    #[test]
    fn stream_append_end() {
        let init = || -> Stream<TestMemoryBlocks> {
            let mut stream = Stream::new(TestMemoryBlocks::new());
            stream.initialize().unwrap();
            // Fill the stream, leaving few 7 bytes at the end.
            for iteration in 0..9 {
                let words: [u64; 17] = array::from_fn(|_| (iteration as u64).to_le());
                let bytes = test_words_as_bytes(&words);
                append_stream!(stream, bytes);
            }
            assert_meta!(stream.blocks.get_mut(); @13 48 32, @14 104 0, !15 24 0);
            // CRC32 here and below were generated with python3, e.g.:
            // `hex(binascii.crc32(struct.pack("<" + "q"*3, *[8]*3)))`
            assert_eq!(stream.blocks.get_mut().read_meta_crc32(15), 0xd13a7264);
            assert_eq!(&stream.blocks.get_mut().read_data_block(15)[..3], &[8; 3]);
            assert_eq!(&stream.blocks.get_mut().read_data_block(15)[3..], &[0; 7]);
            assert_positions!(stream, (1224, 1224, 1280), "stream_append_end init");
            stream
        };

        let case = "aligned append at the end";
        let mut stream = init();
        let words = [9_u64.to_le(); 7];
        let bytes = test_words_as_bytes(&words);
        append_stream!(stream, bytes);
        let blocks = stream.blocks.get_mut();
        assert_meta!(blocks; @15 24 0);
        assert_eq!(blocks.read_meta_crc32(15), 0x110a413f, "{case}");
        assert_eq!(&blocks.read_data_block(15)[..3], &[8; 3], "{case}");
        assert_eq!(&blocks.read_data_block(15)[3..], &[9; 7], "{case}");
        assert_positions!(stream, (1280, 1280, 1280), "{case}");

        let case = "partial append";
        let mut stream = init();
        let words = [9_u64.to_le(); 17];
        let bytes = test_words_as_bytes(&words);
        append_stream!(stream, bytes, 56);
        let blocks = stream.blocks.get_mut();
        assert_meta!(blocks; @15 24 56);
        assert_eq!(blocks.read_meta_crc32(15), 0x110a413f, "{case}");
        assert_eq!(&blocks.read_data_block(15)[..3], &[8; 3], "{case}");
        assert_eq!(&blocks.read_data_block(15)[3..], &[9; 7], "{case}");
        assert_positions!(stream, (1280, 1280, 1280 - 56), "{case}");

        let case = "non-zero append to a full stream should fail";
        let err = stream.append(bytes).expect_err(case);
        stream.sync().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)));
        assert_meta!(stream.blocks.get_mut(); @15 24 56);
        assert_eq!(err.to_string(), "blockstream: stream is full", "{case}");

        let case = "zero append to a full stream should fail";
        let err = stream.append(&[]).expect_err(case);
        stream.sync().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)));
        assert_meta!(stream.blocks.get_mut(); @15 24 56);
        assert_eq!(err.to_string(), "blockstream: stream is full", "{case}");
    }

    #[test]
    fn stream_append_with_opts() {
        // Append is before the block boundary.
        let case = "append smaller than the block";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        let words = [8_u64.to_le(); 8];
        let bytes = test_words_as_bytes(&words);
        let written = stream.append_with_opts(bytes, true).expect(case);
        assert_eq!(written, bytes.len(), "{case}");
        stream.sync().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)));
        let blocks = stream.blocks.get_mut();
        assert_meta!(blocks; !0 64 0);
        // CRC32 here and below were generated with python3, e.g.:
        // `hex(binascii.crc32(struct.pack("<" + "q"*8, *[8]*8)))`
        assert_eq!(blocks.read_meta_crc32(0), 0xe6c8e9ee, "{case}");
        assert_eq!(&blocks.read_data_block(0)[..8], &[8; 8], "{case}");
        assert_eq!(&blocks.read_data_block(0)[8..], &[0; 2], "{case}");

        let case = "spilled append after the first append should fail";
        let err = stream
            .append_with_opts(test_words_as_bytes(&[]), true)
            .expect_err(case);
        let msg = "blockstream: spilled option is allowed only on the first append";
        assert_meta!(stream.blocks.get_mut(); !0 64 0);
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "append aligned to the block boundary";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        let words = [10_u64.to_le(); 10];
        let bytes = test_words_as_bytes(&words);
        let written = stream.append_with_opts(bytes, true).expect(case);
        assert_eq!(written, bytes.len(), "{case}");
        stream.sync().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)));
        let blocks = stream.blocks.get_mut();
        assert_meta!(blocks; @0 80 0);
        assert_eq!(blocks.read_meta_crc32(0), 0x17ae2e04, "{case}");
        assert_eq!(&blocks.read_data_block(0)[..10], &[10; 10], "{case}");
        assert_eq!(blocks.read_meta_crc32(1), 0x00000000, "{case}");
        assert_eq!(&blocks.read_data_block(1)[..], &[0; 10], "{case}");

        let case = "append larger than the block";
        let mut stream = Stream::new(TestMemoryBlocks::new());
        stream.initialize().unwrap();
        let words = [12_u64.to_le(); 12];
        let bytes = test_words_as_bytes(&words);
        let written = stream.append_with_opts(bytes, true).expect(case);
        assert_eq!(written, bytes.len(), "{case}");
        stream.sync().unwrap();
        assert!(stream.verify(|status| eprintln!("{:?}", status)));
        let blocks = stream.blocks.get_mut();
        assert_meta!(blocks; @0 96 0, !1 16 0);
        assert_eq!(blocks.read_meta_crc32(0), 0x60d53c5a, "{case}");
        assert_eq!(&blocks.read_data_block(0)[..10], &[12; 10], "{case}");
        assert_eq!(blocks.read_meta_crc32(1), 0x1c368dd2, "{case}");
        assert_eq!(&blocks.read_data_block(1)[..2], &[12; 2], "{case}");
        assert_eq!(&blocks.read_data_block(1)[2..], &[0; 8], "{case}");
    }

    #[test]
    fn about_block() {
        let mut about = AboutBlock::new();

        let case = "about block is empty";
        assert!(about.is_empty(), "{case}");
        assert!(!about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 0, "{case}");
        assert_eq!(about.trail_bytes(), 0, "{case}");

        let case = "about block is complete";
        about.set_complete();
        assert!(!about.is_empty(), "{case}");
        assert!(about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 0, "{case}");
        assert_eq!(about.trail_bytes(), 0, "{case}");

        let case = "about block is complete with spilled bytes";
        about.set_spilled_bytes(74);
        assert!(!about.is_empty(), "{case}");
        assert!(about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 74, "{case}");
        assert_eq!(about.trail_bytes(), 0, "{case}");

        let case = "about block is complete with spilled and trail bytes";
        about.set_trail_bytes(32);
        assert!(!about.is_empty(), "{case}");
        assert!(about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 74, "{case}");
        assert_eq!(about.trail_bytes(), 32, "{case}");

        let case = "about block is complete with max spilled and trail bytes";
        about.set_spilled_bytes((1 << 30) - 1);
        about.set_trail_bytes((1 << 28) - 1);
        assert!(!about.is_empty(), "{case}");
        assert!(about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 0x3fffffff, "{case}");
        assert_eq!(about.trail_bytes(), 0xfffffff, "{case}");

        let case = "about block is complete with spilled and trail bytes reset";
        about.set_spilled_bytes(0);
        about.set_trail_bytes(0);
        assert!(!about.is_empty(), "{case}");
        assert!(about.is_complete(), "{case}");
        assert_eq!(about.spilled_bytes(), 0, "{case}");
        assert_eq!(about.trail_bytes(), 0, "{case}");
    }

    #[test]
    fn about_block_parity() {
        let mut about = AboutBlock::new();

        let case = "about block is empty";
        assert!(about.verify(about.parity()), "{case}");
        assert_eq!(about.parity(), 0xffffffff, "{case}");

        let case = "about block is complete";
        about.set_complete();
        assert!(about.verify(about.parity()), "{case}");
        assert_eq!(about.parity(), 0xfffffffe, "{case}");

        let case = "about block is complete with spilled and trail bytes set";
        about.set_spilled_bytes(0b1010_1100_1001);
        about.set_trail_bytes(0b1111_0110_0101);
        let parity = about.parity();
        assert!(about.verify(parity), "{case}");
        //                 0b0000_0000_0000_0000_0010_1011_0010_0100
        //                   ^-----------------------------------^
        //                               spilled_bytes
        //                 0b0000_0000_0000_0000_1111_0110_0101_0001
        //                   ^--------------------------------^ ^--^
        //                               trail_bytes            flag
        //                 0b0000_0000_0000_0000_1101_1101_0111_0101
        //                   ^-------------------------------------^
        //                                    xor
        assert_eq!(parity, 0b1111_1111_1111_1111_0010_0010_1000_1010, "{case}");
    }

    #[test]
    #[should_panic(expected = "too much bytes")]
    fn about_block_spilled_too_much() {
        let mut about = AboutBlock::new();
        about.set_spilled_bytes(1 << 30);
    }

    #[test]
    #[should_panic(expected = "too much bytes")]
    fn about_block_trail_too_much() {
        let mut about = AboutBlock::new();
        about.set_trail_bytes(1 << 28);
    }

    static TEST_WORDS_REPEATING: [u64; 170] = [0xcccccccc_u64.to_le(); 170];
    static TEST_BYTES_REPEATING: &'static [u8] = test_words_as_bytes(&TEST_WORDS_REPEATING);

    #[inline(always)]
    const fn test_words_as_bytes(words: &[u64]) -> &[u8] {
        unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() << 3) }
    }

    #[derive(Clone, Debug)]
    struct TestMemoryBlocks(io::Cursor<Vec<u8>>);

    impl TestMemoryBlocks {
        const BLOCK_COUNT: u64 = 16;
        const BLOCK_BITS: u32 = 7;
        const BLOCK_SIZE: u64 = 1 << Self::BLOCK_BITS;

        #[inline(always)]
        fn new() -> Self {
            Self(io::Cursor::new(vec![
                0;
                (Self::BLOCK_SIZE * Self::BLOCK_COUNT)
                    as usize
            ]))
        }

        fn read_meta_about(&self, block: u64) -> AboutBlock {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            assert_eq!(
                &self.0.get_ref()[offset + 16..offset + META_SECTION_SIZE_BYTES as usize],
                &[0; META_SECTION_SIZE_BYTES as usize - 16],
                "only the first 16 bytes of the meta section must be used"
            );
            AboutBlock::try_from(&self.0.get_ref()[offset..offset + 8]).unwrap()
        }

        fn read_meta_parity(&self, block: u64) -> u32 {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            u32::from_le_bytes(
                self.0.get_ref()[offset + 8..offset + 12]
                    .try_into()
                    .unwrap(),
            )
        }

        fn read_meta_crc32(&self, block: u64) -> u32 {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            u32::from_le_bytes(
                self.0.get_ref()[offset + 12..offset + 16]
                    .try_into()
                    .unwrap(),
            )
        }

        fn write_meta_block(&mut self, block: u64, about: AboutBlock, size: usize) {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            self.0.get_mut()[offset..offset + 8].copy_from_slice(&<[u8; 8]>::from(about));
            if !about.is_empty() {
                self.write_meta_parity(block, about.parity());
            } else {
                self.write_meta_parity(block, 0);
            }
            self.write_meta_crc32(block, size);
        }

        fn write_meta_parity(&mut self, block: u64, parity: u32) {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            self.0.get_mut()[offset + 8..offset + 12].copy_from_slice(&parity.to_le_bytes());
        }

        fn write_meta_crc32(&mut self, block: u64, size: usize) {
            let offset = (block * Self::BLOCK_SIZE) as usize;
            let data_offset = offset + META_SECTION_SIZE_BYTES as usize;
            let crc32 = crc32fast::hash(&self.0.get_ref()[data_offset..data_offset + size]);
            self.0.get_mut()[offset + 12..offset + 16].copy_from_slice(&crc32.to_le_bytes());
        }

        fn read_data_block(
            &self,
            block: u64,
        ) -> [u64; ((Self::BLOCK_SIZE - META_SECTION_SIZE_BYTES) >> 3) as usize] {
            let mut words = [0; ((Self::BLOCK_SIZE - META_SECTION_SIZE_BYTES) >> 3) as usize];
            let mut offset = (block * Self::BLOCK_SIZE + META_SECTION_SIZE_BYTES) as usize;
            for pos in 0..words.len() {
                words[pos] = u64::from_le_bytes(
                    <[u8; 8]>::try_from(&self.0.get_ref()[offset..offset + 8]).unwrap(),
                );
                offset += 8;
            }
            words
        }
    }

    impl Blocks for TestMemoryBlocks {
        #[inline(always)]
        fn block_count(&self) -> u64 {
            Self::BLOCK_COUNT
        }

        #[inline(always)]
        fn block_shift(&self) -> u32 {
            Self::BLOCK_BITS
        }

        #[inline(always)]
        fn load_from(&mut self, block: u64, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<()> {
            self.0
                .seek(io::SeekFrom::Start(block << Self::BLOCK_BITS))?;
            for buf in bufs {
                self.0.read_exact(buf)?;
            }
            Ok(())
        }

        #[inline(always)]
        fn store_at(&mut self, block: u64, bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
            self.0
                .seek(io::SeekFrom::Start(block << Self::BLOCK_BITS))?;
            for buf in bufs {
                self.0.write_all(buf)?;
            }
            Ok(())
        }
    }
}
