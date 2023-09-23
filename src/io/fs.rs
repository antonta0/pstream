//! IO backed by regular files in a generic filesystem.
//!
//! The implementation is aimed to be as portable as Rust files, with few
//! tweaks for Linux. All IO is synchronous.
//!
//! For more details, see [`File`] and [`FileSequence`] documentation.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("file io works only on 64-bit platforms");

use std::{
    cmp, fs,
    hash::{Hash, Hasher},
    io,
    num::NonZeroU64,
    path::{Path, PathBuf},
    sync::atomic,
};

use crate::{Blocks, BlocksAllocator};

#[cfg(not(all(feature = "libc", target_os = "linux")))]
use std::io::{Read, Seek, Write};

#[cfg(all(feature = "libc", target_os = "linux"))]
use std::{os::fd::AsRawFd, ptr::NonNull};

/// A file implementing [`Blocks`].
///
/// These files can be used either directly, or via [`FileSequence`]. In both
/// cases, the behavior for creating and opening a file is the same.
///
/// Internally, it uses [`std::fs::File`] API.
///
/// # Linux specifics
///
/// The implementation has tweaks for Linux which are really optimizations
/// around the syscall API it provides.
///
/// When the file is created, it is pre-allocated with `fallocate`, so that
/// the actual space is guaranteed by the filesystem and blocks will not run
/// out of it during runtime.
///
/// Reads are doing a normal vectored IO via `p*` syscalls, saving on a `seek`.
/// The kernel is advised about the read pattern, which is sequential, to
/// double the read-ahead page count. After every read the page cache is
/// dropped for that range.
///
/// Writes are normally done in chunks of 8 MiB, which could be smaller or
/// larger, depending on the block size and maximum length of `iovec` per
/// syscall. These chunks are aligned at block size boundary. If the write fits
/// into a single syscall, it will be done with `RWF_DSYNC` flag. Otherwise,
/// `sync_file_range` is used to start asynchronous sync on a chunk, then
/// proceed writing the next chunk with asynchronous sync, followed by a wait
/// of a sync on the previous chunk. Since the pattern is append-only and no
/// overwrites are expected once the blocks are done, and reads are onto the
/// memory that is managed by a block stream, page cache is dropped every time
/// the chunk has been written. So, in the ideal case, page caches should not
/// consume more than a chunk size worth of memory.
///
/// # Handling sync errors
///
/// This implementation takes on a paranoid approach of failing any further
/// reads or writes if sync returns an error. The returned error is set to
/// [`FileSyncError`] to allow callers to detect this specific case. To recover,
/// the file has to be re-opened.
///
/// For more context, you can check a good summary on
/// [PostgreSQL wiki][pgsql-fsync], which also includes a link to fsyncgate
/// thread somewhere from 2018, which is an interesting read.
///
/// [pgsql-fsync]: https://wiki.postgresql.org/wiki/Fsync_Errors
#[derive(Debug)]
pub struct File {
    /// The open file. Using the regular Rust file representation. I don't see
    /// any issues with that. The file is dropped if the sync encounters an
    /// error at any time, preventing later operations.
    inner: Option<fs::File>,
    /// The number of blocks allocated for that file.
    block_count: u64,
    /// The size of a single block expressed as a power of two.
    block_shift: u32,
    /// A stamp associated with a [`FileSequence`], allowing the allocator to
    /// release only the files it created.
    stamp: Option<NonZeroU64>,
    /// An index associated with a [`FileSequence`], allowing the allocator to
    /// release the correct file.
    index: Option<WrappingSeq>,
}

impl File {
    /// Verifies whether `block_count` and `block_shift` are correct and
    /// returns the total size of the file in bytes. If not correct, an error
    /// of [`io::ErrorKind::InvalidInput`] kind with a message is returned.
    fn verify_input(block_count: u64, block_shift: u32) -> io::Result<u64> {
        if !(12..=28).contains(&block_shift) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "block shift must be between 12 and 28 inclusive",
            ));
        }
        if block_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "block count must be non-zero",
            ));
        }
        let total_size = block_count << block_shift;
        if total_size > i64::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "total blocks size too large",
            ));
        }
        Ok(total_size)
    }

    /// Verifies whether `bufs` have the structure guaranteed by `BlockStream`.
    fn verify_bufs<T: core::ops::Deref<Target = [u8]>>(
        bufs: &[T],
        block_size: usize,
    ) -> io::Result<()> {
        if bufs.len() & 1 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "odd number of bufs",
            ));
        }
        if bufs.len() > i32::MAX as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "too many bufs"));
        }
        let mut uneven_bufs = bufs
            .chunks(2)
            .skip_while(|pair| pair[0].len() + pair[1].len() == block_size);
        if let Some([left, right]) = uneven_bufs.next() {
            if left.len() + right.len() > block_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "uneven pair of bufs is too large",
                ));
            }
        }
        if uneven_bufs.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "uneven pair of bufs is not the last pair",
            ));
        }
        Ok(())
    }

    /// Creates a file at `path` with the provided `block_count` and
    /// `block_shift`.
    ///
    /// The resulting file length is set to the maximum based on the input
    /// arguments, which is to save from syncing metadata during operation.
    /// Additionally, on Linux the file is pre-allocated, preventing it from
    /// running out of space.
    ///
    /// # Errors
    ///
    /// If `block_count` is 0, `block_shift` is less than 12 or greater than 28,
    /// or the total file size is greater than [`i64::MAX`], then the error is
    /// of [`io::ErrorKind::InvalidInput`] kind with the message explaining the
    /// problem.
    ///
    /// In other cases returns the IO error from the underlying [`fs::File`]
    /// API or the operating system.
    pub fn create<P: AsRef<Path>>(path: P, block_count: u64, block_shift: u32) -> io::Result<File> {
        let size = File::verify_input(block_count, block_shift)?;

        let file = fs::File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        file.set_len(size)?;
        #[cfg(all(feature = "libc", target_os = "linux"))]
        {
            reserve(&file, size)?;
            double_readahead_pages(&file, size)?;
        }
        file.sync_all()?;
        Ok(File {
            block_count,
            block_shift,
            inner: Some(file),
            index: None,
            stamp: None,
        })
    }

    /// Opens a file at `path` with the given `block_shift`.
    ///
    /// The block count is calculated from the file length, which must be
    /// aligned.
    ///
    /// # Errors
    ///
    /// If `block_shift` is less than 12 or greater than 28, then the error is
    /// of [`io::ErrorKind::InvalidInput`] kind with the message explaining the
    /// problem.
    ///
    /// If file length is 0, larger than [`i64::MAX`], or not divisible by
    /// block size, the error of [`io::ErrorKind::InvalidData`] is returned
    /// along with the message.
    ///
    /// In other cases returns the IO error from the underlying [`fs::File`]
    /// API or the operating system.
    pub fn open<P: AsRef<Path>>(path: P, block_shift: u32) -> io::Result<File> {
        // A hack with a dummy block count just to validate the block shift.
        File::verify_input(1, block_shift)?;

        let file = fs::File::options().read(true).write(true).open(path)?;
        let size = file.metadata()?.len();
        if size == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zero file size"));
        }
        if size > i64::MAX as u64 {
            // FUTURE: There is FileTooLarge available via io_error_more feature.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file size too large",
            ));
        }
        if size & ((1 << block_shift) - 1) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file size not aligned to block size",
            ));
        }
        #[cfg(all(feature = "libc", target_os = "linux"))]
        double_readahead_pages(&file, size)?;

        Ok(File {
            block_count: size >> block_shift,
            block_shift,
            inner: Some(file),
            index: None,
            stamp: None,
        })
    }

    /// Sets the stamp and the index, which is used by the allocator, or
    /// [`FileSequence`] to be more specific.
    #[inline(always)]
    #[must_use]
    fn with_alloc_info(mut self, stamp: NonZeroU64, index: WrappingSeq) -> File {
        self.stamp = Some(stamp);
        self.index = Some(index);
        self
    }

    /// Returns the underlying open file, or an error otherwise. If the open
    /// file is not available, then it was dropped after sync has encountered
    /// an error, therefore the error returned is of [`io::ErrorKind::Other`]
    /// kind and [`FileSyncError`] value.
    #[inline(always)]
    fn inner(&mut self) -> io::Result<&mut fs::File> {
        self.inner
            .as_mut()
            .ok_or(io::Error::new(io::ErrorKind::Other, FileSyncError))
    }
}

impl Blocks for File {
    #[inline(always)]
    fn block_count(&self) -> u64 {
        self.block_count
    }

    #[inline(always)]
    fn block_shift(&self) -> u32 {
        self.block_shift
    }

    /// Loads the data from a file into `bufs` starting from `block`.
    ///
    /// There are two implementation: a generic one and a specialized for the
    /// Linux kernel. Generic implementation does `seek` to a `block`, followed
    /// by a vectored read. For Linux specifics, check
    /// [type][File#linux-specifics] documentation.
    ///
    /// # Errors
    ///
    /// If `bufs` length exceeds [`i32::MAX`], an error of
    /// [`io::ErrorKind::InvalidInput`] kind is returned.
    ///
    /// If the end of file is reached before the `bufs` has been filled, an
    /// error of [`io::ErrorKind::UnexpectedEof`] is returned. In other cases,
    /// returns the underlying IO error.
    ///
    /// Note, that sync errors persistently make the file unusable. See
    /// [type][File#handling-sync-errors] documentation for more details.
    fn load_from(&mut self, block: u64, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<()> {
        if bufs.len() > i32::MAX as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "too many bufs"));
        }
        self.read_exact_vectored_at(bufs, block << self.block_shift)
    }

    /// Stores the data from `bufs` by writing them to a file starting at
    /// `block`.
    ///
    /// There are two implementation: a generic one and a specialized for the
    /// Linux kernel. Generic implementation does `seek` to a `block`, followed
    /// by vectored write of buffers, finished with a single `flush`. This is
    /// not the most optimal pattern, but it is surely portable.
    ///
    /// For Linux specifics, check [type][File#linux-specifics] documentation.
    ///
    /// # Errors
    ///
    /// If `bufs` structure do not match what is guaranteed by `BlockStream`
    /// implementation, an error of [`io::ErrorKind::InvalidInput`] is returned.
    /// See [`Blocks::store_at`] for details.
    ///
    /// If the total length of `bufs` exceed the capcity of a file, an error
    /// of [`io::ErrorKind::OutOfMemory`] is returned. If the underlying file
    /// writes zero bytes, an error of [`io::ErrorKind::WriteZero`] is returned.
    /// In other cases, an IO error is returned.
    ///
    /// Note, that sync errors persistently make the file unusable. See
    /// [`File`] documentation for more details.
    fn store_at(&mut self, block: u64, bufs: &mut [io::IoSlice<'_>]) -> io::Result<()> {
        let offset = block << self.block_shift;
        let total_len = bufs.iter().map(|buf| buf.len()).sum::<usize>() as u64;
        if offset.saturating_add(total_len) > self.block_count << self.block_shift {
            // FUTURE: StorageFull seems to be a better choice, but requires
            // io_error_more feature.
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "write exceeds file capacity",
            ));
        }
        Self::verify_bufs(bufs, 1 << self.block_shift)?;
        self.write_all_vectored_at(bufs, offset)
    }
}

#[cfg(all(feature = "libc", target_os = "linux"))]
impl File {
    /// Returns the shift of a chunk for a single synced write operation.
    ///
    /// It is normally set to 8 MiB (23), unless the iovec limit is hit with
    /// smaller blocks, or the block size is larger than 8 MiB.
    #[inline(always)]
    #[must_use]
    fn chunk_shift(block_shift: u32) -> u32 {
        // Because there are two buffers per block, therefore this is halved to
        // get the number of blocks limited by the constant.
        let iovec_block_limit = libc::UIO_MAXIOV.ilog2() - 1;
        cmp::max(cmp::min(block_shift + iovec_block_limit, 23), block_shift)
    }

    /// Advises the kernel to drop pages.
    ///
    /// This is a one-time operation and not an advise about the future.
    #[inline(always)]
    fn free_cached_pages(&mut self, offset: u64, size: u64) -> io::Result<()> {
        let ret = unsafe {
            // The total size of a file is verified when created or opened.
            #[allow(clippy::cast_possible_wrap)]
            libc::posix_fadvise(
                self.inner()?.as_raw_fd(),
                offset as libc::off64_t,
                size as libc::off64_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if ret == 0 {
            return Ok(());
        }
        Err(io::Error::from_raw_os_error(ret))
    }

    /// Frees cached pages in complete aligned chunks of `shift` size, and is
    /// a no-op otherwise. This is used to reduce the rate of syscalls during
    /// writes.
    #[inline(always)]
    fn maybe_free_cached_pages(&mut self, offset: u64, size: u64, shift: u32) -> io::Result<()> {
        let start = offset >> shift;
        let end = (offset + size) >> shift;
        if start == end {
            return Ok(());
        }
        self.free_cached_pages(start << shift, end << shift)
    }

    /// A wrapper around `sync_file_range` that allows setting generic flags.
    #[inline(always)]
    fn sync_file_range(&mut self, offset: u64, size: u64, flags: libc::c_uint) -> io::Result<()> {
        let ret = unsafe {
            // The total size of a file is verified when created or opened.
            #[allow(clippy::cast_possible_wrap)]
            libc::sync_file_range(
                self.inner()?.as_raw_fd(),
                offset as libc::off64_t,
                size as libc::off64_t,
                flags,
            )
        };
        match ret {
            0 => Ok(()),
            -1 => Err(io::Error::last_os_error()),
            _ => unreachable!("sync_file_range: unexpected return {ret}"),
        }
    }

    /// Starts an asynchronous sync of data range starting at `offset`.
    #[inline(always)]
    fn sync_file_range_start(&mut self, offset: u64, size: u64) -> io::Result<()> {
        self.sync_file_range(offset, size, libc::SYNC_FILE_RANGE_WRITE)
    }

    /// Waits until the data range starting at `offset` has been synced.
    #[inline(always)]
    fn sync_file_range_wait(&mut self, offset: u64, size: u64) -> io::Result<()> {
        self.sync_file_range(
            offset,
            size,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER,
        )
    }

    #[inline(always)]
    fn read_vectored_at(
        &mut self,
        bufs: &mut [io::IoSliceMut<'_>],
        offset: u64,
    ) -> io::Result<usize> {
        let ret = unsafe {
            // The total size of a file is verified when created or opened.
            // The length of bufs is verified in `load_from`.
            #[allow(clippy::cast_possible_wrap)]
            libc::preadv(
                self.inner()?.as_raw_fd(),
                bufs.as_mut_ptr().cast::<libc::iovec>(),
                // The size of c_int is likely smaller than the constant anyway.
                #[allow(clippy::cast_possible_truncation)]
                cmp::min(bufs.len() as libc::c_int, libc::UIO_MAXIOV),
                offset as libc::off64_t,
            )
        };
        if ret >= 0 {
            #[allow(clippy::cast_sign_loss)]
            Ok(ret as usize)
        } else if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            unreachable!("preadv: unexpected return {ret}")
        }
    }

    #[inline(always)]
    fn write_vectored_at(&mut self, bufs: &[io::IoSlice<'_>], offset: u64) -> io::Result<usize> {
        let ret = unsafe {
            // The total size of a file is verified when created or opened.
            // The length of bufs is verified in `store_at`.
            #[allow(clippy::cast_possible_wrap)]
            libc::pwritev(
                self.inner()?.as_raw_fd(),
                bufs.as_ptr().cast::<libc::iovec>(),
                // The size of c_int is likely smaller than the constant anyway.
                #[allow(clippy::cast_possible_truncation)]
                cmp::min(bufs.len() as libc::c_int, libc::UIO_MAXIOV),
                offset as libc::off64_t,
            )
        };
        if ret >= 0 {
            #[allow(clippy::cast_sign_loss)]
            Ok(ret as usize)
        } else if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            unreachable!("pwritev: unexpected return {ret}")
        }
    }

    /// A vectored write that writes the data synchronously to the storage,
    /// removing the need to call `fdatasync` after write.
    #[inline(always)]
    fn write_vectored_at_dsync(
        &mut self,
        bufs: &[io::IoSlice<'_>],
        offset: u64,
    ) -> io::Result<usize> {
        let ret = unsafe {
            // The total size of a file is verified when created or opened.
            // The length of bufs is verified in `store_at`.
            #[allow(clippy::cast_possible_wrap)]
            libc::pwritev2(
                self.inner()?.as_raw_fd(),
                bufs.as_ptr().cast::<libc::iovec>(),
                // The size of c_int is likely smaller than the constant anyway.
                #[allow(clippy::cast_possible_truncation)]
                cmp::min(bufs.len() as libc::c_int, libc::UIO_MAXIOV),
                offset as libc::off64_t,
                libc::RWF_DSYNC,
            )
        };
        if ret >= 0 {
            #[allow(clippy::cast_sign_loss)]
            Ok(ret as usize)
        } else if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            unreachable!("pwritev2: unexpected return {ret}")
        }
    }

    #[inline(always)]
    fn read_exact_vectored_at(
        &mut self,
        mut bufs: &mut [io::IoSliceMut<'_>],
        offset: u64,
    ) -> io::Result<()> {
        let mut current = offset;
        while !bufs.is_empty() {
            // FUTURE: Use unix_file_vectored_at feature once stable.
            match self.read_vectored_at(bufs, current) {
                Ok(0) => break,
                Ok(read) => {
                    // FUTURE: Use io_slice_advance feature when stable.
                    advance_iovec(&mut bufs, read, |buf| {
                        // SAFETY: The iovec was already mutable. Advancing it does
                        // not violate the non-overlapping property.
                        io::IoSliceMut::new(unsafe {
                            // Get a mutable borrow to bytes by creating a new slice.
                            core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.len())
                        })
                    });
                    self.free_cached_pages(current, read as u64)?;
                    current = current.saturating_add(read as u64);
                }
                Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        if !bufs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill all buffers",
            ));
        }
        Ok(())
    }

    #[inline(always)]
    fn write_all_vectored_at(
        &mut self,
        mut bufs: &mut [io::IoSlice<'_>],
        offset: u64,
    ) -> io::Result<()> {
        let chunk_shift = File::chunk_shift(self.block_shift);
        let total_len = bufs.iter().map(|buf| buf.len()).sum::<usize>();

        // Fast path. The write fits into a single chunk. Make a direct synced
        // write in one call.
        if total_len <= 1 << chunk_shift {
            let mut total_written = 0usize;
            while !bufs.is_empty() {
                match self
                    .write_vectored_at_dsync(bufs, offset.saturating_add(total_written as u64))
                {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "failed to write all buffers",
                        ));
                    }
                    Ok(written) => {
                        total_written = total_written.saturating_add(written);
                        // FUTURE: Use io_slice_advance feature when stable.
                        advance_iovec(&mut bufs, written, io::IoSlice::new);
                    }
                    Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
                    Err(err) => return Err(err),
                }
            }
            return self.maybe_free_cached_pages(offset, total_written as u64, chunk_shift);
        }

        // Slow path. Here we make use of asynchronous sync while writing the
        // next chunk, and then waiting for that sync to complete before
        // proceeding to the next one.
        let bufs_per_chunk = 2 << (chunk_shift - self.block_shift);
        let mut offset = offset;
        let mut chunks = bufs.chunks_mut(bufs_per_chunk).peekable();
        let mut is_first = true;
        while let Some(mut chunk) = chunks.next() {
            let current = offset;
            while !chunk.is_empty() {
                match self.write_vectored_at(chunk, offset) {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "failed to write all buffers",
                        ));
                    }
                    Ok(written) => {
                        // FUTURE: Use io_slice_advance feature when stable.
                        advance_iovec(&mut chunk, written, io::IoSlice::new);
                        offset = offset.saturating_add(written as u64);
                    }
                    Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
                    Err(err) => return Err(err),
                }
            }

            let written = offset - current;
            if let Err(err) = self.sync_file_range_start(current, written) {
                self.inner.take();
                return Err(err);
            }
            if !is_first {
                // All preceding chunks are guaranteed to be the size of a chunk.
                let chunk_size = 1 << chunk_shift;
                let previous = current - chunk_size;
                if let Err(err) = self.sync_file_range_wait(previous, chunk_size) {
                    self.inner.take();
                    return Err(err);
                }
                self.maybe_free_cached_pages(previous, chunk_size, chunk_shift)?;
            }
            // Sync the last chunk, as the loop is going to terminate.
            if chunks.peek().is_none() {
                if let Err(err) = self.sync_file_range_wait(current, written) {
                    self.inner.take();
                    return Err(err);
                }
                self.maybe_free_cached_pages(current, written, chunk_shift)?;
            }
            is_first = false;
        }
        Ok(())
    }
}

#[cfg(not(all(feature = "libc", target_os = "linux")))]
impl File {
    #[inline(always)]
    fn read_exact_vectored_at(
        &mut self,
        mut bufs: &mut [io::IoSliceMut<'_>],
        offset: u64,
    ) -> io::Result<()> {
        // NOTE: This is different from the Linux implementation in that it
        // changes the position of the file. Violating this property saves from
        // doing two extra syscalls, and this property is not relevant for block
        // streams anyway, as they don't use the standard Read and Write traits.
        let file = self.inner()?;
        file.seek(io::SeekFrom::Start(offset))?;
        while !bufs.is_empty() {
            match file.read_vectored(bufs) {
                Ok(0) => break,
                Ok(read) => {
                    // FUTURE: Use io_slice_advance feature when stable.
                    advance_iovec(&mut bufs, read, |buf| {
                        // SAFETY: The iovec was already mutable. Advancing it does
                        // not violate the non-overlapping property.
                        io::IoSliceMut::new(unsafe {
                            // Get a mutable borrow to bytes by creating a new slice.
                            core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.len())
                        })
                    });
                }
                Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        if !bufs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill all buffers",
            ));
        }
        Ok(())
    }

    #[inline(always)]
    fn write_all_vectored_at(
        &mut self,
        mut bufs: &mut [io::IoSlice<'_>],
        offset: u64,
    ) -> io::Result<()> {
        let file = self.inner()?;
        // NOTE: See `read_exact_vectored_at` about use of Seek.
        file.seek(io::SeekFrom::Start(offset))?;
        // FUTURE: Use write_all_vectored feature when stable.
        while !bufs.is_empty() {
            match file.write_vectored(bufs) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write all buffers",
                    ));
                }
                // FUTURE: Use io_slice_advance feature when stable.
                Ok(written) => advance_iovec(&mut bufs, written, io::IoSlice::new),
                Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        if let Err(err) = file.flush() {
            self.inner.take();
            return Err(err);
        }
        Ok(())
    }
}

/// A special error type, indicating that the [`File`] is no longer usable,
/// because the underlying storage returned an error while syncing buffers.
#[derive(Debug)]
pub struct FileSyncError;

impl std::error::Error for FileSyncError {}

impl core::fmt::Display for FileSyncError {
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "file is unavailable due to prior sync error")
    }
}

/// Errors specific to the [`FileSequence`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum FileSequenceError {
    /// The error indicating the result of an attempt to parse options from the
    /// file name. The pattern that is checked is `<index:u16>-<shift:u32>`,
    /// where index is in hex and shift in decimal.
    ParseOptsError(ParseOptsErrorKind, String),
    /// The sequence is not monotonically increasing.
    Broken,
    /// Duplicate files with the same index found. This can happen when two
    /// files with the differnt shift appear somehow. The enclosed value is the
    /// non-unique index.
    Duplicate(u16),
    /// The sequence is locked, meaning this or another process has opened the
    /// sequence. In rare cases the lock could have been left after unclean
    /// shutdown, in which case check whether the process id the lock file
    /// points to is running.
    Locked,
    /// Attempt to release a file that is not the left-most file in the
    /// sequence. The enclosed value is the index of that file.
    OutOfOrder(u16),
    /// A file was encountered with the index outside the range of a sequence.
    /// The enclosed value is the index which is outside the range.
    OutOfRange(u16),
    /// The sequence is too long. The enclosed values are the start and the end
    /// of the range.
    TooLong(u16, u16),
    /// The file is not managed by this sequence.
    Unrecognized,
}

impl From<FileSequenceError> for io::Error {
    #[inline(always)]
    fn from(value: FileSequenceError) -> io::Error {
        io::Error::new(io::ErrorKind::Other, value)
    }
}

impl std::error::Error for FileSequenceError {}

impl core::fmt::Display for FileSequenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ParseOptsError(kind, ref opts) => match kind {
                ParseOptsErrorKind::BadIndex => {
                    write!(f, "filesequence: bad index: '{opts}'")
                }
                ParseOptsErrorKind::BadShift => {
                    write!(f, "filesequence: bad shift: '{opts}'")
                }
                ParseOptsErrorKind::MissingFields => {
                    write!(f, "filesequence: fields missing: '{opts}'")
                }
                ParseOptsErrorKind::TooManyFields => {
                    write!(f, "filesequence: too many fields: '{opts}'")
                }
            },
            Self::Broken => write!(f, "filesequence: indexes are not monotonically increasing"),
            Self::Duplicate(index) => write!(f, "filesequence: duplicate index {index:#06x}"),
            Self::Locked => write!(f, "filesequence: lock file is held by another instance"),
            Self::OutOfOrder(index) => write!(f, "filesequence: index {index:#06x} out of order"),
            Self::OutOfRange(index) => write!(f, "filesequence: index {index:#06x} out of range"),
            Self::TooLong(start, end) => {
                write!(f, "filesequence: {start:#06x}..{end:#06x} is too long")
            }
            Self::Unrecognized => write!(f, "filesequence: file does not belong to this sequence"),
        }
    }
}

/// The various kinds of failures that can happen when parsing options from
/// a file name.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParseOptsErrorKind {
    /// The index part cannot be parsed from the string.
    BadIndex,
    /// The shift part cannot be parsed from the string.
    BadShift,
    /// The string does not have enough fields separated by `-`.
    MissingFields,
    /// The string has too many fields separated by `-`.
    TooManyFields,
}

/// A [`BlocksAllocator`] implemented as a sequence of [`File`].
///
/// A sequence is identified by name and operated within a directory. Block
/// count and shift define the shape of newly allocated files. The shape of
/// previously allocated files is retrieved from the file system.
///
/// The layout of the directory is flat, assuming that files are reasonably
/// large and the memory limit will be hit before the directory blows up.
/// The meta, name in particular, is used to store the state necessary to
/// recover. It is composed of a prefix unique to the allocator, a sequence
/// number, also called *index* and a block shift, with which the file was
/// created. The length of the file is used to calculate the number of blocks.
///
/// The index is a `u16` that is always incrementing, starting from 1 wrapping
/// around and starting over again. Due to wrapping, there could be a maximum of
/// 2<sup>14</sup> &minus; 1 active files at a time.
///
/// There can be only one instance of a sequence with the same name and
/// directory. To maintain that property, a lock file is created when the
/// sequence is opened that includes this process id. The lock file is removed
/// when sequence is dropped, or it can be closed to capture any errors via
/// [`FileSequence::close`].
///
/// `BlocksAllocator` API is thread-safe with optimistic concurrency model,
/// relying on the operating system to handle conflicts. In other words,
/// concurrent access may result in an error, some of which can be retried.
/// It's a bit clunky of an API if one wants to handle retries, however.
///
/// Note, that directory managed by this sequence is not flushed to disk on
/// operating systems other than Linux. This is usually necessary after
/// creating or removing files, so keep that in mind.
///
/// # Tips for manually operating on the files
///
/// Before doing any manual operations, ensure that there is no lock file for
/// that sequence, and then create it via `ln -s <msg> <name>.lock`
/// within the directory of that sequence, where `<msg>` can be anything, but
/// try to make sure it does not point to an actual file to avoid confusion.
/// This will prevent software from opening that sequence by accident.
///
/// Renaming the sequence is OK, just ensure to manually create two lock files -
/// one for the old name and one for the new one.
///
/// You can remove files in between and shift indexes, just make sure it is safe
/// from the application point of view, and there is no data spanning between
/// the files. Make sure that the end result is a monotonically-increasing
/// sequence.
///
/// Block shift in the name should not be modified in general, as it will likely
/// result in garbage when reading the file back.
///
/// May be there will be a tool for mangling the files later, that will assist
/// in this and even more. May be...
#[derive(Debug)]
pub struct FileSequence {
    /// The root directory to scan for, and create the sequence of files. Also
    /// the place for a lock file.
    root: Dir,
    /// An arbitrary name of this file sequence. Acts as a unique identifier.
    name: String,
    /// The default block count to allocate files with.
    block_count: u64,
    /// The default block shift to allocate files with.
    block_shift: u32,
    /// The stamp associated with this file sequence. It really just a hash
    /// of the root directory path and the name.
    stamp: NonZeroU64,
    /// The starting index of this sequence. Only files with that index can
    /// be released. This index never advances past `end`. The value may wrap.
    start: atomic::AtomicU16,
    /// The ending index of this sequence. Files are allocated with that index.
    /// The value may wrap.
    end: atomic::AtomicU16,
    /// The lock file held by this sequence.
    lock: LockFile,
}

impl FileSequence {
    /// Parses options from the extension of a file name. The opts are in form
    /// of `xxxx-s`, where `x` is a hex-encoded character, 4 of which make up
    /// an index, and `s` is a decimal string encoding a block shift. On
    /// success, both index and shift are returned. If string is malformed,
    /// a [`ParseOptsError`] is returned.
    #[inline(always)]
    fn parse_opts(opts: &str) -> Result<(WrappingSeq, u32), FileSequenceError> {
        use ParseOptsErrorKind::{BadIndex, BadShift, MissingFields, TooManyFields};
        let mut parts = opts.split('-');
        let index = WrappingSeq(
            parts
                .next()
                .map(|index| u16::from_str_radix(index, 16))
                .expect("first field is always present")
                .map_err(|_| FileSequenceError::ParseOptsError(BadIndex, opts.to_owned()))?,
        );
        let block_shift = parts
            .next()
            .map(str::parse)
            .ok_or_else(|| FileSequenceError::ParseOptsError(MissingFields, opts.to_string()))?
            .map_err(|_| FileSequenceError::ParseOptsError(BadShift, opts.to_string()))?;
        if parts.next().is_some() {
            return Err(FileSequenceError::ParseOptsError(
                TooManyFields,
                opts.to_string(),
            ));
        }
        Ok((index, block_shift))
    }

    /// Opens a new file sequence at `root` directory with a given name.
    ///
    /// The `block_count` and `block_shift` define the block shape for new
    /// allocations.
    ///
    /// This function does not validate whether the file sequence is contiguous.
    /// If this is not the case, future `retrieve` will result in an error.
    ///
    /// # Errors
    ///
    /// If `block_count` is 0, `block_shift` is less than 12 or greater than 28,
    /// or the total file size is greater than [`i64::MAX`], then the error is
    /// of [`io::ErrorKind::InvalidInput`] kind with the message explaining the
    /// problem.
    ///
    /// Custom errors are of [`io::ErrorKind::Other`] and the value is set to
    /// one of the following:
    ///
    /// -   [`FileSequenceError::Locked`] if a lock file already exists.
    /// -   [`FileSequenceError::OutOfRange`] if the files managed by this
    ///     sequence have indexes too far from each other, such that they are
    ///     incomparable.
    /// -   [`FileSequenceError::TooLong`] if the distance between the smallest
    ///     and the largest index is greater than 2<sup>14</sup> &minus; 1.
    ///
    /// On Linux, if the `root` path is longer than 255 bytes or has unexpected
    /// `0x00` byte, an error of [`io::ErrorKind::InvalidInput`] will be
    /// returned.
    ///
    /// In other cases, the error is propagated from the underlying IO.
    ///
    /// # Panics
    ///
    /// In an extremely unlikely case, when the hash of this sequence happens
    /// to be 0. Is this even possible?
    pub fn open<P: AsRef<Path>>(
        root: P,
        name: &str,
        block_count: u64,
        block_shift: u32,
    ) -> io::Result<FileSequence> {
        File::verify_input(block_count, block_shift)?;

        let root = Dir::open(root.as_ref().canonicalize()?)?;
        let lock = LockFile::acquire(root.path.join(format!("{name}.lock"))).map_err(|err| {
            if err.kind() == io::ErrorKind::AlreadyExists {
                FileSequenceError::Locked.into()
            } else {
                err
            }
        })?;
        root.sync()?;

        let (mut start, mut end): (Option<WrappingSeq>, Option<WrappingSeq>) = (None, None);
        for entry in root.read_dir()? {
            let path = entry?.path();
            if !path.is_file() {
                continue;
            }
            if let Some((prefix, opts)) = path.file_stem().zip(path.extension()) {
                if prefix != name {
                    continue;
                }
                let (index, _) = FileSequence::parse_opts(opts.to_str().unwrap_or_default())?;
                let start_index = *start.get_or_insert(index);
                let end_index = *end.get_or_insert(index.inc());
                if index >= start_index && index < end_index {
                    continue;
                } else if index < start_index {
                    start.replace(index);
                } else if index >= end_index {
                    end.replace(index.inc());
                } else {
                    return Err(FileSequenceError::OutOfRange(index.into()).into());
                }
            }
        }
        start
            .zip(end)
            .map(|(start, end)| match start.partial_cmp(&end) {
                Some(cmp::Ordering::Greater | cmp::Ordering::Equal) => {
                    unreachable!("start never reached end")
                }
                Some(_) => Ok(()),
                None => Err(FileSequenceError::TooLong(start.into(), end.into())),
            })
            .transpose()?;

        let mut stamp = std::collections::hash_map::DefaultHasher::new();
        root.path.hash(&mut stamp);
        name.hash(&mut stamp);

        Ok(FileSequence {
            root,
            name: name.into(),
            block_count,
            block_shift,
            stamp: stamp.finish().try_into().expect("stamp should be non-zero"),
            start: start.map_or(1, WrappingSeq::into).into(),
            end: end.map_or(1, WrappingSeq::into).into(),
            lock,
        })
    }

    /// Close this file sequence gracefully, releasing the lock file.
    ///
    /// This function allows catching errors which would have been ignored
    /// during `drop`. In practice it differs from dropping the value only by
    /// a sync on the directory.
    ///
    /// # Errors
    ///
    /// An error is returned if an IO error is encountered.
    pub fn close(mut self) -> io::Result<()> {
        // Both LockFile and Dir have destructors that will run after this
        // function returns. In case of a LockFile, the actual file is not
        // double-released on drop. As for Dir, the errors are not expected
        // on drop and it will panic if encounters one.
        self.lock.release().and_then(|_| self.root.sync())
    }

    /// A convenience function to build the full path of the file in sequence.
    #[inline(always)]
    #[must_use]
    fn get_filename(&self, index: WrappingSeq, block_shift: u32) -> PathBuf {
        self.root.path.join(format!(
            "{name}.{index:04x}-{shift}",
            name = self.name,
            shift = block_shift
        ))
    }
}

unsafe impl BlocksAllocator for FileSequence {
    type Blocks = File;

    /// Allocates new file-backed [`Blocks`].
    ///
    /// The next available index is selected. The file is created with the
    /// default block shape, as specified during [`FileSequence::open`]. The
    /// file is created in exactly same way as via [`File::create`].
    ///
    /// Concurrent behavior depends on the underlying [`fs::File`]
    /// implementation, and hence the operating system. Generally, attempts
    /// to create a file are atomic.
    ///
    /// # Errors
    ///
    /// Propagates the underlying IO error if encountered.
    ///
    /// If the sequence is too long and no more files can be allocated, an
    /// error of [`io::ErrorKind::Other`] with [`FileSequenceError::TooLong`]
    /// value is returned. This error is non-retriable, unless more space is
    /// made available via [`FileSequence::release`]. The maximum size of the
    /// sequence is 2<sup>14</sup> &minus; 1.
    ///
    /// Concurrent attempts to create a file may or may not result in
    /// [`io::ErrorKind::AlreadyExists`]. In which case attempt could be
    /// retried, unless the file with the same index has been injected into the
    /// file system, in which case this check will break down. Read the note
    /// below about directory sync as well.
    ///
    /// On Linux, if the sync of the underlying directory has failed, all
    /// future attempts to allocate a file will fail, as it could have been
    /// created, and the internal state was not updated to reflect that.
    ///
    /// # Panics
    ///
    /// If the internal state is double-incremented on the same index. This is
    /// supposed to be impossible as long as the operating system creating a
    /// file doing that atomically.
    fn alloc(&self) -> io::Result<File> {
        // Taking relaxed ordering, since only atomic value is of interest.
        // Both values are managed internally, and concurrent call should result
        // only in a single one to succeed.
        let start = WrappingSeq(self.start.load(atomic::Ordering::Relaxed));
        let end = WrappingSeq(self.end.load(atomic::Ordering::Relaxed));
        if start.distance(end.inc()).is_none() {
            return Err(FileSequenceError::TooLong(start.into(), end.into()).into());
        }

        // Under normal operation, a previous attempt to create a file could
        // finish followed by a failed sync. In this case allocations will
        // always fail, as the file already exists and the counter has not
        // been incremented.
        let path = self.get_filename(end, self.block_shift);
        let file = File::create(path, self.block_count, self.block_shift)?
            .with_alloc_info(self.stamp, end);
        self.root.sync()?;

        // This is just to be on a safe side, and to communicate the intent.
        // The code above must finish before the atomic is incremented.
        atomic::compiler_fence(atomic::Ordering::SeqCst);
        // Creating a file concurrently is assumed to complete only for a
        // single attempt, which should prevent double counting.
        assert_eq!(self.end.fetch_add(1, atomic::Ordering::Relaxed), end.into());
        Ok(file)
    }

    /// Releases previously allocated file-backed [`Blocks`].
    ///
    /// Files can be released only from the start to avoid breaking the
    /// sequence. Only the files that has been returned by this allocator can
    /// be released.
    ///
    /// Concurrent behavior depends on the implementation of [`fs::remove_file`].
    ///
    /// # Errors
    ///
    /// Propagates the underlying IO error if encountered.
    ///
    /// Otherwise, returns an error with [`io::ErrorKind::Other`] kind and a
    /// value depending on the case:
    ///
    /// -   [`FileSequenceError::Unrecognized`] if the file has not been
    ///     created by this allocator.
    /// -   [`FileSequenceError::OutOfOrder`] if attempting to remove a file
    ///     that is not the left-most file in the sequence.
    /// -   [`FileSequenceError::OutOfRange`] if removing a file that has
    ///     already been removed. Normally, this may be observed during
    ///     concurrent attempts to release, in which case the file can be
    ///     simply discarded.
    ///
    /// On Linux, if the sync of the underlying directory has failed, all
    /// future attempts to remove a file will fail, as it could have been
    /// removed before the failed sync, and the internal state was not updated.
    ///
    /// # Panics
    ///
    /// If the inner state is double-incremented for the same file. This should
    /// be impossible as long as the operating system removing a file is atomic.
    fn release(&self, blocks: File) -> Result<(), (File, io::Error)> {
        if blocks.stamp.filter(|stamp| *stamp == self.stamp).is_none() {
            return Err((blocks, FileSequenceError::Unrecognized.into()));
        }

        let Some(index) = blocks.index else {
            return Err((blocks, FileSequenceError::Unrecognized.into())) };
        // Taking relaxed order, because only the atomic value is relevant.
        // The callers are not supposed to pass the value with the same index
        // in arguments, and even if they do, only one of them would succeed.
        let start = WrappingSeq(self.start.load(atomic::Ordering::Relaxed));
        let end = WrappingSeq(self.end.load(atomic::Ordering::Relaxed));
        if index > start && index < end {
            return Err((blocks, FileSequenceError::OutOfOrder(index.into()).into()));
        } else if index != start || start == end {
            return Err((blocks, FileSequenceError::OutOfRange(index.into()).into()));
        }

        // In case of a successful remove followed by a failed sync, any
        // retry will fail because the counter has not been incremented.
        let path = self.get_filename(index, blocks.block_shift);
        match fs::remove_file(path) {
            Ok(_) => {}
            Err(err) => return Err((blocks, err)),
        }
        match self.root.sync() {
            Ok(_) => {}
            // Returns blocks that don't have the file anymore. That should be
            // alright. The file still has the open descriptor.
            Err(err) => return Err((blocks, err)),
        }

        // This is just to be on a safe side, and to communicate the intent.
        // The code above must finish before the atomic is incremented.
        atomic::compiler_fence(atomic::Ordering::SeqCst);
        // Removing a file is expected to be atomic and that should prevent
        // double counting. The assert ensures this property is maintained.
        // Using relaxed order, because relying on atomicity via file removal.
        assert_eq!(
            self.start.fetch_add(1, atomic::Ordering::Relaxed),
            index.into()
        );
        Ok(())
    }

    /// Retrieves all files within this sequence in the order they were
    /// allocated.
    ///
    /// The returned files are guaranteed to be monotonically increasing and
    /// correspond to the internal state of the sequence, as long as the
    /// function returns successfully.
    ///
    /// Concurrent calls are OK, but if there are competing allocation or
    /// release, the errors may be weird and unpredictable. A retry will scan
    /// the directory again and is very inefficient.
    ///
    /// # Errors
    ///
    /// Propagates the underlying IO error if encountered.
    ///
    /// Otherwise, returns an error with [`io::ErrorKind::Other`] kind and a
    /// value depending on the case:
    ///
    /// -   [`FileSequenceError::OutOfRange`] if the state of the files in the
    ///     directory do not match the internal state - the index of a file
    ///     does not fall within the internal range.
    /// -   [`FileSequenceError::Duplicate`] if more than one file with the same
    ///     index is encountered, which normally should not happen.
    /// -   [`FileSequenceError::Broken`] if the sequence is not monotonically
    ///     increasing.
    ///
    /// In concurrent setting, `OutOfRange`, `Broken` and an error of
    /// [` io::ErrorKind::NotFound`] are likely to be retryable, unless the
    /// filesystem state has been modified externally.
    ///
    /// # Panics
    ///
    /// It will panic if internal state invariant is violated, which should be
    /// impossible, unless the memory is modified externally somehow.
    fn retrieve(&self, mut f: impl FnMut(File)) -> io::Result<()> {
        let start = WrappingSeq(self.start.load(atomic::Ordering::Relaxed));
        let end = WrappingSeq(self.end.load(atomic::Ordering::Relaxed));
        if start == end {
            return Ok(());
        }

        let paths = self
            .root
            .read_dir()?
            .map(|entry| entry.map(|entry| entry.path()).ok())
            .filter(|path| {
                path.as_ref().is_some_and(|path| {
                    path.is_file()
                        && path
                            .file_stem()
                            .and_then(std::ffi::OsStr::to_str)
                            .is_some_and(|name| name == self.name)
                })
            });

        let size = usize::from(
            start
                .distance(end)
                .expect("distance between start and end is always correct")
                .unsigned_abs(),
        );
        let mut files: Vec<Option<(WrappingSeq, u32, PathBuf)>> = vec![None; size];
        for path in paths {
            let (index, shift) = FileSequence::parse_opts(
                path.as_ref()
                    .and_then(|path| path.extension())
                    .and_then(std::ffi::OsStr::to_str)
                    .unwrap_or_default(),
            )?;
            if index < start || index >= end {
                return Err(FileSequenceError::OutOfRange(index.into()).into());
            }
            let pos = usize::from(
                index
                    .distance(start)
                    .ok_or_else(|| FileSequenceError::OutOfRange(index.into()))?
                    .unsigned_abs(),
            );
            if files[pos].is_some() {
                return Err(FileSequenceError::Duplicate(index.into()).into());
            }
            files[pos] = Some((index, shift, path.unwrap()));
        }
        if files.iter().any(Option::is_none) {
            return Err(FileSequenceError::Broken.into());
        }

        for file in files.into_iter().map(Option::unwrap) {
            f(File::open(file.2, file.1)?.with_alloc_info(self.stamp, file.0));
        }
        Ok(())
    }
}

unsafe impl Send for FileSequence {}
unsafe impl Sync for FileSequence {}

/// An open directory.
///
/// The primary use case for this type is to synchronize the directory after
/// modifying it, e.g. when creating or removing files. Implemented on Linux
/// only, the generic implementation is a no-op.
///
/// # Linux specifics
///
/// The path of the directory can be no larger than 255 characters to avoid
/// heap allocations when building C strings.
///
/// The directory is closed on drop and it will panic if the attempt to close
/// returns an error, other than of [`io::ErrorKind::Interrupted`] kind.
#[derive(Debug)]
struct Dir {
    path: PathBuf,
    #[cfg(all(feature = "libc", target_os = "linux"))]
    ptr: NonNull<libc::DIR>,
}

#[cfg(all(feature = "libc", target_os = "linux"))]
impl Dir {
    fn open<P: AsRef<Path>>(path: P) -> io::Result<Dir> {
        use std::os::unix::ffi::OsStrExt;

        let bytes = path.as_ref().as_os_str().as_bytes();
        if bytes.len() > 255 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "path too long"));
        }
        let mut buf = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let buf_ptr = buf.as_mut_ptr().cast::<u8>();
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr, bytes.len());
            buf_ptr.add(bytes.len()).write(0);
        }
        let cstr = core::ffi::CStr::from_bytes_with_nul(unsafe {
            core::slice::from_raw_parts(buf_ptr, bytes.len() + 1)
        })
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "path contains unexpected NUL byte",
            )
        })?;

        let ptr = NonNull::new(unsafe { libc::opendir(cstr.as_ptr()) })
            .ok_or_else(io::Error::last_os_error)?;
        Ok(Dir {
            path: path.as_ref().to_path_buf(),
            ptr,
        })
    }

    #[inline(always)]
    fn sync(&self) -> io::Result<()> {
        let fd = match unsafe { libc::dirfd(self.ptr.as_ptr()) } {
            fd if fd >= 0 => fd,
            -1 => return Err(io::Error::last_os_error()),
            ret => unreachable!("dirfd: unexpected return {ret}"),
        };
        match unsafe { libc::fsync(fd) } {
            0 => Ok(()),
            -1 => Err(io::Error::last_os_error()),
            ret => unreachable!("fsync: unexpected return {ret}"),
        }
    }
}

#[cfg(not(all(feature = "libc", target_os = "linux")))]
impl Dir {
    // Parity with the libc implementation, hence allowed unused.
    #[allow(clippy::unnecessary_wraps)]
    #[inline(always)]
    fn open<P: AsRef<Path>>(path: P) -> io::Result<Dir> {
        Ok(Dir {
            path: path.as_ref().to_path_buf(),
        })
    }

    // Parity with the libc implementation, hence allowed unused.
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    #[inline(always)]
    fn sync(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Dir {
    #[inline(always)]
    fn read_dir(&self) -> io::Result<fs::ReadDir> {
        self.path.read_dir()
    }
}

#[cfg(all(feature = "libc", target_os = "linux"))]
impl Drop for Dir {
    #[inline(always)]
    fn drop(&mut self) {
        let ret = unsafe { libc::closedir(self.ptr.as_ptr()) };
        assert!(
            ret == 0
                || (ret == -1 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted),
            "closedir: unexpected return {ret}: error: {:?}",
            io::Error::last_os_error()
        );
    }
}

/// A minimal file that is supposed to be atomically created and removed.
/// The lock file is removed on drop, if it has not been removed via
/// [`LockFile::release`].
#[derive(Debug)]
struct LockFile(Option<PathBuf>);

impl LockFile {
    /// Acquires a lock file by creating a symlink that holds this process id
    /// in the data section, making it a metadata-only atomic operation.
    #[inline(always)]
    fn acquire(path: PathBuf) -> io::Result<LockFile> {
        std::os::unix::fs::symlink(std::process::id().to_string(), &path)?;
        Ok(LockFile(Some(path)))
    }

    /// Release the lock file by removing the symlink file.
    ///
    /// # Panics
    ///
    /// If this lock file has been already released.
    #[inline(always)]
    fn release(&mut self) -> io::Result<()> {
        fs::remove_file(self.0.take().expect("double release"))
    }
}

impl Drop for LockFile {
    #[inline(always)]
    fn drop(&mut self) {
        if self.0.is_some() {
            let _ = self.release();
        }
    }
}

/// Increases the number of the kernel read-ahead pages by 2.
#[cfg(all(feature = "libc", target_os = "linux"))]
#[inline(always)]
fn double_readahead_pages<F: AsRawFd>(f: &F, size: u64) -> io::Result<()> {
    let advise = libc::POSIX_FADV_SEQUENTIAL;
    // The total size of a file is verified when created or opened.
    #[allow(clippy::cast_possible_wrap)]
    let ret = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, size as libc::off64_t, advise) };
    if ret == 0 {
        return Ok(());
    }
    Err(io::Error::from_raw_os_error(ret))
}

/// Reserves a space of a certain size for a file.
///
/// The allocated space is zero-initialized. This function should be called
/// once on a newly created file. That's it.
#[cfg(all(feature = "libc", target_os = "linux"))]
#[inline(always)]
fn reserve<F: AsRawFd>(f: &F, size: u64) -> io::Result<()> {
    let ret = unsafe {
        // The total size of a file is verified when created or opened.
        #[allow(clippy::cast_possible_wrap)]
        libc::fallocate(
            f.as_raw_fd(),
            /*mode=*/ 0,
            /*offset=*/ 0,
            size as libc::off64_t,
        )
    };
    match ret {
        0 => Ok(()),
        -1 => Err(io::Error::last_os_error()),
        _ => unreachable!("fallocate: unexpected return {ret}"),
    }
}

/// A `u16` that wraps around. Similar to [`core::num::Wrapping`], but it
/// implements only limited number of operations and comparison function treats
/// wrapped numbers such that `u16::MAX` as less than 0. The comparison,
/// therefore, works only on numbers which are no more than 14 bits apart.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)] // manual: PartialOrd
#[repr(transparent)]
struct WrappingSeq(u16);

impl WrappingSeq {
    /// Increment the number by 1, wrapping around if reached the maximum.
    #[inline(always)]
    #[must_use]
    fn inc(self) -> WrappingSeq {
        WrappingSeq(self.0.wrapping_add(1))
    }

    /// Computes the distance between two numbers, returning `None` if they
    /// are incomparable because the distance is greater than the half of
    /// the `i16` limit. The sign of the returned value indicates whether
    /// `self` comes before (negative) or after (positive) `other`.
    #[inline(always)]
    #[must_use]
    fn distance(self, other: WrappingSeq) -> Option<i16> {
        // The wrap is intentional and is checked right after.
        #[allow(clippy::cast_possible_wrap)]
        let distance = self.0.wrapping_sub(other.0) as i16;
        if distance <= i16::MIN >> 1 || distance > i16::MAX >> 1 {
            None
        } else {
            Some(distance)
        }
    }
}

impl From<WrappingSeq> for u16 {
    #[inline(always)]
    fn from(value: WrappingSeq) -> u16 {
        value.0
    }
}

impl cmp::PartialOrd for WrappingSeq {
    #[inline(always)]
    fn partial_cmp(&self, other: &WrappingSeq) -> Option<cmp::Ordering> {
        self.distance(*other)
            .and_then(|value| value.partial_cmp(&0))
    }
}

impl core::fmt::LowerHex for WrappingSeq {
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// A workaround to advance slice of `IoSlice` or `IoSliceMut` types in a way
/// that does not depend on implementation details. The values are replaced
/// with a result of `F`, which receives a slice of the underlying buffer of
/// the previous `IoSlice`.
///
/// # Panics
///
/// Panics if `n` is greater than the sum of lengths of `iovec`.
#[allow(clippy::mut_mut)]
#[inline(always)]
fn advance_iovec<'a, T: core::ops::Deref<Target = [u8]>, F>(iovec: &mut &mut [T], n: usize, new: F)
where
    F: Fn(&'a [u8]) -> T,
{
    let mut next = 0;
    let mut remaining = n;
    for bytes in iovec.iter() {
        if bytes.len() > remaining {
            break;
        }
        next += 1;
        remaining -= bytes.len();
    }

    *iovec = &mut core::mem::take(iovec)[next..];
    if iovec.is_empty() {
        assert_eq!(remaining, 0, "advancing iovec beyond length");
        return;
    }
    // SAFETY: Slice re-created is the same slice, yet with a different lifetime
    // to trick the borrow checker. Based on the context where it is used,
    // the memory where inner slices are pointing to outlive the iovec.
    let bytes = new(unsafe {
        let slice = &iovec[0][remaining..];
        core::slice::from_raw_parts(slice.as_ptr(), slice.len())
    });
    let _ = core::mem::replace(&mut iovec[0], bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_verify_input() {
        let case = "returns total size";
        assert_eq!(File::verify_input(4, 12).expect(case), 16384, "{case}");

        let case = "block shift outside range";
        let err = File::verify_input(1, 11).expect_err(case);
        let msg = "block shift must be between 12 and 28 inclusive";
        assert_eq!(err.to_string(), msg, "{case}");
        let err = File::verify_input(1, 29).expect_err(case);
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "zero block count";
        let err = File::verify_input(0, 12).expect_err(case);
        let msg = "block count must be non-zero";
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "total size too large";
        let err = File::verify_input(1 << (63 - 28), 28).expect_err(case);
        let msg = "total blocks size too large";
        assert_eq!(err.to_string(), msg, "{case}");
    }

    #[test]
    fn file_verify_bufs() {
        // Regular behavior is checked implicitly, here only cold code is tested.
        let bytes = &[1, 2, 3, 4, 5];

        let case = "odd bufs";
        #[rustfmt::skip]
        let bufs = [
            &bytes[..2], &bytes[..4],
            &bytes[..2],
        ];
        let err = File::verify_bufs(&bufs, 6).expect_err(case);
        let msg = "odd number of bufs";
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "large uneven pair";
        #[rustfmt::skip]
        let bufs = [
            &bytes[..2], &bytes[..4],
            &bytes[..2], &bytes[..5],
            &bytes[..2], &bytes[..4],
        ];
        let err = File::verify_bufs(&bufs, 6).expect_err(case);
        let msg = "uneven pair of bufs is too large";
        assert_eq!(err.to_string(), msg, "{case}");

        let case = "uneven pair before last";
        #[rustfmt::skip]
        let bufs = [
            &bytes[..2], &bytes[..4],
            &bytes[..2], &bytes[..2],
            &bytes[..2], &bytes[..2],
        ];
        let err = File::verify_bufs(&bufs, 6).expect_err(case);
        let msg = "uneven pair of bufs is not the last pair";
        assert_eq!(err.to_string(), msg, "{case}");
        #[rustfmt::skip]
        let bufs = [
            &bytes[..2], &bytes[..4],
            &bytes[..2], &bytes[..2],
            &bytes[..2], &bytes[..4],
        ];
        let err = File::verify_bufs(&bufs, 6).expect_err(case);
        assert_eq!(err.to_string(), msg, "{case}");
        #[rustfmt::skip]
        let bufs = [
            &bytes[..2], &bytes[..4],
            &[], &[],
            &bytes[..2], &bytes[..4]
        ];
        let err = File::verify_bufs(&bufs, 6).expect_err(case);
        assert_eq!(err.to_string(), msg, "{case}");
    }

    #[cfg(all(feature = "libc", target_os = "linux"))]
    #[test]
    fn file_chunk_shift() {
        assert_eq!(File::chunk_shift(12), 21);
        assert_eq!(File::chunk_shift(13), 22);
        for shift in 14..=23 {
            assert_eq!(File::chunk_shift(shift), 23);
        }
        for shift in 24..=28 {
            assert_eq!(File::chunk_shift(shift), shift);
        }
    }

    #[test]
    fn filesequence_parse_opts() {
        let case = "good opts";
        let parsed = FileSequence::parse_opts("0ccc-7").expect(case);
        assert_eq!(parsed, (WrappingSeq(0x0ccc), 7), "{case}");
        let parsed = FileSequence::parse_opts("cc-09").expect(case);
        assert_eq!(parsed, (WrappingSeq(0x00cc), 9), "{case}");
        let parsed = FileSequence::parse_opts("0000000f-0").expect(case);
        assert_eq!(parsed, (WrappingSeq(0x000f), 0), "{case}");

        let case = "bad opts";
        let table = [
            ("", "filesequence: bad index: ''"),
            ("-9", "filesequence: bad index: '-9'"),
            ("x0-9", "filesequence: bad index: 'x0-9'"),
            ("f0", "filesequence: fields missing: 'f0'"),
            ("f0-a", "filesequence: bad shift: 'f0-a'"),
            ("f0-9-", "filesequence: too many fields: 'f0-9-'"),
        ];
        for row in table {
            let err = FileSequence::parse_opts(row.0).expect_err(case);
            assert_eq!(err.to_string(), row.1, "{case}");
        }
    }

    #[test]
    fn wrappingseq_parital_cmp() {
        let zero = WrappingSeq(0);
        let one = WrappingSeq(1);
        let quarter = WrappingSeq(u16::MAX >> 2);
        let half = WrappingSeq(u16::MAX >> 1);
        let max = WrappingSeq(u16::MAX);

        assert!(max < one);
        assert!(!(max > one));
        assert!(max != one);
        assert!(!(one < max));
        assert!(one > max);
        assert!(one != max);

        assert!(zero < quarter);
        assert!(!(zero > quarter));
        assert!(zero != quarter);
        assert!(!(quarter < zero));
        assert!(quarter > zero);
        assert!(quarter != zero);

        assert!(!(max < quarter));
        assert!(!(max > quarter));
        assert!(max != quarter);
        assert_eq!(max.partial_cmp(&quarter), None);
        assert!(!(quarter < max));
        assert!(!(quarter > max));
        assert!(quarter != max);
        assert_eq!(quarter.partial_cmp(&max), None);

        assert!(!(max < half));
        assert!(!(max > half));
        assert!(max != half);
        assert_eq!(max.partial_cmp(&half), None);
        assert!(!(half < max));
        assert!(!(half > max));
        assert!(half != max);
        assert_eq!(half.partial_cmp(&max), None);
    }

    #[test]
    fn advance_iovec_fn() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut iovec = [
            io::IoSlice::new(&bytes[0..2]),
            io::IoSlice::new(&bytes[2..4]),
            io::IoSlice::new(&bytes[4..8]),
        ];
        let mut iovec = iovec.as_mut_slice();
        let concat = |iovec: &[io::IoSlice]| {
            iovec
                .iter()
                .flat_map(|bytes| bytes.iter().copied())
                .collect::<Vec<_>>()
        };

        advance_iovec(&mut iovec, 0, io::IoSlice::new);
        assert_eq!(iovec.len(), 3);
        assert_eq!(concat(iovec), &bytes[0..8]);
        advance_iovec(&mut iovec, 1, io::IoSlice::new);
        assert_eq!(iovec.len(), 3);
        assert_eq!(concat(iovec), &bytes[1..8]);
        advance_iovec(&mut iovec, 1, io::IoSlice::new);
        assert_eq!(iovec.len(), 2);
        assert_eq!(concat(iovec), &bytes[2..8]);
        advance_iovec(&mut iovec, 3, io::IoSlice::new);
        assert_eq!(iovec.len(), 1);
        assert_eq!(concat(iovec), &bytes[5..8]);
        advance_iovec(&mut iovec, 3, io::IoSlice::new);
        assert_eq!(iovec.len(), 0);
        assert_eq!(concat(iovec), &[]);

        let mut iovec = [io::IoSlice::new(&[0x01, 0x02]), io::IoSlice::new(&[0x03])];
        let mut iovec = iovec.as_mut_slice();
        advance_iovec(&mut iovec, 3, io::IoSlice::new);
        assert_eq!(iovec.len(), 0);
        assert_eq!(concat(&iovec), &[]);

        let mut iovec = [
            io::IoSlice::new(&[0x01, 0x02]),
            io::IoSlice::new(&[]),
            io::IoSlice::new(&[0x03]),
            io::IoSlice::new(&[]),
        ];
        let mut iovec = iovec.as_mut_slice();
        advance_iovec(&mut iovec, 3, io::IoSlice::new);
        assert_eq!(iovec.len(), 0);
        assert_eq!(concat(&iovec), &[]);
    }

    #[test]
    #[should_panic(expected = "advancing iovec beyond length")]
    fn advance_iovec_fn_beyond_length() {
        let mut iovec = [io::IoSlice::new(&[0x01, 0x02]), io::IoSlice::new(&[0x03])];
        advance_iovec(&mut &mut iovec.as_mut_slice(), 4, io::IoSlice::new);
    }
}
