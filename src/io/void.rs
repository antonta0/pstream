//! IO via nothing. Can be used to create memory-only streams.

use std::io;

use crate::{Blocks, BlocksAllocator};

/// A piece of void subdivided into blocks.
///
/// It implements all traits necessary to be used by the streams, and is as
/// minimal as it can be. The IO operations are dummy - the reads are not
/// modifying the buffer, the writes are into nothing. In other words it is
/// similar to `/dev/null`.
///
/// Each allocation is merely a copy of itself, and no state is kept by the
/// allocator whatsoever.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Void {
    block_count: u64,
    block_shift: u32,
}

impl Void {
    /// Creates a new fixed-sized piece of void.
    #[inline(always)]
    #[must_use]
    pub fn new(block_count: u64, block_shift: u32) -> Void {
        Self {
            block_count,
            block_shift,
        }
    }
}

impl Blocks for Void {
    #[inline(always)]
    fn block_count(&self) -> u64 {
        self.block_count
    }

    #[inline(always)]
    fn block_shift(&self) -> u32 {
        self.block_shift
    }
}

unsafe impl BlocksAllocator for Void {
    type Blocks = Void;

    #[inline(always)]
    fn alloc(&self) -> io::Result<Void> {
        Ok(*self)
    }

    #[inline(always)]
    fn release(&self, blocks: Void) -> Result<(), (Void, io::Error)> {
        assert_eq!(blocks, *self);
        Ok(())
    }

    #[inline(always)]
    fn retrieve(&self, _f: impl FnMut(Void)) -> io::Result<()> {
        Ok(())
    }
}

impl io::Read for Void {
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(buf.len())
    }
}

impl io::Seek for Void {
    #[inline(always)]
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        if let io::SeekFrom::Start(pos) = pos {
            return Ok(pos);
        }
        unimplemented!("block stream only seeks from the start")
    }
}

impl io::Write for Void {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
