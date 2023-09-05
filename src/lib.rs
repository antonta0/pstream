//! A persistent byte stream over block storage.
//!
//! The original idea was to have a primitive for writing reliable persistent
//! software, and a byte stream seems like a good starting point - it could be
//! used as a primary store for simpler software, and it has a place in
//! more complex systems.
//!
//! There are numerous applications for a byte stream: a write-ahead log of a
//! database, message broker, just any kind of a log, a persistent buffer, etc.
//! I think it is a versatile, efficient and easy to reason about structure.
//! Nearly any kind of data store can be modelled with that as long as data
//! fits in memory, which is the main limitation of this library.
//!
//! The aim is to keep the library *simple*, such that code can be reasoned
//! by inspection and be maintainable, yet have the API flexible enough to
//! cover a variety of use cases. Simplicity also generally leads to better
//! performance and reliability. It is designed to be used in concurrent code.
//!
//! Since secondary memory is block-based, the core abstraction is a contiguous
//! byte stream backed by blocks, which is then used further to build a
//! conceptually endless stream of data. Refer to relevant module documentation
//! for details.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)]

pub mod block;
pub mod endless;
pub mod io;

#[doc(inline)]
pub use block::Blocks;
#[doc(inline)]
pub use endless::BlocksAllocator;

pub use block::Stream as BlockStream;
pub use endless::Stream as EndlessStream;
