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
//!
//! # Features
//!
//! This library aims to be minimal, hence extra functionality which is not
//! part of the core implementation sits behind [Cargo features][features] for
//! conditional compilation. The following features are available:
//! -   `io-filesystem` - includes storage implementation for block streams
//!     backed by a generic filesystem.
//! -   `libc` - if enabled, file-backed IO will use more efficient and reliable
//!     calls to the Linux kernel. Makes sense only for Linux.
//!
//! [features]: https://doc.rust-lang.org/cargo/reference/features.html
//!
//! # Examples
//!
//! ```
//! use std::io;
//!
//! use pstream::{EndlessStream, io::Void};
//!
//! fn main() -> io::Result<()> {
//!     let void = Void::new(10, 17);
//!     let stream = EndlessStream::new(void);
//!     stream.grow()?;
//!     let data = [10u8; 8].as_slice();
//!     stream.append(data)?;
//!     for chunk in stream.iter() {
//!         assert_eq!(chunk.bytes().unwrap().as_ref(), data);
//!     }
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod block;
pub mod endless;
pub mod io;

#[doc(inline)]
pub use block::Blocks;
#[doc(inline)]
pub use endless::BlocksAllocator;

pub use block::Stream as BlockStream;
pub use endless::Stream as EndlessStream;
