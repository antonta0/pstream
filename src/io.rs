//! Built-in IO implementations for the streams.

#[cfg(all(not(feature = "io-filesystem"), feature = "libc"))]
compile_error!("there is no use for libc without \"io-filesystem\" feature");

#[cfg(feature = "io-filesystem")]
mod fs;
mod void;

#[cfg(feature = "io-filesystem")]
#[cfg_attr(docsrs, doc(cfg(feature = "io-filesystem")))]
pub use fs::*;
pub use void::Void;
