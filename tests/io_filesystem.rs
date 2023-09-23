use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Barrier, Once},
};

use pstream::{
    io::{File, FileSequence, FileSequenceError},
    Blocks, BlocksAllocator,
};

static SETUP: Once = Once::new();

fn setup() -> PathBuf {
    let dir = std::env::temp_dir().join("pstream:tests:io_filesystem");
    SETUP.call_once(|| {
        if dir.is_dir() {
            fs::remove_dir_all(&dir).expect("init");
        }
        fs::create_dir(&dir).expect("init");
    });
    dir
}

fn with_dummy_file<P: AsRef<Path>, F: FnOnce()>(path: P, f: F) -> io::Result<()> {
    fs::File::options()
        .write(true)
        .create_new(true)
        .open(&path)?;
    f();
    fs::remove_file(&path)
}

#[test]
fn file_create() {
    let root = setup();
    let path = root.join("file_create");

    let case = "with invalid input fails";
    let err = File::create(&path, 10, 11).expect_err(case);
    let msg = "block shift must be between 12 and 28 inclusive";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "succeeds";
    File::create(&path, 10, 17).expect(case);
}

#[test]
fn file_open() -> io::Result<()> {
    let root = setup();
    let path = root.join("file_open");

    let case = "after create succeeds";
    File::create(&path, 10, 17)?;
    File::open(&path, 17).expect(case);
    File::open(&path, 18).expect(case);

    let case = "with block shape mismatch fails";
    let err = File::open(&path, 19).expect_err(case);
    let msg = "file size not aligned to block size";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "with invalid input fails";
    let err = File::open(&path, 11).expect_err(case);
    let msg = "block shift must be between 12 and 28 inclusive";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "with uneven file length fails";
    let file = fs::File::options().read(true).write(true).open(&path)?;
    file.set_len(123)?;
    let err = File::open(&path, 17).expect_err(case);
    let msg = "file size not aligned to block size";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "with zero file length fails";
    file.set_len(0)?;
    let err = File::open(&path, 17).expect_err(case);
    assert_eq!(err.to_string(), "zero file size", "{case}");

    Ok(())
}

#[test]
fn file_load_store() -> io::Result<()> {
    let to_iovec = |bytes: &[u8], size| {
        bytes[..size]
            .chunks(1 << 17)
            .flat_map(|chunk| {
                [
                    io::IoSlice::new(unsafe { core::slice::from_raw_parts(chunk.as_ptr(), 48) }),
                    io::IoSlice::new(unsafe {
                        core::slice::from_raw_parts(chunk.as_ptr().add(48), chunk.len() - 48)
                    }),
                ]
            })
            .collect::<Vec<_>>()
    };
    let to_iovec_mut = |bytes: &mut [u8], size| {
        bytes[..size]
            .chunks_mut(1 << 17)
            .flat_map(|chunk| {
                [
                    io::IoSliceMut::new(unsafe {
                        core::slice::from_raw_parts_mut(chunk.as_ptr() as *mut u8, 48)
                    }),
                    io::IoSliceMut::new(unsafe {
                        core::slice::from_raw_parts_mut(
                            chunk.as_ptr().add(48).cast_mut(),
                            chunk.len() - 48,
                        )
                    }),
                ]
            })
            .collect::<Vec<_>>()
    };

    let root = setup();

    let words = {
        // That's 16 MiB + 64 KiB worth of bytes.
        let mut words = vec![0u64; 2105344];
        let mut range = 1u64..;
        words.fill_with(|| range.next().unwrap());
        words
    };
    let bytes =
        unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() << 3) };
    let mut buffer = vec![0u8; 256 << 17];

    let path = root.join("file_load_store");
    let mut file = File::create(&path, 256, 17)?;

    macro_rules! assert_write {
        ($block:expr, $size:expr, $case:expr) => {
            let mut bufs = to_iovec(bytes, $size);
            file.store_at($block, &mut bufs).expect($case);
            let read_until = if ($block << 17) + $size + 96 > 256 << 17 {
                $size
            } else {
                // These 96 bytes could panic in certain cases, but who cares?
                $size + 96
            };
            let mut bufs = to_iovec_mut(&mut buffer, read_until);
            file.load_from($block, &mut bufs).expect($case);
            assert_eq!(buffer[..$size], bytes[..$size], "{}", $case);
            assert!(
                buffer[$size..read_until].iter().all(|&v| v == 0),
                "{}",
                $case
            );
            buffer[..read_until].fill(0);
        };
    }

    assert_write!(0, 96, "write small");
    assert_write!(0, 128, "write small overwrite");
    assert_write!(0, 256, "write small overwrite more");
    assert_write!(1, 1 << 17, "write one whole block");
    assert_write!(2, 2 << 17, "write two blocks aligned");
    assert_write!(2, (2 << 17) + 256, "write two blocks with partial");
    assert_write!(5, 1 << 23, "write large");
    assert_write!(5, (1 << 23) + 256, "write large with partial");
    assert_write!(5, (1 << 23) + (1 << 17), "write large with full block");
    assert_write!(5, 2 << 23, "write very large");
    assert_write!(5, (2 << 23) + 1024, "write very large with partial");
    assert_write!(128, 128 << 17, "write half end aligned");

    let case = "store past the end of the file";
    let mut bufs = to_iovec(bytes, 50);
    let err = file.store_at(256, &mut bufs).expect_err(case);
    assert_eq!(err.to_string(), "write exceeds file capacity", "{case}");
    let mut bufs = to_iovec(bytes, (1 << 17) + 50);
    let err = file.store_at(255, &mut bufs).expect_err(case);
    assert_eq!(err.to_string(), "write exceeds file capacity", "{case}");

    let case = "load past the end of the file";
    let mut bufs = to_iovec_mut(&mut buffer, 50);
    let err = file.load_from(256, &mut bufs).expect_err(case);
    assert_eq!(err.to_string(), "failed to fill all buffers", "{case}");
    let mut bufs = to_iovec_mut(&mut buffer, (1 << 17) + 50);
    let err = file.load_from(255, &mut bufs).expect_err(case);
    assert_eq!(err.to_string(), "failed to fill all buffers", "{case}");

    let case = "store with bad bufs";
    let bufs = &mut [io::IoSlice::new(&[])];
    let err = file
        .store_at(255, &mut bufs.as_mut_slice())
        .expect_err(case);
    assert_eq!(err.to_string(), "odd number of bufs", "{case}");

    let case = "store zero";
    let bufs = &mut [io::IoSlice::new(&[]), io::IoSlice::new(&[])];
    let err = file
        .store_at(255, &mut bufs.as_mut_slice())
        .expect_err(case);
    assert_eq!(err.to_string(), "failed to write all buffers", "{case}");
    let bufs = &mut [];
    file.store_at(255, &mut bufs.as_mut_slice()).expect(case);

    let case = "load zero";
    let bufs = &mut [io::IoSliceMut::new(&mut []), io::IoSliceMut::new(&mut [])];
    let err = file
        .load_from(255, &mut bufs.as_mut_slice())
        .expect_err(case);
    assert_eq!(err.to_string(), "failed to fill all buffers", "{case}");
    let bufs = &mut [];
    file.load_from(255, &mut bufs.as_mut_slice()).expect(case);
    Ok(())
}

#[test]
fn filesequence_open() -> io::Result<()> {
    // Regular usage is tested implicitly in other functions.
    let root = setup();

    let case = "with invalid input fails";
    let err = FileSequence::open(&root, "filesequence_open", 2, 11).expect_err(case);
    let msg = "block shift must be between 12 and 28 inclusive";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "with unparsable file fails";
    with_dummy_file(root.join("filesequence_open.xxx"), || {
        let err = FileSequence::open(&root, "filesequence_open", 2, 12).expect_err(case);
        let msg = "filesequence: bad index: 'xxx'";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;

    let case = "duplicate fails";
    let fileseq = FileSequence::open(&root, "filesequence_open", 2, 12)?;
    let err = FileSequence::open(&root, "filesequence_open", 2, 12).expect_err(case);
    let msg = "filesequence: lock file is held by another instance";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "comparison out of range fails";
    fileseq.alloc()?;
    fileseq.close()?;
    with_dummy_file(root.join("filesequence_open.4002-12"), || {
        let err_msg = FileSequence::open(&root, "filesequence_open", 2, 12)
            .expect_err(case)
            .to_string();
        let msg1 = "filesequence: index 0x0001 out of range";
        let msg2 = "filesequence: index 0x4002 out of range";
        if err_msg != msg1 && err_msg != msg2 {
            panic!("{case}: no match for message: {err_msg}");
        }
    })?;

    let case = "with non-contiguous range succeeds";
    with_dummy_file(root.join("filesequence_open.1fff-12"), || {
        let fileseq = FileSequence::open(&root, "filesequence_open", 2, 12).expect(case);
        // Allocate a file with MAX / 2 index for the next test to use.
        fileseq.alloc().unwrap();
    })?;

    let case = "distance too long fails";
    with_dummy_file(root.join("filesequence_open.4000-12"), || {
        let err = FileSequence::open(&root, "filesequence_open", 2, 12).expect_err(case);
        let msg = "filesequence: 0x0001..0x4001 is too long";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;

    Ok(())
}

#[test]
fn filesequence_alloc() -> io::Result<()> {
    let root = setup();

    let fileseq = FileSequence::open(&root, "filesequence_alloc", 2, 12)?;
    let case = "allocate twice";
    let first = fileseq.alloc().expect(case);
    let second = fileseq.alloc().expect(case);

    let case = "allocate existing fails";
    with_dummy_file(root.join("filesequence_alloc.0003-12"), || {
        let err = fileseq.alloc().expect_err(case);
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists, "{case}");
    })?;

    let case = "allocate many";
    while let Ok(_) = fileseq.alloc() {}
    let err = fileseq.alloc().expect_err(case);
    let msg = "filesequence: 0x0001..0x4000 is too long";
    assert_eq!(err.to_string(), msg, "{case}");
    fileseq.release(first).map_err(|(_, err)| err)?;
    fileseq.alloc().expect(case);
    let err = fileseq.alloc().expect_err(case);
    let msg = "filesequence: 0x0002..0x4001 is too long";
    assert_eq!(err.to_string(), msg, "{case}");
    fileseq.release(second).map_err(|(_, err)| err)?;
    fileseq.alloc().expect(case);
    let err = fileseq.alloc().expect_err(case);
    let msg = "filesequence: 0x0003..0x4002 is too long";
    assert_eq!(err.to_string(), msg, "{case}");

    // Clean up, as there was plenty of files created
    fileseq.retrieve(|file| fileseq.release(file).unwrap())?;

    Ok(())
}

#[test]
fn filesequence_release() -> io::Result<()> {
    let root = setup();
    let fileseq = FileSequence::open(&root, "filesequence_release", 2, 12)?;

    let first = fileseq.alloc()?;
    let second = fileseq.alloc()?;
    let third = fileseq.alloc()?;

    let case = "non-first fails";
    let (second, err) = fileseq.release(second).expect_err(case);
    let msg = "filesequence: index 0x0002 out of order";
    assert_eq!(err.to_string(), msg, "{case}");
    let (third, err) = fileseq.release(third).expect_err(case);
    let msg = "filesequence: index 0x0003 out of order";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "first succeeds";
    fileseq.release(first).expect(case);
    let mut count = 0;
    fileseq.retrieve(|_| count += 1)?;
    assert_eq!(count, 2, "{case}");

    let case = "double release fails";
    let mut second_second: Option<File> = None;
    fileseq.retrieve(|file| {
        second_second.get_or_insert(file);
    })?;
    fileseq.release(second).expect(case);
    let (second, err) = fileseq.release(second_second.unwrap()).expect_err(case);
    let msg = "filesequence: index 0x0002 out of range";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "last succeeds";
    fileseq.release(third).expect(case);
    fileseq.retrieve(|_| unreachable!("{case}"))?;

    let case = "release from another sequence fails";
    let fileseq_extra = FileSequence::open(&root, "filesequence_release_extra", 2, 12)?;
    let (_, err) = fileseq_extra.release(second).expect_err(case);
    let msg = "filesequence: file does not belong to this sequence";
    assert_eq!(err.to_string(), msg, "{case}");

    Ok(())
}

#[test]
fn filesequence_retrieve() -> io::Result<()> {
    let root = setup();

    let case = "empty";
    let fileseq = FileSequence::open(&root, "filesequence_retrieve", 2, 12)?;
    fileseq.retrieve(|_| unreachable!("{case}")).expect(case);

    let case = "regular usage";
    fileseq.release(fileseq.alloc()?).map_err(|(_, err)| err)?;
    fileseq.alloc()?;
    fileseq.close()?;
    let fileseq = FileSequence::open(&root, "filesequence_retrieve", 2, 13)?;
    fileseq.alloc()?;
    let mut count = 0;
    fileseq.retrieve(|_| count += 1).expect(case);
    assert_eq!(count, 2, "{case}");

    let case = "with duplicates fails";
    with_dummy_file(root.join("filesequence_retrieve.0003-20"), || {
        let err = fileseq
            .retrieve(|_| unreachable!("{case}"))
            .expect_err(case);
        let msg = "filesequence: duplicate index 0x0003";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;

    let case = "out of range fails";
    with_dummy_file(root.join("filesequence_retrieve.000f-20"), || {
        let err = fileseq
            .retrieve(|_| unreachable!("{case}"))
            .expect_err(case);
        let msg = "filesequence: index 0x000f out of range";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;
    with_dummy_file(root.join("filesequence_retrieve.0ff0-20"), || {
        let err = fileseq
            .retrieve(|_| unreachable!("{case}"))
            .expect_err(case);
        let msg = "filesequence: index 0x0ff0 out of range";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;

    let case = "broken sequence fails";
    fileseq.alloc()?;
    fs::remove_file(root.join("filesequence_retrieve.0003-13"))?;
    let err = fileseq
        .retrieve(|_| unreachable!("{case}"))
        .expect_err(case);
    let msg = "filesequence: indexes are not monotonically increasing";
    assert_eq!(err.to_string(), msg, "{case}");

    let case = "bad opts";
    fileseq.close()?;
    fs::remove_file(root.join("filesequence_retrieve.0002-12"))?;
    let fileseq = FileSequence::open(&root, "filesequence_retrieve", 2, 13)?;
    with_dummy_file(root.join("filesequence_retrieve.000x-20"), || {
        let err = fileseq
            .retrieve(|_| unreachable!("{case}"))
            .expect_err(case);
        let msg = "filesequence: bad index: '000x-20'";
        assert_eq!(err.to_string(), msg, "{case}");
    })?;
    fs::rename(
        root.join("filesequence_retrieve.0004-13"),
        root.join("filesequence_retrieve.0004-30"),
    )?;
    let err = fileseq
        .retrieve(|_| unreachable!("{case}"))
        .expect_err(case);
    let msg = "block shift must be between 12 and 28 inclusive";
    assert_eq!(err.to_string(), msg, "{case}");

    Ok(())
}

#[test]
fn filesequence_wrapping() -> io::Result<()> {
    let root = setup();

    let case = "wrap while allocating";
    File::create(root.join("filesequence_wrapping.fffd-12"), 2, 12)?;
    let fileseq = FileSequence::open(&root, "filesequence_wrapping", 2, 12)?;
    for _ in 1..=4 {
        fileseq.alloc().expect(case);
    }
    let mut count = 0;
    fileseq.retrieve(|_| count += 1).expect(case);
    assert_eq!(count, 5, "{case}");
    fileseq.close()?;

    let case = "open wrapping";
    let fileseq = FileSequence::open(&root, "filesequence_wrapping", 2, 12).expect(case);
    let mut count = 0;
    fileseq.retrieve(|_| count += 1).expect(case);
    assert_eq!(count, 5, "{case}");
    fileseq.close()?;

    Ok(())
}

#[test]
fn filesequence_threads() -> io::Result<()> {
    let root = setup();

    // Create 10 files to have some buffer, then start 4 threads, each
    // alternating between allocating or releasing a file for a fixed number
    // of iterations. The end result should be a 10 files with the index
    // shifted by the number of iterations * 4.
    let fileseq = FileSequence::open(&root, "filesequence_threads", 2, 12)?;
    for _ in 1..=10 {
        fileseq.alloc()?;
    }
    let fileseq = Arc::new(fileseq);
    let barrier = Arc::new(Barrier::new(4));
    let iterations = 2000;
    let mut threads = vec![];
    for t in 1..=4 {
        let fileseq_clone = Arc::clone(&fileseq);
        let barrier_clone = Arc::clone(&barrier);
        threads.push(std::thread::spawn(move || {
            barrier_clone.wait();
            for i in 0..iterations {
                'iter: loop {
                    if i & 1 == t & 1 {
                        match fileseq_clone.alloc() {
                            Ok(_) => break 'iter,
                            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                                continue 'iter
                            }
                            Err(err) => panic!("{err}"),
                        }
                    } else {
                        let mut file: Option<File> = None;
                        match fileseq_clone.retrieve(|f| {
                            file.get_or_insert(f);
                        }) {
                            Ok(_) => {}
                            Err(err) if err.kind() == io::ErrorKind::NotFound => continue 'iter,
                            Err(err) if err.kind() == io::ErrorKind::Other => {
                                let inner = err
                                    .get_ref()
                                    .unwrap()
                                    .downcast_ref::<FileSequenceError>()
                                    .unwrap();
                                match inner {
                                    FileSequenceError::Broken
                                    | FileSequenceError::OutOfRange(_) => continue 'iter,
                                    _ => panic!("{err}"),
                                }
                            }
                            Err(err) => panic!("{err}"),
                        }
                        if file.is_none() {
                            continue 'iter;
                        }
                        match fileseq_clone.release(file.unwrap()) {
                            Ok(_) => break 'iter,
                            Err((_, err)) if err.kind() == io::ErrorKind::NotFound => {
                                continue 'iter
                            }
                            Err((_, err)) if err.kind() == io::ErrorKind::Other => {
                                let inner = err
                                    .get_ref()
                                    .unwrap()
                                    .downcast_ref::<FileSequenceError>()
                                    .unwrap();
                                match inner {
                                    FileSequenceError::OutOfRange(_) => continue 'iter,
                                    _ => panic!("{err}"),
                                }
                            }
                            Err((_, err)) => panic!("{err}"),
                        }
                    }
                }
            }
        }));
    }
    threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .count();

    // The order is verified implicitly via retrieve and release.
    let mut count = 0;
    fileseq
        .retrieve(|file| {
            count += 1;
            fileseq.release(file).unwrap();
        })
        .unwrap();
    assert_eq!(count, 10);

    fileseq.alloc()?;
    Arc::into_inner(fileseq).unwrap().close()?;
    for entry in fs::read_dir(&root)? {
        let path = entry?.path();
        if path
            .file_stem()
            .is_some_and(|prefix| prefix == "filesequence_threads")
        {
            let expected = std::ffi::OsStr::new("filesequence_threads.0fab-12");
            assert_eq!(path.file_name(), Some(expected));
        }
    }

    Ok(())
}
