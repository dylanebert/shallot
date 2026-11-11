use shallot_draco::decode_faces;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

#[derive(Clone, Copy, Default, Debug)]
struct Meter {
    active: bool,
    calls: usize,
    allocated: usize,
    budget: usize,
    fail_at: Option<usize>,
    denied: bool,
}
thread_local! { static METER: Cell<Meter> = const { Cell::new(Meter { active: false, calls: 0, allocated: 0, budget: 0, fail_at: None, denied: false }) }; }
struct Allocator;
fn admit(size: usize) -> bool {
    METER
        .try_with(|cell| {
            let mut meter = cell.get();
            if !meter.active {
                return true;
            }
            let call = meter.calls;
            meter.calls += 1;
            let allowed =
                meter.fail_at != Some(call) && size <= meter.budget.saturating_sub(meter.allocated);
            if allowed {
                meter.allocated += size;
            } else {
                meter.denied = true;
            }
            cell.set(meter);
            allowed
        })
        .unwrap_or(true)
}
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            unsafe { System.alloc(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            unsafe { System.alloc_zeroed(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if admit(size) {
            unsafe { System.realloc(ptr, layout, size) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(budget: usize, fail_at: Option<usize>, f: impl FnOnce() -> T) -> (T, Meter) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            METER.with(|cell| {
                let mut m = cell.get();
                m.active = false;
                cell.set(m);
            });
        }
    }
    METER.with(|cell| {
        cell.set(Meter {
            active: true,
            budget,
            fail_at,
            ..Meter::default()
        })
    });
    let reset = Reset;
    let result = f();
    let meter = METER.with(Cell::get);
    drop(reset);
    (result, meter)
}
fn varint(mut n: usize, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push((n as u8 & 127) | 128);
        n >>= 7;
    }
    out.push(n as u8);
}
fn stream(method: u8, metadata: bool, counts: &[usize], tail: &[u8]) -> Vec<u8> {
    let mut bytes = b"DRACO\x02\x02\x01".to_vec();
    bytes.extend_from_slice(&[method, 0, if metadata { 128 } else { 0 }]);
    for &count in counts {
        varint(count, &mut bytes);
    }
    bytes.extend_from_slice(tail);
    bytes
}

#[test]
fn reviewer_valence_bombs_are_refused_before_any_allocation() {
    let small = [
        0x44, 0x52, 0x41, 0x43, 0x4f, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00, 0x02, 0xa0, 0x8d, 0x06,
        0x80, 0xad, 0xe2, 0x04, 0x00, 0x80, 0xad, 0xe2, 0x04, 0x00, 0x00, 0x80, 0x01, 0x00, 0x80,
        0xad, 0xe2, 0x04, 0x01, 0x01, 0x01, 0x01, 0x40, 0x01, 0x00,
    ];
    let mut large = small.to_vec();
    for range in [29..33, 20..24, 15..19] {
        let mut count = Vec::new();
        varint(600_000_000, &mut count);
        large.splice(range, count);
    }
    assert_eq!(small.len(), 40);
    assert_eq!(large.len(), 43);
    for bytes in [small.as_slice(), large.as_slice()] {
        let (result, meter) = measured(64 * 1024, None, || decode_faces(bytes, 2, 4));
        assert_eq!(result.unwrap_err(), "Draco face count mismatch");
        assert_eq!(meter.calls, 0, "{meter:?}");
        assert_eq!(meter.allocated, 0);
    }
}

#[test]
fn malformed_count_families_stay_within_64_kib() {
    let huge = 600_000_000;
    let mut cases: Vec<(&str, Vec<u8>, usize, usize)> = vec![
        ("attribute metadata", stream(0, true, &[huge], &[]), 3, 3),
        ("metadata entries", stream(0, true, &[0, huge], &[]), 3, 3),
        (
            "metadata value bytes",
            stream(0, true, &[0, 1, 0, huge], &[]),
            3,
            3,
        ),
        (
            "metadata children",
            stream(0, true, &[0, 0, huge], &[]),
            3,
            3,
        ),
        ("sequential faces", stream(0, false, &[huge, 3], &[1]), 3, 3),
        (
            "sequential points",
            stream(0, false, &[3, huge], &[1]),
            3,
            3,
        ),
        (
            "Edgebreaker vertices",
            stream(1, false, &[0, huge, 4], &[]),
            4,
            4,
        ),
        (
            "Edgebreaker faces",
            stream(1, false, &[2, 4, huge], &[]),
            4,
            4,
        ),
        (
            "Edgebreaker symbols",
            stream(1, false, &[2, 4, 4, 0, huge], &[]),
            4,
            4,
        ),
        (
            "Edgebreaker splits",
            stream(1, false, &[2, 4, 4, 0, 3, huge], &[]),
            4,
            4,
        ),
        (
            "topology events",
            stream(1, false, &[2, 4, 4, 0, 3, 0, huge], &[]),
            4,
            4,
        ),
        // Initial binary stream then the first valence context count.
        (
            "valence context symbols",
            stream(1, false, &[2, 4, 4, 0, 3, 0, 0], &[128, 1, 0]),
            4,
            4,
        ),
        (
            "raw entropy alphabet",
            stream(0, false, &[3, 3], &[0, 1, 1]),
            3,
            3,
        ),
    ];
    varint(huge, &mut cases[11].1);
    varint(huge, &mut cases[12].1);
    // Enough bytes to pass the sequential face-count plausibility check.
    cases[12].1.extend_from_slice(&[0; 9]);
    cases.push((
        "attribute seam decoders",
        stream(1, false, &[2, 4, 4], &[255, 3, 0, 0, 128, 1, 0, 128, 1, 0]),
        4,
        4,
    ));
    // A tiny raw table can request a 4 MiB lookup. A denied allocation must be
    // an ordinary decoder error, not handle_alloc_error/process abort.
    cases.push((
        "entropy lookup allocation",
        stream(0, false, &[3, 3], &[0, 1, 18, 1, 2, 0, 64, 1, 0, 0]),
        3,
        3,
    ));
    for (name, bytes, faces, points) in cases {
        let (result, meter) = measured(64 * 1024, None, || decode_faces(&bytes, faces, points));
        assert!(result.is_err(), "{name} accepted");
        assert!(
            meter.allocated <= 64 * 1024 && meter.calls <= 128,
            "{name}: {meter:?}"
        );
        if name == "entropy lookup allocation" {
            assert!(
                meter.denied && result.unwrap_err().contains("allocation"),
                "{meter:?}"
            );
        }
        println!(
            "{name}: {} allocation calls, {} allocated bytes",
            meter.calls, meter.allocated
        );
    }
}

#[test]
fn every_fixture_allocation_can_fail_without_aborting() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for row in include_str!("records/faces.tsv")
        .lines()
        .filter(|line| line.starts_with("fixtures/"))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let faces: usize = fields[1].parse().unwrap();
        let points: usize = fields
            .last()
            .unwrap()
            .parse()
            .expect("recorded true point count");
        let bytes = std::fs::read(root.join(format!("{}.drc", fields[0]))).unwrap();
        let (baseline, meter) = measured(16 * 1024 * 1024, None, || {
            decode_faces(&bytes, faces, points)
        });
        assert!(
            baseline.is_ok(),
            "{} baseline: {baseline:?} {meter:?}",
            fields[0]
        );
        println!(
            "{}: {} allocation failures exercised",
            fields[0], meter.calls
        );
        for failure in 0..meter.calls {
            let (result, attempt) = measured(16 * 1024 * 1024, Some(failure), || {
                decode_faces(&bytes, faces, points)
            });
            assert!(
                attempt.denied && result.is_err(),
                "{} allocation {failure}: {attempt:?}",
                fields[0]
            );
        }
    }
}
