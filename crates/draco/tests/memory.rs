use shallot_draco::{decode, decode_faces};
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
    largest: usize,
    live: usize,
    peak: usize,
    fail_at: Option<usize>,
    denied: bool,
}
thread_local! { static METER: Cell<Meter> = const { Cell::new(Meter { active: false, calls: 0, allocated: 0, budget: 0, largest: 0, live: 0, peak: 0, fail_at: None, denied: false }) }; }
struct Allocator;
// Live bytes for peak memory; frees of memory allocated before metering saturate.
fn release(size: usize) {
    let _ = METER.try_with(|cell| {
        let mut meter = cell.get();
        if meter.active {
            meter.live = meter.live.saturating_sub(size);
            cell.set(meter);
        }
    });
}
fn admit(size: usize) -> bool {
    METER
        .try_with(|cell| {
            let mut meter = cell.get();
            if !meter.active {
                return true;
            }
            let call = meter.calls;
            meter.largest = meter.largest.max(size);
            meter.calls += 1;
            let allowed =
                meter.fail_at != Some(call) && size <= meter.budget.saturating_sub(meter.live);
            if allowed {
                meter.allocated += size;
                meter.live += size;
                meter.peak = meter.peak.max(meter.live);
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
            release(layout.size());
            unsafe { System.realloc(ptr, layout, size) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        release(layout.size());
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

#[test]
fn every_fixture_attribute_allocation_can_fail_without_aborting() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for row in include_str!("records/faces.tsv")
        .lines()
        .filter(|line| line.starts_with("fixtures/"))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let faces: usize = fields[1].parse().unwrap();
        let points: usize = fields.last().unwrap().parse().unwrap();
        let bytes = std::fs::read(root.join(format!("{}.drc", fields[0]))).unwrap();
        let ids: Vec<u32> = include_str!("records/attributes.tsv")
            .lines()
            .filter(|line| line.split('\t').next() == Some(fields[0]))
            .map(|line| line.split('\t').nth(2).unwrap().parse().unwrap())
            .collect();
        let (baseline, meter) = measured(64 * 1024 * 1024, None, || {
            decode(&bytes, faces, points, &ids)
        });
        assert!(baseline.is_ok(), "{} baseline", fields[0]);
        // Stage 1's bound: the 4 MiB rANS lookup, then caller-count-sized buffers
        // (at most 255 components of 8 bytes per point or corner).
        let bound = 4 << 20 | (points.max(3 * faces) * 255 * 8);
        assert!(meter.largest <= bound, "{}: {meter:?}", fields[0]);
        for failure in 0..meter.calls {
            let (result, attempt) = measured(64 * 1024 * 1024, Some(failure), || {
                decode(&bytes, faces, points, &ids)
            });
            assert!(
                attempt.denied && result.is_err(),
                "{} allocation {failure}",
                fields[0]
            );
        }
    }
}

#[test]
fn declared_attribute_counts_stay_within_64_kib() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    // sequential-u8 ends with zero attribute decoders: replace that byte with
    // one decoder whose attribute count, then component count, is far too large.
    let mut base = std::fs::read(root.join("sequential-u8.drc")).unwrap();
    assert_eq!(base.pop(), Some(0));
    let mut count = base.clone();
    count.push(1);
    varint(600_000_000, &mut count);
    let mut components = base.clone();
    components.extend_from_slice(&[1, 1, 4, 10, 255, 0, 0, 0]);
    // The 255-component float64 attribute is sized by the caller's 255 points.
    for (name, bytes, budget) in [
        ("attributes", count, 64 * 1024),
        ("components", components, 64 * 1024 + 255 * 255 * 8),
    ] {
        let (result, meter) = measured(budget, None, || decode(&bytes, 3, 255, &[0]));
        assert!(result.is_err(), "{name} accepted");
        assert!(
            meter.allocated <= budget && !meter.denied,
            "{name}: {meter:?}"
        );
    }
}

#[test]
fn unrequested_attributes_are_freed_once_decoded() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut bytes = std::fs::read(root.join("sequential-u8.drc")).unwrap();
    assert_eq!(bytes.pop(), Some(0));
    // One decoder of 2,000 int32 attributes of 255 components at the caller's
    // 255 points: 260 KB each, 1 GB if all were kept, in a 24 KB stream. Each
    // has no prediction and zero-width raw values.
    let count = 2000;
    bytes.push(1);
    varint(count, &mut bytes);
    for id in 0..count {
        bytes.extend_from_slice(&[4, 5, 255, 0]);
        varint(id, &mut bytes);
    }
    bytes.extend(std::iter::repeat_n(1, count));
    for _ in 0..count {
        bytes.extend_from_slice(&[0xfe, 0, 0]);
    }
    assert!(bytes.len() < 24 * 1024);
    let one = 255 * 255 * 4;
    let (result, meter) = measured(64 * 1024 * 1024, None, || decode(&bytes, 3, 255, &[7]));
    let mesh = result.unwrap();
    assert_eq!(mesh.attributes.len(), 1);
    assert_eq!(mesh.attributes[0].unique_id, 7);
    // Peak live bytes: the kept attribute plus one decoding at a time.
    assert!(meter.peak <= 4 * one + 512 * 1024, "{meter:?}");
    println!("peak {} bytes for {count} attributes", meter.peak);
    let (result, _) = measured(64 * 1024 * 1024, None, || {
        decode(&bytes, 3, 255, &[count as u32])
    });
    assert_eq!(
        result.err(),
        Some("requested Draco attribute unique id is not in the stream")
    );
}
