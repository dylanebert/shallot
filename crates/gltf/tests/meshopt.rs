use shallot_gltf::meshopt::View;

use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn tool() -> Command {
    let mut command = Command::new("bun");
    command
        .current_dir(root())
        .arg("diagnostics/meshopt-records/records.ts");
    command
}
fn inputs() -> &'static (String, Vec<u8>) {
    static INPUTS: OnceLock<(String, Vec<u8>)> = OnceLock::new();
    INPUTS.get_or_init(|| {
        let output = tool().output().expect("run the corpus sync with bun");
        assert!(
            output.status.success(),
            "meshopt corpus: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let path = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
        let records = (
            std::fs::read_to_string(path.join("views.tsv")).unwrap(),
            std::fs::read(path.join("sources.bin")).unwrap(),
        );
        std::fs::remove_dir_all(path).unwrap();
        records
    })
}
fn assert_record(asset: &str, view: View, output: &[u8], expected: &str) {
    if format!("{:x}", Sha256::digest(output)) == expected {
        return;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "meshopt-{}-{}.bin",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, output).unwrap();
    let diagnostic = tool()
        .args(["--compare", asset, &view.index.to_string()])
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    panic!(
        "{asset} bufferView {}: recorded SHA-256 mismatch\n{}{}",
        view.index,
        String::from_utf8_lossy(&diagnostic.stdout),
        String::from_utf8_lossy(&diagnostic.stderr)
    );
}

fn views() -> impl Iterator<Item = (&'static str, View, &'static [u8], &'static str)> {
    let (manifest, records) = inputs();
    manifest
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(move |line| {
            let fields: Vec<_> = line.split('\t').collect();
            let n: Vec<usize> = fields[1..10].iter().map(|v| v.parse().unwrap()).collect();
            let view = View {
                index: n[0],
                count: n[1],
                stride: n[2],
                mode: n[3] as u32,
                filter: n[4] as u32,
                khr: n[5] != 0,
            };
            (fields[0], view, &records[n[6]..n[6] + n[7]], fields[10])
        })
}

#[test]
fn every_recorded_view_is_bit_equal() {
    let mut modes = [0; 3];
    let mut filters = [0; 5];
    let mut versions = [0; 2];
    for (asset, view, source, expected) in views() {
        let mut output = vec![0; view.count * view.stride];
        view.decode(source, &mut output)
            .unwrap_or_else(|e| panic!("{asset}: {e}"));
        assert_record(asset, view, &output, expected);
        modes[view.mode as usize] += 1;
        filters[view.filter as usize] += 1;
        if view.mode == 0 {
            versions[(source[0] & 15) as usize] += 1;
        }
    }
    assert!(modes.iter().all(|n| *n > 0));
    assert!(filters.iter().all(|n| *n > 0));
    assert!(versions.iter().all(|n| *n > 0));
    println!("modes {modes:?}; filters {filters:?}; attribute versions {versions:?}");
}

macro_rules! recorded_class {
    ($name:ident, $predicate:expr) => {
        #[test]
        fn $name() {
            let mut covered = 0;
            for (asset, view, source, expected) in views() {
                if ($predicate)(view, source) {
                    let mut output = vec![0; view.count * view.stride];
                    view.decode(source, &mut output).unwrap();
                    assert_record(asset, view, &output, expected);
                    covered += 1;
                }
            }
            assert!(covered > 0);
        }
    };
}
recorded_class!(attributes_v0, |v: View, s: &[u8]| v.mode == 0
    && s[0] == 0xa0);
recorded_class!(attributes_v1, |v: View, s: &[u8]| v.mode == 0
    && s[0] == 0xa1);
recorded_class!(triangles, |v: View, _: &[u8]| v.mode == 1);
recorded_class!(indices, |v: View, _: &[u8]| v.mode == 2);
recorded_class!(filter_none, |v: View, _: &[u8]| v.filter == 0);
recorded_class!(filter_octahedral, |v: View, _: &[u8]| v.filter == 1);
recorded_class!(filter_quaternion, |v: View, _: &[u8]| v.filter == 2);
recorded_class!(filter_exponential, |v: View, _: &[u8]| v.filter == 3);
recorded_class!(filter_color, |v: View, _: &[u8]| v.filter == 4);

#[test]
fn truncated_and_corrupted_streams_name_the_view_without_panicking() {
    for (asset, view, source, _) in views() {
        let mut output = vec![0; view.count * view.stride];
        for length in [0, 1, source.len() / 2, source.len() - 1] {
            let error = view
                .decode(&source[..length], &mut output)
                .expect_err(asset);
            assert!(error.contains(&format!("bufferView {}:", view.index)));
        }
        let mut corrupt = source.to_vec();
        corrupt[0] = 0;
        assert!(view.decode(&corrupt, &mut output).is_err(), "{asset}");
    }
}

#[test]
fn extension_validity_rules() {
    let (_, view, source, _) = views().next().unwrap();
    let mut output = vec![0; view.count * view.stride];
    for bad in [
        View { count: 0, ..view },
        View {
            count: usize::MAX,
            ..view
        },
        View { stride: 0, ..view },
        View { stride: 3, ..view },
        View {
            stride: 260,
            ..view
        },
        View { mode: 3, ..view },
        View { filter: 5, ..view },
    ] {
        assert!(bad.decode(source, &mut output).is_err());
    }
    for mode in 0..3 {
        for stride in 1..=260 {
            for filter in 0..=5 {
                let view = View {
                    index: 99,
                    count: 3,
                    stride,
                    mode,
                    filter,
                    khr: true,
                };
                let mut output = vec![0; 3 * stride];
                let error = view.decode(&[], &mut output).unwrap_err();
                let layout_valid = if mode == 0 {
                    (4..=256).contains(&stride) && stride % 4 == 0
                } else {
                    (stride == 2 || stride == 4) && filter == 0
                };
                let filter_valid = match filter {
                    0 => true,
                    1 | 4 => stride == 4 || stride == 8,
                    2 => stride == 8,
                    3 => stride % 4 == 0,
                    _ => false,
                };
                assert_eq!(
                    error.contains("invalid mode") || error.contains("invalid filter"),
                    !layout_valid || !filter_valid,
                    "{mode} {stride} {filter}: {error}"
                );
            }
        }
    }
}

#[test]
fn ext_refuses_version_one_and_color_but_khr_accepts_both() {
    for (_, view, source, _) in views() {
        if view.mode == 0 && (source[0] == 0xa1 || view.filter == 4) {
            let mut output = vec![0; view.count * view.stride];
            View { khr: true, ..view }
                .decode(source, &mut output)
                .unwrap();
            assert!(View { khr: false, ..view }
                .decode(source, &mut output)
                .is_err());
        }
    }
    let (_, view, source, _) = views().find(|(_, v, _, _)| v.mode == 1).unwrap();
    let bad = View { count: 1, ..view };
    assert!(bad
        .decode(source, &mut vec![0; bad.stride])
        .unwrap_err()
        .contains("invalid mode"));
}

#[test]
fn attribute_block_boundaries_support_every_valid_stride() {
    for stride in (4..=256).step_by(4) {
        for count in [1usize, 16, 17, 257, 513] {
            for version in [0, 1] {
                let block_size = ((8192 / stride) & !15).min(256);
                let mut source = vec![0xa0 + version];
                for offset in (0..count).step_by(block_size) {
                    let n = (count - offset).min(block_size);
                    if version == 1 {
                        source.extend(vec![0xaa; stride / 4]);
                    } else {
                        source.extend(vec![0; n.div_ceil(16).div_ceil(4) * stride]);
                    }
                }
                let tail_size = stride + if version == 1 { stride / 4 } else { 0 };
                let padding = tail_size.max(if version == 0 { 32 } else { 24 });
                source.extend(vec![0; padding - tail_size]);
                source.extend((0..stride).map(|b| b as u8));
                if version == 1 {
                    source.extend(vec![0; stride / 4]);
                }
                let view = View {
                    index: 0,
                    count,
                    stride,
                    mode: 0,
                    filter: 0,
                    khr: true,
                };
                let mut output = vec![0; count * stride];
                view.decode(&source, &mut output).unwrap();
                for value in output.chunks_exact(stride) {
                    assert!(value.iter().enumerate().all(|(j, b)| *b == j as u8));
                }
                source.push(0);
                assert!(view.decode(&source, &mut output).is_err());
            }
        }
    }
}

#[test]
fn malformed_payloads_are_bounded() {
    let mut random = 0x12345678u32;
    for mode in 0..3 {
        for stride in if mode == 0 {
            vec![4, 8, 12, 256]
        } else {
            vec![2, 4]
        } {
            for size in 0..512 {
                let mut source = vec![0; size];
                for b in &mut source {
                    random ^= random << 13;
                    random ^= random >> 17;
                    random ^= random << 5;
                    *b = random as u8;
                }
                if size > 0 {
                    source[0] = [0xa1, 0xe1, 0xd1][mode as usize];
                }
                let view = View {
                    index: 7,
                    count: 33,
                    stride,
                    mode,
                    filter: 0,
                    khr: true,
                };
                let _ = view.decode(&source, &mut vec![0; view.count * stride]);
            }
        }
    }
}

#[test]
fn unwritten_fifos_match_the_pinned_decoder() {
    // generate.ts records: 18 source bytes, then three indices; three codes per width.
    let mut records = include_bytes!("meshopt/fifo.bin").as_slice();
    for stride in [2, 4] {
        for _ in 0..3 {
            let (source, rest) = records.split_at(18);
            let (expected, rest) = rest.split_at(3 * stride);
            records = rest;
            let view = View {
                index: 42,
                count: 3,
                stride,
                mode: 1,
                filter: 0,
                khr: true,
            };
            let mut output = vec![0; expected.len()];
            view.decode(source, &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }
    assert!(records.is_empty());
}

#[test]
fn invalid_varints_channels_and_tables_are_refused() {
    let mut triangle = vec![0xe1, 0x00];
    triangle.extend([0; 16]);
    let view = View {
        index: 42,
        count: 3,
        stride: 2,
        mode: 1,
        filter: 0,
        khr: true,
    };
    triangle[1] = 0xf0;
    triangle[2] = 0xff;
    assert!(view
        .decode(&triangle, &mut [0; 6])
        .unwrap_err()
        .contains("table"));
    let view = View {
        mode: 2,
        count: 1,
        ..view
    };
    assert!(view
        .decode(&[0xd1, 255, 255, 255, 255, 255, 0, 0, 0, 0], &mut [0; 2])
        .unwrap_err()
        .contains("varint"));
    for channel in [3, 4, 0x10, 0x11, 255] {
        let mut source = vec![0xa1, 0xaa];
        source.extend([0; 24]);
        *source.last_mut().unwrap() = channel;
        let view = View {
            mode: 0,
            stride: 4,
            count: 1,
            ..view
        };
        assert!(view
            .decode(&source, &mut [0; 4])
            .unwrap_err()
            .contains("channel"));
    }
}
