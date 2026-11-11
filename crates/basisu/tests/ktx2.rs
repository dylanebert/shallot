use sha2::{Digest, Sha256};
use shallot_basisu::{Ktx2, Model};
use std::{path::PathBuf, process::Command, sync::OnceLock};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}
fn corpus() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let output = Command::new("bun")
            .current_dir(directory().join("../../.."))
            .arg("diagnostics/basisu-records/records.ts")
            .output()
            .expect("run pinned corpus sync with bun");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
    })
}
struct Record {
    length: usize,
    sha256: String,
}
type Asset = (String, Vec<u8>, Vec<Record>);

fn records() -> Vec<Asset> {
    let manifest = std::fs::read_to_string(directory().join("records/levels.tsv")).unwrap();
    let mut assets: Vec<Asset> = Vec::new();
    for line in manifest.lines().filter(|s| !s.starts_with('#')) {
        let fields: Vec<_> = line.split('\t').collect();
        let name = fields[0];
        let (model, file) = name.split_once('/').unwrap();
        let level: usize = fields[1].parse().unwrap();
        if level == 0 {
            let path = if model == "fixtures" {
                directory().join(format!("fixtures/{file}.ktx2"))
            } else {
                corpus().join(format!("{model}/glTF-KTX-BasisU/{file}.ktx2"))
            };
            assets.push((name.to_owned(), std::fs::read(path).unwrap(), Vec::new()));
        }
        let asset = assets.last_mut().unwrap();
        assert_eq!(asset.0, name);
        assert_eq!(asset.2.len(), level);
        asset.2.push(Record {
            length: fields[2].parse().unwrap(),
            sha256: fields[3].to_owned(),
        });
    }
    assets
}

#[test]
fn every_recorded_level_is_bit_equal() {
    let mut levels = 0;
    let mut classes = [0; 3];
    for (name, bytes, expected) in records() {
        let image = Ktx2::parse(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        classes[image.supercompression] += 1;
        assert_eq!(image.levels.len(), expected.len());
        for (i, expected) in expected.iter().enumerate() {
            let actual = image
                .level(i)
                .unwrap_or_else(|e| panic!("{name} level {i}: {e}"));
            let digest = format!("{:x}", Sha256::digest(&actual));
            assert!(
                actual.len() == expected.length && digest == expected.sha256,
                "{name} level {i}: expected length {} SHA-256 {}, actual length {} SHA-256 {digest}",
                expected.length, expected.sha256, actual.len()
            );
            levels += 1;
        }
    }
    assert_eq!(levels, 212);
    assert!(classes.iter().all(|n| *n > 0));
    println!("{levels} equal length/SHA-256 records, supercompression counts {classes:?}");
}

#[test]
fn every_etc1s_target_is_bit_equal() {
    use shallot_basisu::Target;
    let assets = records();
    let manifest = std::fs::read_to_string(directory().join("records/etc1s.tsv")).unwrap();
    let mut counts = [0; 4];
    for line in manifest.lines().filter(|s| !s.starts_with('#')) {
        let f: Vec<_> = line.split('\t').collect();
        let asset = assets.iter().find(|a| a.0 == f[0]).unwrap();
        let image = Ktx2::parse(&asset.1).unwrap();
        let level = f[1].parse().unwrap();
        let (index, target) = match f[2] {
            "etc1" => (0, Target::Etc1),
            "etc2-rgba" => (1, Target::Etc2Rgba),
            "bc7" => (2, Target::Bc7),
            "rgba8" => (3, Target::Rgba8),
            _ => panic!("unknown target"),
        };
        let actual = image
            .transcode_etc1s(level, target)
            .unwrap_or_else(|e| panic!("{} level {level} {}: {e}", f[0], f[2]));
        let hash = format!("{:x}", Sha256::digest(&actual));
        assert_eq!(
            (actual.len(), hash.as_str()),
            (f[3].parse().unwrap(), f[4]),
            "{} level {level} {}",
            f[0],
            f[2]
        );
        counts[index] += 1;
    }
    let levels: usize = assets
        .iter()
        .filter(|a| Ktx2::parse(&a.1).unwrap().model == Model::Etc1s)
        .map(|a| a.2.len())
        .sum();
    assert!(levels > 0);
    assert_eq!(counts, [levels; 4]);
    println!("ETC1S bit-equal levels by target: {counts:?}");
}

#[test]
fn malformed_basislz_is_refused_without_panics() {
    use shallot_basisu::Target;
    let targets = [Target::Etc1, Target::Etc2Rgba, Target::Bc7, Target::Rgba8];
    let bytes = std::fs::read(directory().join("fixtures/etc1s-rg.ktx2")).unwrap();
    let image = Ktx2::parse(&bytes).unwrap();
    let refuse = |image: &Ktx2<'_>, label: &str| {
        for target in targets {
            let result = std::panic::catch_unwind(|| image.transcode_etc1s(0, target));
            assert!(result.is_ok(), "{label}: {target:?} panicked");
            assert!(result.unwrap().is_err(), "{label}: {target:?} accepted");
        }
    };
    for end in 0..image.global_data.len() {
        let mut bad = Ktx2::parse(&bytes).unwrap();
        bad.global_data = &image.global_data[..end];
        refuse(&bad, "truncated global data");
    }
    for (offset, value) in [
        (0, 0),
        (2, 0),
        (4, u32::MAX),
        (8, 0),
        (12, 0),
        (16, u32::MAX),
        (20, 2),
        (28, 0),
        (36, 0),
    ] {
        let mut global = image.global_data.to_vec();
        if offset < 4 {
            global[offset..offset + 2].fill(0);
        } else {
            global[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let mut bad = Ktx2::parse(&bytes).unwrap();
        bad.global_data = &global;
        refuse(&bad, "malformed global header or slice descriptor");
    }
    let start = 20 + image.levels.len() * 20;
    let ep = word_for_test(image.global_data, 4);
    let sel = word_for_test(image.global_data, 8);
    let tab = word_for_test(image.global_data, 12);
    for (at, n) in [(start, ep), (start + ep, sel), (start + ep + sel, tab)] {
        for fill in [0, 255] {
            let mut global = image.global_data.to_vec();
            global[at..at + n].fill(fill);
            let mut bad = Ktx2::parse(&bytes).unwrap();
            bad.global_data = &global;
            refuse(&bad, "malformed entropy table/codebook");
        }
    }
    for end in [0, 1, image.levels[0].bytes.len() / 2] {
        let mut bad = Ktx2::parse(&bytes).unwrap();
        bad.levels[0].bytes = &image.levels[0].bytes[..end];
        refuse(&bad, "truncated slices");
    }
    // Entropy has no checksum: a bit flip can encode a different valid image.
    // Adversarial entropy must terminate without panicking even when accepted.
    for i in 0..image.global_data.len() {
        let mut global = image.global_data.to_vec();
        global[i] ^= 255;
        let mut bad = Ktx2::parse(&bytes).unwrap();
        bad.global_data = &global;
        assert!(
            std::panic::catch_unwind(|| bad.transcode_etc1s(0, Target::Bc7)).is_ok(),
            "global byte {i}"
        );
    }
    for i in 0..image.levels[0].bytes.len() {
        let mut payload = image.levels[0].bytes.to_vec();
        payload[i] ^= 255;
        let mut bad = Ktx2::parse(&bytes).unwrap();
        bad.levels[0].bytes = &payload;
        assert!(
            std::panic::catch_unwind(|| bad.transcode_etc1s(0, Target::Bc7)).is_ok(),
            "slice byte {i}"
        );
    }
}
fn word_for_test(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}

#[test]
fn truncated_containers_and_levels_are_refused_without_panics() {
    for (name, bytes, _) in records() {
        let image = Ktx2::parse(&bytes).unwrap();
        // Every prefix of the container metadata, and every indexed payload end.
        let first_payload = image
            .levels
            .iter()
            .map(|l| l.bytes.as_ptr() as usize - bytes.as_ptr() as usize)
            .min()
            .unwrap();
        for end in 0..first_payload {
            assert!(Ktx2::parse(&bytes[..end]).is_err(), "{name}: prefix {end}");
        }
        for level in &image.levels {
            let end = level.bytes.as_ptr() as usize - bytes.as_ptr() as usize + level.bytes.len();
            assert!(
                Ktx2::parse(&bytes[..end - 1]).is_err(),
                "{name}: payload end {end}"
            );
        }
        if image.model == Model::Uastc && image.supercompression == 2 {
            for index in 0..image.levels.len() {
                let data = image.levels[index].bytes;
                let probes = [0, 1, 4, 5, data.len() / 2, data.len() - 1];
                for length in probes {
                    let mut truncated = Ktx2::parse(&bytes).unwrap();
                    truncated.levels[index].bytes = &data[..length];
                    assert!(
                        truncated.level(index).is_err(),
                        "{name}: compressed prefix {length}"
                    );
                }
            }
        }
    }
}

#[test]
fn corrupted_headers_and_zstd_frames_are_refused_without_panics() {
    for (name, bytes, _) in records() {
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(Ktx2::parse(&bad).is_err(), "{name}: magic");
        let image = Ktx2::parse(&bytes).unwrap();
        if image.supercompression == 2 {
            for (index, level) in image.levels.iter().enumerate() {
                let mut payload = level.bytes.to_vec();
                payload[0] ^= 1;
                let mut bad = Ktx2::parse(&bytes).unwrap();
                bad.levels[index].bytes = &payload;
                assert!(bad.level(index).is_err(), "{name}: frame magic");
            }
        }
    }
}
