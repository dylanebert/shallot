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
