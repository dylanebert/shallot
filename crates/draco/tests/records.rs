use sha2::{Digest, Sha256};
use shallot_draco::{decode_faces, parse_header};
use std::{collections::BTreeSet, path::PathBuf, process::Command, sync::OnceLock};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}
fn corpus() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let output = Command::new("bun")
            .current_dir(directory().join("../../.."))
            .arg("diagnostics/draco-records/records.ts")
            .output()
            .expect("run pinned Draco corpus sync with bun");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
    })
}

struct Record<'a> {
    name: &'a str,
    faces: usize,
    length: usize,
    sha256: &'a str,
    connectivity_end: usize,
    input_sha256: &'a str,
    paths: &'a str,
    points: usize,
}
impl Record<'_> {
    fn bytes(&self) -> Vec<u8> {
        let path = if self.name.starts_with("fixtures/") {
            directory().join(format!("{}.drc", self.name))
        } else {
            corpus().join(format!("{}.drc", self.name))
        };
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            self.input_sha256,
            "{} input SHA-256",
            self.name
        );
        bytes
    }
}
fn records() -> Vec<Record<'static>> {
    include_str!("records/faces.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|line| {
            let f: Vec<_> = line.split('\t').collect();
            assert_eq!(f.len(), 8);
            Record {
                name: f[0],
                faces: f[1].parse().unwrap(),
                length: f[2].parse().unwrap(),
                sha256: f[3],
                connectivity_end: f[4].parse().unwrap(),
                input_sha256: f[5],
                paths: f[6],
                points: f[7].parse().unwrap(),
            }
        })
        .collect()
}

#[test]
fn every_recorded_face_list_is_bit_equal_to_the_pinned_decoder() {
    let mut models = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut fixtures = 0;
    let mut streams = 0;
    for record in records() {
        let bytes = record.bytes();
        let faces = decode_faces(&bytes, record.faces, record.points)
            .unwrap_or_else(|e| panic!("{}: {e}", record.name));
        let flattened: Vec<_> = faces
            .iter()
            .flat_map(|face| face.iter().flat_map(|index| index.to_le_bytes()))
            .collect();
        let hash = format!("{:x}", Sha256::digest(&flattened));
        assert_eq!(faces.len(), record.faces, "{} face count", record.name);
        assert_eq!(
            (flattened.len(), hash.as_str()),
            (record.length, record.sha256),
            "{} little-endian flattened u32 faces",
            record.name
        );
        // Stage 1 consumes connectivity only; attribute bytes are not a premise.
        assert_eq!(
            decode_faces(
                &bytes[..record.connectivity_end],
                record.faces,
                record.points
            )
            .unwrap_or_else(|e| panic!("{} connectivity-only: {e}", record.name)),
            faces,
            "{} connectivity-only faces",
            record.name
        );
        paths.extend(record.paths.split(','));
        if record.name.starts_with("fixtures/") {
            fixtures += 1;
        } else {
            models.insert(record.name.split('/').next().unwrap());
            streams += 1;
        }
    }
    assert_eq!(models.len(), 18);
    assert!(models.contains("ABeautifulGame"));
    assert!(models.contains("SunglassesKhronos"));
    assert_eq!(streams, 370);
    assert_eq!(fixtures, 16);
    for path in [
        "sequential.u8",
        "sequential.u16",
        "sequential.varint",
        "sequential.u32",
        "sequential.compressed",
        "symbols.raw",
        "symbols.tagged",
        "rans.frequency.token.0",
        "rans.frequency.token.1",
        "rans.frequency.token.2",
        "rans.frequency.token.3",
        "rans.symbol.init.0",
        "rans.symbol.init.1",
        "rans.symbol.init.2",
        "rans.symbol.init.3",
        "rans.symbol.renormalize",
        "rans.binary.init.0",
        "rans.binary.init.1",
        "rans.binary.init.2",
        "rans.binary.zero",
        "rans.binary.one",
        "edgebreaker.standard",
        "edgebreaker.valence",
        "topology.C",
        "topology.L",
        "topology.E",
        "topology.R",
        "topology.S",
        "topology.split-event",
        "topology.split-merge",
        "topology.split-left",
        "topology.split-right",
        "start.boundary",
        "start.interior",
        "seams.yes",
        "seams.no",
        "vertices.boundary",
        "vertices.interior",
        "vertices.remove-invalid",
        "vertices.retain-invalid",
        "valence.context.2",
        "valence.context.3",
        "valence.context.4",
        "valence.context.5",
        "valence.context.6",
        "valence.context.7",
    ] {
        assert!(paths.contains(path), "uncovered reference path: {path}");
    }
    println!(
        "{streams} corpus streams in {} models, {fixtures} fixtures, {} observed paths",
        models.len(),
        paths.len()
    );
}

#[test]
fn truncated_connectivity_is_refused_without_panics() {
    for record in records() {
        let bytes = record.bytes();
        let header = parse_header(&bytes).unwrap();
        // Exhaust every byte boundary of each small committed fixture. Corpus
        // streams additionally exercise truncation of large tables and topology.
        let ends: Vec<_> = if record.name.starts_with("fixtures/") {
            (0..record.connectivity_end).collect()
        } else {
            vec![
                0,
                header.size,
                record.connectivity_end / 2,
                record.connectivity_end - 1,
            ]
        };
        for end in ends {
            let result = std::panic::catch_unwind(|| {
                decode_faces(&bytes[..end], record.faces, record.points)
            });
            assert!(result.is_ok(), "{} truncated at {end}: panic", record.name);
            assert!(
                result.unwrap().is_err(),
                "{} truncated at {end}: accepted",
                record.name
            );
        }
    }
}
