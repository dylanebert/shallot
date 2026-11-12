use sha2::{Digest, Sha256};
use shallot_draco::{decode, decode_faces, parse_header, DataType};
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
    assert_eq!(fixtures, 24);
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
        "attribute.decoder.generic",
        "attribute.decoder.integer",
        "attribute.decoder.normals",
        "attribute.decoder.quantization",
        "attribute.integer.i16",
        "attribute.integer.i32",
        "attribute.integer.i8",
        "attribute.integer.u16",
        "attribute.integer.u32",
        "attribute.integer.u8",
        "attribute.prediction.constrained-multi-parallelogram",
        "attribute.prediction.constrained-multi-parallelogram.crease",
        "attribute.prediction.constrained-multi-parallelogram.fallback",
        "attribute.prediction.delta",
        "attribute.prediction.dropped",
        "attribute.prediction.geometric-normal",
        "attribute.prediction.geometric-normal.flip",
        "attribute.prediction.multi-parallelogram",
        "attribute.prediction.multi-parallelogram.fallback",
        "attribute.prediction.none",
        "attribute.prediction.parallelogram",
        "attribute.prediction.parallelogram.fallback",
        "attribute.prediction.tex-coords-deprecated",
        "attribute.prediction.tex-coords-portable",
        "attribute.prediction.tex-coords-portable.equal",
        "attribute.prediction.tex-coords-portable.fallback",
        "attribute.prediction.tex-coords-portable.orientation",
        "attribute.sequencer.linear",
        "attribute.transform.octahedron",
        "attribute.transform.octahedron-canonicalized",
        "attribute.transform.wrap",
        "attribute.transform.wrap.clamp",
        "attribute.transform.wrap.high",
        "attribute.transform.wrap.low",
        "attribute.traversal.corner",
        "attribute.traversal.depth-first",
        "attribute.traversal.prediction-degree",
        "attribute.values.compressed",
        "attribute.values.raw.4",
        "attribute.values.raw.narrow",
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

struct AttributeRecord<'a> {
    name: &'a str,
    index: usize,
    unique_id: u32,
    kind: u8,
    data_type: u8,
    components: u8,
    normalized: bool,
    output: u8,
    // None when the pinned wrapper refuses the conversion.
    digest: Option<(usize, &'a str)>,
}
/// Every unique id the records hold for `name`, as a glTF caller would request.
fn unique_ids(name: &str) -> Vec<u32> {
    let ids: BTreeSet<_> = attribute_records()
        .into_iter()
        .filter(|a| a.name == name)
        .map(|a| a.unique_id)
        .collect();
    ids.into_iter().collect()
}
fn attribute_records() -> Vec<AttributeRecord<'static>> {
    include_str!("records/attributes.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|line| {
            let f: Vec<_> = line.split('\t').collect();
            assert_eq!(f.len(), 10);
            AttributeRecord {
                name: f[0],
                index: f[1].parse().unwrap(),
                unique_id: f[2].parse().unwrap(),
                kind: f[3].parse().unwrap(),
                data_type: f[4].parse().unwrap(),
                components: f[5].parse().unwrap(),
                normalized: f[6] == "1",
                output: f[7].parse().unwrap(),
                digest: (f[8] != "refused").then(|| (f[8].parse().unwrap(), f[9])),
            }
        })
        .collect()
}

#[test]
fn every_recorded_attribute_is_bit_equal_to_the_pinned_decoder() {
    let attributes = attribute_records();
    let mut checked = 0;
    let mut types = BTreeSet::new();
    for record in records() {
        let bytes = record.bytes();
        let mesh = decode(
            &bytes,
            record.faces,
            record.points,
            &unique_ids(record.name),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", record.name));
        assert_eq!(mesh.points, record.points, "{} points", record.name);
        let rows: Vec<_> = attributes
            .iter()
            .filter(|a| a.name == record.name)
            .collect();
        let recorded: BTreeSet<_> = rows.iter().map(|a| a.index).collect();
        assert_eq!(
            recorded,
            (0..mesh.attributes.len()).collect(),
            "{} attribute count",
            record.name
        );
        for row in rows {
            let attribute = &mesh.attributes[row.index];
            assert_eq!(
                (
                    attribute.unique_id,
                    attribute.kind as u8,
                    attribute.data_type as u8,
                    attribute.components,
                    attribute.normalized
                ),
                (
                    row.unique_id,
                    row.kind,
                    row.data_type,
                    row.components,
                    row.normalized
                ),
                "{} attribute {} descriptor",
                record.name,
                row.index
            );
            let output = DataType::from_id(row.output).unwrap();
            let result = attribute.points(output);
            match row.digest {
                Some((length, digest)) => {
                    let values = result.unwrap_or_else(|e| {
                        panic!("{} attribute {} as {output:?}: {e}", record.name, row.index)
                    });
                    assert_eq!(
                        (
                            values.len(),
                            format!("{:x}", Sha256::digest(&values)).as_str()
                        ),
                        (length, digest),
                        "{} attribute {} as {output:?}",
                        record.name,
                        row.index
                    );
                }
                None => assert!(
                    result.is_err(),
                    "{} attribute {} as {output:?}: the pinned wrapper refuses",
                    record.name,
                    row.index
                ),
            }
            types.insert((row.data_type, row.output));
            checked += 1;
        }
    }
    println!("{checked} attribute outputs in {} type pairs", types.len());
}

#[test]
fn truncated_and_corrupted_attributes_are_refused_without_panics() {
    let mut state = 0x2545_f491u32;
    let mut cases = 0;
    for record in records() {
        let bytes = record.bytes();
        let start = record.connectivity_end;
        let ids = unique_ids(record.name);
        let fixture = record.name.starts_with("fixtures/");
        // A deterministic sample: each fixture, which together reach every
        // attribute path, at 16 boundaries and 64 bits; corpus streams at 2 and 2.
        let (end_count, flip_count) = if fixture { (16, 64) } else { (2, 2) };
        let span = bytes.len() - start;
        let ends: Vec<_> = (0..end_count)
            .map(|i| start + i * span / end_count)
            .collect();
        for end in ends {
            let result = std::panic::catch_unwind(|| {
                decode(&bytes[..end], record.faces, record.points, &ids)
            });
            assert!(result.is_ok(), "{} truncated at {end}: panic", record.name);
            assert!(
                result.unwrap().is_err(),
                "{} truncated at {end}: accepted",
                record.name
            );
            cases += 1;
        }
        let flips: Vec<_> = (0..flip_count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                start * 8 + state as usize % (span * 8)
            })
            .collect();
        let mut corrupt = bytes.clone();
        for bit in flips {
            corrupt[bit / 8] ^= 1 << (bit % 8);
            let result = std::panic::catch_unwind(|| {
                decode(&corrupt, record.faces, record.points, &ids).map(|mesh| {
                    for attribute in &mesh.attributes {
                        for id in [1, 2, 3, 4, 5, 6, 9] {
                            let _ = attribute.points(DataType::from_id(id).unwrap());
                        }
                    }
                })
            });
            assert!(result.is_ok(), "{} bit {bit} flipped: panic", record.name);
            corrupt[bit / 8] ^= 1 << (bit % 8);
            cases += 1;
        }
    }
    println!("{cases} truncated or corrupted attribute streams");
}

#[test]
fn wide_and_bool_outputs_are_refused_like_the_pinned_wrapper() {
    // GetAttributeDataArrayForAllPoints takes int8..uint32 and float32 only,
    // even for an attribute already in a 64-bit or bool type.
    let bytes = std::fs::read(directory().join("fixtures/attributes-sequential.drc")).unwrap();
    let ids = unique_ids("fixtures/attributes-sequential");
    let mesh = decode(&bytes, 4, 4, &ids).unwrap();
    let mut refused = 0;
    for attribute in &mesh.attributes {
        for id in [7, 8, 10, 11] {
            assert_eq!(
                attribute.points(DataType::from_id(id).unwrap()).err(),
                Some("unsupported Draco output data type: 64-bit and bool are refused")
            );
        }
        if matches!(
            attribute.data_type,
            DataType::Int64 | DataType::Uint64 | DataType::Float64 | DataType::Bool
        ) {
            assert!(attribute.points(attribute.data_type).is_err());
            refused += 1;
        }
    }
    assert_eq!(refused, 6);
}
