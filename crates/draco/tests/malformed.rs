use shallot_draco::{decode_faces, parse_header};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}.drc",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}
fn refuse(bytes: &[u8], faces: usize, points: usize, label: &str) {
    let result = std::panic::catch_unwind(|| decode_faces(bytes, faces, points));
    assert!(result.is_ok(), "{label}: panic");
    assert!(result.unwrap().is_err(), "{label}: accepted");
}
fn varint(bytes: &[u8], offset: &mut usize) -> u32 {
    let mut value = 0;
    let mut shift = 0;
    loop {
        let byte = bytes[*offset];
        *offset += 1;
        value |= u32::from(byte & 127) << shift;
        if byte < 128 {
            return value;
        }
        shift += 7;
    }
}
fn sequential_data(bytes: &[u8]) -> usize {
    let mut at = parse_header(bytes).unwrap().size;
    varint(bytes, &mut at);
    varint(bytes, &mut at);
    at + 1
}

#[test]
fn sequential_indices_cannot_exceed_the_point_count_in_any_raw_width() {
    for (name, width, points) in [
        ("sequential-u8", 1, 255),
        ("sequential-u16", 2, 65535),
        ("sequential-u32", 4, 2097152),
    ] {
        let mut bytes = fixture(name);
        let at = sequential_data(&bytes);
        bytes[at..at + width].fill(255);
        refuse(&bytes, 3, points, name);
    }
    let mut bytes = fixture("sequential-varint");
    let at = sequential_data(&bytes);
    bytes.splice(at..at + 1, [255, 255, 255, 255, 15]);
    refuse(&bytes, 3, 2097151, "varint index beyond point count");
}
#[test]
fn sequential_varint_indices_cannot_overflow_u32() {
    let mut bytes = fixture("sequential-varint");
    let at = sequential_data(&bytes);
    bytes.splice(at..at + 1, [255, 255, 255, 255, 31]);
    refuse(&bytes, 3, 2097151, "overflowing index varint");
}
#[test]
fn unreasonable_sequential_face_count_is_refused_before_allocation() {
    for (name, points) in [("sequential-u8", 255), ("sequential-compressed-raw", 3)] {
        let mut bytes = fixture(name);
        let at = parse_header(&bytes).unwrap().size;
        bytes.splice(at..at + 1, [255, 255, 255, 255, 15]);
        refuse(&bytes, 3, points, "unreasonable face count");
    }
}
#[test]
fn encoder_output_refused_by_the_pinned_decoders_face_count_bound_is_refused() {
    use sha2::{Digest, Sha256};
    for line in include_str!("records/rejections.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let bytes = fixture(fields[0].strip_prefix("fixtures/").unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), fields[1]);
        refuse(&bytes, 64, 3, fields[2]);
    }
}
#[test]
fn unknown_symbol_scheme_and_raw_precision_are_refused() {
    for (name, faces, points) in [
        ("sequential-compressed-raw", 3, 3),
        ("sequential-compressed-tagged", 64, 131072),
    ] {
        let mut bytes = fixture(name);
        let at = sequential_data(&bytes);
        bytes[at] = 2;
        refuse(&bytes, faces, points, "unknown symbol coding scheme");
    }
    for precision in [0, 19, 255] {
        let mut bytes = fixture("sequential-compressed-raw");
        let at = sequential_data(&bytes);
        assert_eq!(bytes[at], 1, "fixture must exercise raw symbols");
        bytes[at + 1] = precision;
        refuse(&bytes, 3, 3, "invalid raw symbol precision");
    }
}
#[test]
fn entropy_alphabet_and_encoded_size_cannot_overflow() {
    let original = fixture("sequential-compressed-raw");
    let at = sequential_data(&original);
    // Raw scheme, precision, then the varUI32 alphabet count.
    let mut bytes = original.clone();
    let mut end = at + 2;
    varint(&bytes, &mut end);
    bytes.splice(at + 2..end, [255, 255, 255, 255, 31]);
    refuse(&bytes, 3, 3, "overflowing entropy alphabet count");
    // A valid alphabet followed by a data size outside the input is a
    // structural corruption, unlike a flip of an otherwise valid rANS symbol.
    let mut bytes = original;
    let mut cursor = at + 2;
    let symbols = varint(&bytes, &mut cursor);
    let mut i = 0;
    while i < symbols {
        let token = bytes[cursor] & 3;
        let extra = bytes[cursor] >> 2;
        cursor += 1;
        if token == 3 {
            i += u32::from(extra) + 1;
        } else {
            cursor += token as usize;
            i += 1;
        }
    }
    let start = cursor;
    varint(&bytes, &mut cursor);
    bytes.splice(
        start..cursor,
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 1],
    );
    refuse(&bytes, 3, 3, "encoded rANS payload exceeds input");
}

#[test]
fn edgebreaker_counts_must_describe_a_possible_mesh() {
    for name in ["standard-closed", "valence-closed"] {
        let bytes = fixture(name);
        let start = parse_header(&bytes).unwrap().size + 1;
        let mut cursor = start;
        let vertex_at = cursor;
        varint(&bytes, &mut cursor);
        let face_at = cursor;
        varint(&bytes, &mut cursor);
        cursor += 1; // non-position attribute-connectivity count
        let symbol_at = cursor;
        varint(&bytes, &mut cursor);
        let split_at = cursor;
        for (at, value, label) in [
            (vertex_at, 127, "more vertices than face corners"),
            (vertex_at, 2, "too few possible vertex edges"),
            (face_at, 0, "zero faces with nonzero vertices"),
            (symbol_at, 5, "more symbols than faces"),
            (symbol_at, 1, "too few symbols for face count"),
            (split_at, 127, "more split symbols than symbols"),
        ] {
            let mut bad = bytes.clone();
            bad[at] = value;
            refuse(&bad, 4, 4, label);
        }
    }
}
#[test]
fn edgebreaker_first_symbol_cannot_attach_to_a_nonexistent_active_edge() {
    let mut bytes = fixture("standard-closed");
    let mut at = parse_header(&bytes).unwrap().size + 1;
    varint(&bytes, &mut at); // vertices
    varint(&bytes, &mut at); // faces
    at += 1; // attribute data
    varint(&bytes, &mut at); // symbols
    varint(&bytes, &mut at); // splits
    assert_eq!(varint(&bytes, &mut at), 0); // topology events
    assert_eq!(varint(&bytes, &mut at), 1); // traversal bytes
    assert_eq!(bytes[at] & 7, 7); // first symbol E
    bytes[at] &= !1; // C instead of E: needs an active edge that does not exist
    refuse(&bytes, 4, 4, "first C symbol without active edge");
}
