use shallot_draco::{decode_faces, parse_header, Method};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}.drc",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}
fn refuse(bytes: &[u8], faces: usize, points: usize, message: &str) {
    let result = std::panic::catch_unwind(|| decode_faces(bytes, faces, points));
    assert!(result.is_ok(), "{message}: panic");
    let error = result.unwrap().expect_err(message);
    assert!(
        error.to_string().to_lowercase().contains(message),
        "expected refusal naming {message}, received {error}"
    );
}

#[test]
fn signature_is_draco() {
    let mut bytes = fixture("sequential-u8");
    bytes[0] = b'X';
    refuse(&bytes, 3, 255, "signature");
}
#[test]
fn only_bitstream_version_2_2_is_claimed() {
    for (major, minor) in [(1, 1), (1, 2), (2, 0), (2, 1), (2, 3), (3, 0)] {
        let mut bytes = fixture("sequential-u8");
        bytes[5] = major;
        bytes[6] = minor;
        refuse(&bytes, 3, 255, "version");
    }
}
#[test]
fn point_cloud_is_refused_by_name() {
    let mut bytes = fixture("sequential-u8");
    bytes[7] = 0;
    refuse(&bytes, 3, 255, "point cloud");
}
#[test]
fn geometry_type_must_be_known() {
    let mut bytes = fixture("sequential-u8");
    bytes[7] = 2;
    refuse(&bytes, 3, 255, "geometry");
}
#[test]
fn mesh_method_must_be_known() {
    let mut bytes = fixture("sequential-u8");
    bytes[8] = 2;
    refuse(&bytes, 3, 255, "method");
}
#[test]
fn predictive_edgebreaker_is_refused_by_name() {
    let mut bytes = fixture("standard-closed");
    let header = parse_header(&bytes).unwrap();
    bytes[header.size] = 1;
    refuse(&bytes, 4, 4, "predictive");
}
#[test]
fn unknown_edgebreaker_traversal_is_refused() {
    let mut bytes = fixture("standard-closed");
    let header = parse_header(&bytes).unwrap();
    bytes[header.size] = 3;
    assert!(decode_faces(&bytes, 4, 4).is_err());
}
#[test]
fn header_methods_and_consumed_ranges() {
    for (name, method) in [
        ("sequential-u8", Method::Sequential),
        ("standard-closed", Method::Edgebreaker),
        ("valence-closed", Method::Edgebreaker),
    ] {
        let bytes = fixture(name);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.method, method);
        assert_eq!(header.size, 11);
        assert!(header.metadata.is_none());
        assert!(header.attributes.is_empty());
    }
}

#[test]
fn metadata_values_use_varui32_not_the_texts_ui8() {
    // Pinned encoder output: UI8 would read 0x82 as 130, consume the second
    // varint byte as value data, and lose the nested metadata boundary.
    let bytes = fixture("metadata-long");
    let header = parse_header(&bytes).unwrap();
    let meta = header.metadata.unwrap();
    assert_eq!(meta.entries.len(), 1);
    assert_eq!(meta.entries[0].0, b"long");
    assert_eq!(meta.entries[0].1, (0..130).collect::<Vec<u8>>());
    assert_eq!(meta.children.len(), 1);
    assert_eq!(meta.children[0].0, b"nested");
    assert_eq!(meta.children[0].1.entries[0].0, b"note");
    assert_eq!(meta.children[0].1.entries[0].1, b"child");
    assert_eq!(header.attributes.len(), 1);
    assert_eq!(header.attributes[0].0, 0);
    assert_eq!(header.attributes[0].1.entries[0].0, b"semantic");
    assert_eq!(header.attributes[0].1.entries[0].1, b"position");
    assert!(header.size > 130);
    assert_eq!(decode_faces(&bytes, 4, 4).unwrap().len(), 4);
    for end in 11..header.size {
        let result = std::panic::catch_unwind(|| parse_header(&bytes[..end]));
        assert!(result.is_ok(), "metadata truncated at {end}: panic");
        assert!(
            result.unwrap().is_err(),
            "metadata truncated at {end}: accepted"
        );
    }
}

#[test]
fn trusted_face_counts_must_match() {
    for (name, faces, points) in [
        ("sequential-u8", 3, 255),
        ("sequential-compressed-raw", 3, 3),
        ("standard-closed", 4, 4),
        ("valence-closed", 4, 4),
    ] {
        let bytes = fixture(name);
        assert!(decode_faces(&bytes, faces, points).is_ok(), "{name}");
        for expected in [faces - 1, faces + 1] {
            refuse(&bytes, expected, points, "face count mismatch");
        }
    }
}
#[test]
fn trusted_point_counts_must_match() {
    for (name, faces, points) in [
        ("sequential-u8", 3, 255),
        ("sequential-compressed-raw", 3, 3),
        ("standard-closed", 4, 4),
        ("valence-closed", 4, 4),
        ("standard-seams", 2, 6),
        ("valence-seams", 2, 6),
    ] {
        let bytes = fixture(name);
        assert!(decode_faces(&bytes, faces, points).is_ok(), "{name}");
        for expected in [points - 1, points + 1] {
            refuse(&bytes, faces, expected, "point count mismatch");
        }
    }
}

#[test]
fn declared_counts_are_checked_before_connectivity_payload() {
    // These fixture prefixes contain only the header, method and mesh counts.
    // Mismatch errors must win over missing geometry/entropy payload errors.
    let sequential = fixture("sequential-u8");
    refuse(&sequential[..12], 2, 255, "face count mismatch");
    refuse(&sequential[..14], 3, 4, "point count mismatch");
    for name in ["standard-closed", "valence-closed"] {
        let bytes = fixture(name);
        refuse(&bytes[..14], 3, 4, "face count mismatch");
        refuse(&bytes[..14], 4, 3, "point count mismatch");
    }
}

fn metadata(body: &[u8]) -> Vec<u8> {
    let original = fixture("sequential-u8");
    let mut bytes = original[..11].to_vec();
    bytes[10] = 128;
    bytes.extend_from_slice(body);
    bytes.extend_from_slice(&original[11..]);
    bytes
}
#[test]
fn zero_length_metadata_value_is_refused_like_the_pinned_decoder() {
    // No attribute metadata, one file entry: key "x", zero value bytes, no children.
    let bytes = metadata(&[0, 1, 1, b'x', 0, 0]);
    let result = std::panic::catch_unwind(|| parse_header(&bytes));
    assert!(result.is_ok());
    assert!(result.unwrap().is_err());
    assert!(decode_faces(&bytes, 3, 255).is_err());
}
#[test]
fn malformed_metadata_counts_and_varints_do_not_panic() {
    for body in [
        vec![255, 255, 255, 255, 31],
        vec![0, 255, 255, 255, 255, 15],
        vec![0, 1, 1, b'x', 255, 255, 255, 255, 31],
        vec![0, 0, 255, 255, 255, 255, 15],
    ] {
        let bytes = metadata(&body);
        let result = std::panic::catch_unwind(|| parse_header(&bytes));
        assert!(result.is_ok(), "malformed metadata: panic");
        assert!(result.unwrap().is_err(), "malformed metadata: accepted");
    }
}
#[test]
fn reserved_header_flags_are_ignored_like_the_pinned_decoder() {
    let original = fixture("sequential-u8");
    let expected = decode_faces(&original, 3, 255).unwrap();
    let mut bytes = original;
    bytes[9] = 255;
    bytes[10] = 127;
    assert_eq!(decode_faces(&bytes, 3, 255).unwrap(), expected);
}
