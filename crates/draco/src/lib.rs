//! Draco 2.2 triangle-mesh decoding, bit-equal to Draco 1.5.9's decoder.
//!
//! `decode` returns faces and attributes; `decode_faces` stops after connectivity.
mod allocation;
mod attributes;
mod buffer;
mod edgebreaker;
mod entropy;
mod prediction;
mod traversal;

pub use attributes::{Attribute, AttributeType, DataType};

use allocation::{copy, push};
use buffer::Reader;
pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Sequential,
    Edgebreaker,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Metadata {
    pub entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub children: Vec<(Vec<u8>, Metadata)>,
}

#[derive(Debug)]
pub struct Header {
    pub method: Method,
    pub attributes: Vec<(u32, Metadata)>,
    pub metadata: Option<Metadata>,
    pub size: usize,
}

fn name(r: &mut Reader<'_>) -> Result<Vec<u8>> {
    let size = usize::from(r.byte()?);
    copy(r.take(size)?)
}
fn metadata_node(r: &mut Reader<'_>) -> Result<(Metadata, usize)> {
    let count = r.count()?;
    if count > r.remaining() / 3 {
        return Err("invalid Draco metadata entry count");
    }
    let mut ordered = Vec::new();
    for index in 0..count {
        let key = name(r)?;
        let size = r.count()?;
        if size == 0 {
            return Err("empty Draco metadata value");
        }
        let value = copy(r.take(size)?)?;
        push(&mut ordered, (key, index, value))?;
    }
    // The pinned decoder keeps the last value for each duplicate key.
    ordered.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for (key, _, value) in ordered {
        if let Some(last) = entries.last_mut().filter(|last| last.0 == key) {
            last.1 = value;
        } else {
            push(&mut entries, (key, value))?;
        }
    }
    let count = r.count()?;
    if count > r.remaining() {
        return Err("invalid Draco submetadata count");
    }
    Ok((
        Metadata {
            entries,
            children: Vec::new(),
        },
        count,
    ))
}
fn metadata(r: &mut Reader<'_>) -> Result<Metadata> {
    let (root, pending) = metadata_node(r)?;
    let mut stack = Vec::new();
    push(&mut stack, (Vec::new(), root, pending))?;
    loop {
        if stack.last().unwrap().2 == 0 {
            let (key, mut node, _) = stack.pop().unwrap();
            node.children.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            if node.children.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err("duplicate Draco submetadata name");
            }
            if let Some((_, parent, _)) = stack.last_mut() {
                push(&mut parent.children, (key, node))?;
            } else {
                return Ok(node);
            }
        } else {
            // The pinned decoder counts the first child at level zero.
            if stack.len() > 1001 {
                return Err("Draco metadata nesting exceeds 1000");
            }
            let key = name(r)?;
            stack.last_mut().unwrap().2 -= 1;
            let (node, count) = metadata_node(r)?;
            push(&mut stack, (key, node, count))?;
        }
    }
}

/// A decoded mesh: faces over `points` points, and each attribute's values.
pub struct Mesh {
    pub faces: Vec<[u32; 3]>,
    pub points: usize,
    pub attributes: Vec<Attribute>,
}

impl Mesh {
    /// The first attribute with `unique_id`, as `GetAttributeByUniqueId`.
    pub fn attribute(&self, unique_id: u32) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.unique_id == unique_id)
    }
}

/// Decodes a mesh only when its counts match the caller's trusted geometry
/// counts, keeping the attributes whose unique ids are `attributes` (in glTF,
/// the extension's `attributes` values) and refusing an id the stream lacks.
pub fn decode(
    bytes: &[u8],
    expected_faces: usize,
    expected_points: usize,
    attributes: &[u32],
) -> Result<Mesh> {
    let header = parse_header(bytes)?;
    let mut r = Reader::new(bytes);
    r.take(header.size)?;
    let (faces, connectivity) = match header.method {
        Method::Sequential => (sequential(&mut r, expected_faces, expected_points)?, None),
        Method::Edgebreaker => {
            let mut connectivity = edgebreaker::decode(&mut r, expected_faces, expected_points)?;
            (std::mem::take(&mut connectivity.faces), Some(connectivity))
        }
    };
    let attributes = attributes::decode(
        &mut r,
        &faces,
        expected_points,
        connectivity.as_ref(),
        attributes,
    )?;
    Ok(Mesh {
        faces,
        points: expected_points,
        attributes,
    })
}

/// Decodes connectivity only when its counts match the caller's trusted geometry counts.
pub fn decode_faces(
    bytes: &[u8],
    expected_faces: usize,
    expected_points: usize,
) -> Result<Vec<[u32; 3]>> {
    let header = parse_header(bytes)?;
    let mut r = Reader::new(bytes);
    r.take(header.size)?;
    match header.method {
        Method::Sequential => sequential(&mut r, expected_faces, expected_points),
        Method::Edgebreaker => {
            Ok(edgebreaker::decode(&mut r, expected_faces, expected_points)?.faces)
        }
    }
}

fn sequential(
    r: &mut Reader<'_>,
    expected_faces: usize,
    expected_points: usize,
) -> Result<Vec<[u32; 3]>> {
    let count = r.count()?;
    if count != expected_faces {
        return Err("Draco face count mismatch");
    }
    let points = r.varint(32)? as u32;
    if points as usize != expected_points {
        return Err("Draco point count mismatch");
    }
    if count > u32::MAX as usize / 3 || count > r.remaining() / 3 {
        return Err("invalid Draco sequential face count");
    }
    let compressed = r.byte()? == 0;
    let mut faces = Vec::new();
    if compressed {
        let values = entropy::symbols(r, count * 3, 1)?;
        let mut previous = 0i32;
        for triangle in values.as_chunks::<3>().0 {
            let mut face = [0; 3];
            for (out, encoded) in face.iter_mut().zip(triangle) {
                let difference = (encoded >> 1) as i32;
                previous = if encoded & 1 != 0 {
                    previous.checked_sub(difference)
                } else {
                    previous.checked_add(difference)
                }
                .ok_or("Draco sequential index overflow")?;
                if previous < 0 || previous as u32 >= points {
                    return Err("Draco sequential index out of range");
                }
                *out = previous as u32;
            }
            push(&mut faces, face)?;
        }
    } else {
        for _ in 0..count {
            let mut face = [0; 3];
            for out in &mut face {
                *out = if points < 256 {
                    u32::from(r.byte()?)
                } else if points < 1 << 16 {
                    u32::from(u16::from_le_bytes(r.take(2)?.try_into().unwrap()))
                } else if points < 1 << 21 {
                    r.varint(32)? as u32
                } else {
                    u32::from_le_bytes(r.take(4)?.try_into().unwrap())
                };
                if *out >= points {
                    return Err("Draco sequential index out of range");
                }
            }
            push(&mut faces, face)?;
        }
    }
    Ok(faces)
}

pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    let mut r = Reader::new(bytes);
    if r.take(5)? != b"DRACO" {
        return Err("invalid Draco signature");
    }
    if r.byte()? != 2 || r.byte()? != 2 {
        return Err("unsupported Draco version: expected 2.2");
    }
    match r.byte()? {
        1 => (),
        0 => return Err("Draco point clouds are not supported"),
        _ => return Err("invalid Draco geometry type"),
    }
    let method = match r.byte()? {
        0 => Method::Sequential,
        1 => Method::Edgebreaker,
        _ => return Err("unsupported Draco mesh encoding method"),
    };
    let flags = u16::from_le_bytes(r.take(2)?.try_into().unwrap());
    let mut attributes = Vec::new();
    let mut file_metadata = None;
    if flags & 0x8000 != 0 {
        let count = r.count()?;
        if count > r.remaining() {
            return Err("invalid Draco attribute metadata count");
        }
        for _ in 0..count {
            let id = r.varint(32)? as u32;
            let value = metadata(&mut r)?;
            push(&mut attributes, (id, value))?;
        }
        file_metadata = Some(metadata(&mut r)?);
    }
    Ok(Header {
        method,
        attributes,
        metadata: file_metadata,
        size: r.offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_binary_probability_matches_pinned_faces() {
        // Pinned standard-seams connectivity, zero seam probability, no attribute
        // decoders. records_reference accepts this full stream with these faces.
        let bytes = [
            0x44, 0x52, 0x41, 0x43, 0x4f, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x04, 0x02,
            0x01, 0x02, 0x00, 0x00, 0x01, 0x1f, 0xff, 0x01, 0x11, 0x00, 0x01, 0x10, 0x00,
        ];
        assert_eq!(decode_faces(&bytes, 2, 4).unwrap(), [[0, 1, 2], [1, 3, 2]]);
    }

    #[test]
    fn every_nonzero_sequential_connectivity_method_is_raw_like_the_pin() {
        let bytes = include_bytes!("../tests/fixtures/sequential-u8.drc");
        let mut r = Reader::new(bytes);
        r.take(11).unwrap();
        r.count().unwrap();
        r.count().unwrap();
        let method = r.offset;
        let expected = decode_faces(bytes, 3, 255).unwrap();
        for value in 1..=255 {
            let mut variant = bytes.to_vec();
            variant[method] = value;
            assert_eq!(decode_faces(&variant, 3, 255).unwrap(), expected);
        }
    }

    #[test]
    fn sequential_counts_are_checked_before_connectivity() {
        let bytes = include_bytes!("../tests/fixtures/sequential-u8.drc");
        let mut r = Reader::new(bytes);
        r.take(11).unwrap();
        r.count().unwrap();
        r.count().unwrap();
        let counts_only = &bytes[..r.offset];
        assert_eq!(
            decode_faces(counts_only, 2, 255),
            Err("Draco face count mismatch")
        );
        assert_eq!(
            decode_faces(counts_only, 3, 4),
            Err("Draco point count mismatch")
        );
    }

    #[test]
    fn metadata_duplicate_entries_overwrite_and_duplicate_children_are_refused() {
        let mut r = Reader::new(&[
            5, 1, b'b', 1, 1, 1, b'a', 1, 2, 1, b'b', 1, 3, 1, b'a', 1, 4, 1, b'b', 1, 5, 0,
        ]);
        assert_eq!(
            metadata(&mut r).unwrap().entries,
            [(b"a".to_vec(), vec![4]), (b"b".to_vec(), vec![5])]
        );
        assert!(metadata(&mut Reader::new(&[0, 2, 1, b'x', 0, 0, 1, b'x', 0, 0])).is_err());
    }

    #[test]
    fn attribute_metadata_ids_can_repeat_and_need_not_name_an_attribute() {
        let fixture = include_bytes!("../tests/fixtures/sequential-u8.drc");
        let mut bytes = b"DRACO\x02\x02\x01\x00\x00\x80".to_vec();
        bytes.extend_from_slice(&[2, 7, 0, 0, 7, 0, 0, 0, 0]);
        bytes.extend_from_slice(&fixture[11..]);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.attributes.len(), 2);
        assert_eq!(header.attributes[0].0, 7);
        assert_eq!(header.attributes[1].0, 7);
        // Full DecodeMeshFromBuffer at the pin accepts this no-attribute stream.
        assert_eq!(
            decode_faces(&bytes, 3, 255).unwrap(),
            decode_faces(fixture, 3, 255).unwrap()
        );
    }

    #[test]
    fn metadata_nesting_limit_matches_pinned_levels() {
        for (depth, accepted) in [(1001, true), (1002, false), (2000, false)] {
            let mut bytes = Vec::new();
            for _ in 0..depth {
                bytes.extend_from_slice(&[0, 1, 0]);
            }
            bytes.extend_from_slice(&[0, 0]);
            assert_eq!(
                metadata(&mut Reader::new(&bytes)).is_ok(),
                accepted,
                "depth {depth}"
            );
        }
    }

    #[test]
    fn metadata_handles_many_distinct_entries_and_children() {
        fn varint(mut value: usize, bytes: &mut Vec<u8>) {
            while value >= 128 {
                bytes.push((value as u8 & 127) | 128);
                value >>= 7;
            }
            bytes.push(value as u8);
        }
        let count = 40_000u32;
        let mut bytes = Vec::new();
        varint(count as usize, &mut bytes);
        for i in (0..count).rev() {
            bytes.push(4);
            bytes.extend_from_slice(&i.to_be_bytes());
            bytes.extend_from_slice(&[1, 42]);
        }
        varint(count as usize, &mut bytes);
        for i in (0..count).rev() {
            bytes.push(4);
            bytes.extend_from_slice(&i.to_be_bytes());
            bytes.extend_from_slice(&[0, 0]);
        }
        let mut reader = Reader::new(&bytes);
        let result = metadata(&mut reader).unwrap();
        assert_eq!(reader.remaining(), 0);
        assert_eq!(result.entries.len(), count as usize);
        assert_eq!(result.children.len(), count as usize);
        for (i, ((name, value), (child_name, child))) in
            result.entries.iter().zip(&result.children).enumerate()
        {
            assert_eq!(name, &(i as u32).to_be_bytes());
            assert_eq!(value, &[42]);
            assert_eq!(child_name, name);
            assert!(child.entries.is_empty());
        }
    }
}
