use shallot_basisu::{Channels, Ktx2, Model, Transfer};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}.ktx2",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}
fn put(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn dfd(bytes: &[u8]) -> usize {
    u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize
}
fn metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
    let start = bytes.len().next_multiple_of(4);
    bytes.resize(start, 0);
    let mut entry = Vec::new();
    entry.extend_from_slice(key.as_bytes());
    entry.push(0);
    entry.extend_from_slice(value.as_bytes());
    entry.push(0);
    bytes.extend_from_slice(&(entry.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&entry);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let length = bytes.len() - start;
    put(bytes, 56, start as u32);
    put(bytes, 60, length as u32);
}

macro_rules! header_rule {
    ($name:ident, $offset:expr, $value:expr) => {
        #[test]
        fn $name() {
            let mut bytes = fixture("uastc-none");
            put(&mut bytes, $offset, $value);
            assert!(Ktx2::parse(&bytes).is_err());
        }
    };
}
header_rule!(identifier, 0, 0);
header_rule!(vk_format, 12, 1);
header_rule!(type_size, 16, 4);
header_rule!(width_multiple_of_four, 20, 6);
header_rule!(height_multiple_of_four, 24, 6);
header_rule!(width_nonzero, 20, 0);
header_rule!(height_nonzero, 24, 0);
header_rule!(no_depth, 28, 4);
header_rule!(no_layers, 32, 1);
header_rule!(one_face, 36, 6);
header_rule!(level_count_nonzero, 40, 0);
header_rule!(level_count_bound, 40, 99);
header_rule!(dfd_range, 48, u32::MAX);
header_rule!(dfd_length, 52, 43);
header_rule!(metadata_range, 56, u32::MAX);
header_rule!(global_range, 64, u32::MAX);
header_rule!(level_range, 80, u32::MAX);
header_rule!(level_length, 88, 0);
header_rule!(uncompressed_length, 96, 17);
header_rule!(no_unexpected_global_data, 72, 20);

macro_rules! descriptor_rule {
    ($name:ident, $offset:expr, $value:expr) => {
        #[test]
        fn $name() {
            let mut bytes = fixture("uastc-none");
            let at = dfd(&bytes) + $offset;
            bytes[at] = $value;
            assert!(Ktx2::parse(&bytes).is_err());
        }
    };
}
descriptor_rule!(dfd_total_size, 0, 0);
descriptor_rule!(dfd_vendor, 4, 1);
descriptor_rule!(dfd_version, 8, 1);
descriptor_rule!(dfd_descriptor_size, 10, 0);
descriptor_rule!(dfd_model, 12, 0);
descriptor_rule!(dfd_transfer, 14, 0);
descriptor_rule!(no_premultiplied_alpha, 15, 1);
descriptor_rule!(block_dimensions, 16, 0);
descriptor_rule!(sample_offset, 28, 1);
descriptor_rule!(sample_length, 30, 63);
descriptor_rule!(sample_qualifiers, 31, 128);
descriptor_rule!(sample_position, 32, 1);
descriptor_rule!(uastc_invalid_channel, 31, 15);

#[test]
fn etc1s_requires_basislz() {
    let mut bytes = fixture("etc1s-red");
    put(&mut bytes, 44, 2);
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn uastc_requires_none_or_zstandard() {
    let mut bytes = fixture("uastc-none");
    put(&mut bytes, 44, 1);
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn etc1s_requires_global_data_and_zero_uncompressed_length() {
    let mut bytes = fixture("etc1s-red");
    put(&mut bytes, 72, 0);
    assert!(Ktx2::parse(&bytes).is_err());
    let mut bytes = fixture("etc1s-red");
    put(&mut bytes, 96, 64);
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn invalid_etc1s_channel_pair_is_refused() {
    let mut bytes = fixture("etc1s-rg");
    let at = dfd(&bytes);
    bytes[at + 47] = 15;
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn channels_describe_the_payload() {
    for (name, model, channels) in [
        ("etc1s-red", Model::Etc1s, Channels::Red),
        ("etc1s-rg", Model::Etc1s, Channels::RedGreen),
        ("uastc-red", Model::Uastc, Channels::Red),
        ("uastc-rg", Model::Uastc, Channels::RedGreen),
        ("uastc-none", Model::Uastc, Channels::Rgb),
    ] {
        let bytes = fixture(name);
        let image = Ktx2::parse(&bytes).unwrap();
        assert_eq!(image.model, model);
        assert_eq!(image.channels, channels);
    }
    for (source, samples, expected) in [
        ("etc1s-red", vec![0], Channels::Rgb),
        ("etc1s-rg", vec![0, 15], Channels::Rgba),
        ("uastc-none", vec![3], Channels::Rgba),
    ] {
        let mut bytes = fixture(source);
        let at = dfd(&bytes);
        for (i, value) in samples.into_iter().enumerate() {
            bytes[at + 31 + i * 16] = value;
        }
        assert_eq!(Ktx2::parse(&bytes).unwrap().channels, expected);
    }
}
#[test]
fn etc1s_alpha_can_carry_the_linear_qualifier() {
    let mut bytes = fixture("etc1s-rg");
    let at = dfd(&bytes);
    bytes[at + 14] = 2;
    bytes[at + 31] = 0;
    bytes[at + 47] = 0x1f;
    assert_eq!(Ktx2::parse(&bytes).unwrap().channels, Channels::Rgba);
}

#[test]
fn swizzle_is_rgba_or_absent() {
    for value in ["rgba", "bgra"] {
        let mut bytes = fixture("uastc-none");
        metadata(&mut bytes, "KTXswizzle", value);
        assert_eq!(Ktx2::parse(&bytes).is_ok(), value == "rgba");
    }
}
#[test]
fn orientation_is_rd_or_absent() {
    for value in ["rd", "ru"] {
        let mut bytes = fixture("uastc-none");
        metadata(&mut bytes, "KTXorientation", value);
        assert_eq!(Ktx2::parse(&bytes).is_ok(), value == "rd");
    }
}
#[test]
fn transfer_matches_the_texture_role() {
    for (value, expected) in [(1, Transfer::Linear), (2, Transfer::Srgb)] {
        let mut bytes = fixture("uastc-none");
        let at = dfd(&bytes);
        bytes[at + 14] = value;
        let image = Ktx2::parse(&bytes).unwrap();
        assert_eq!(image.transfer, expected);
        assert!(image.require_transfer(expected).is_ok());
        assert!(image
            .require_transfer(if value == 1 {
                Transfer::Srgb
            } else {
                Transfer::Linear
            })
            .is_err());
    }
}
#[test]
fn red_and_rg_are_linear() {
    for name in ["etc1s-red", "etc1s-rg", "uastc-red", "uastc-rg"] {
        let mut bytes = fixture(name);
        let at = dfd(&bytes);
        bytes[at + 14] = 2;
        assert!(Ktx2::parse(&bytes).is_err());
    }
}
#[test]
fn primaries_are_not_checked() {
    for value in [0, 1, 255] {
        let mut bytes = fixture("uastc-none");
        let at = dfd(&bytes);
        bytes[at + 13] = value;
        assert!(Ktx2::parse(&bytes).is_ok());
    }
}
#[test]
fn overlapping_ranges_are_refused() {
    let mut bytes = fixture("uastc-none");
    let at = dfd(&bytes) as u32;
    put(&mut bytes, 80, at);
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn metadata_and_level_alignment() {
    let mut bytes = fixture("uastc-none");
    let at = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
    bytes.push(0);
    put(&mut bytes, 80, at + 1);
    assert!(Ktx2::parse(&bytes).is_err());
}
#[test]
fn mutated_bytes_never_panic() {
    for name in ["uastc-none", "uastc-rg", "etc1s-rg"] {
        let original = fixture(name);
        for i in 0..original.len() {
            for mask in [1, 128, 255] {
                let mut bytes = original.clone();
                bytes[i] ^= mask;
                // BasisLZ and UASTC payload integrity belongs to their transcoders,
                // not to the container. Arbitrary mutations need not be invalid.
                if let Ok(image) = Ktx2::parse(&bytes) {
                    for index in 0..image.levels.len() {
                        let _ = image.level(index);
                    }
                }
            }
        }
    }
}
