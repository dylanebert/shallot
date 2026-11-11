//! KTX2 LDR paths from Basis Universal 073eb6a0, restricted to KHR_texture_basisu.
//! Altered source: checked Rust slices; no array, cube, HDR or primaries conversion.
//! Copyright 2016–2026 Binomial LLC. See ../licenses/{LICENSE,NOTICE}-basis-universal.

mod entropy;
mod etc1s;
mod uastc;
#[cfg(target_arch = "wasm32")]
mod wasm;
mod zstd;
pub use etc1s::Target;
pub use uastc::UastcTarget;

type Result<T> = std::result::Result<T, &'static str>;

fn le(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .enumerate()
        .fold(0, |v, (i, b)| v | (usize::from(*b) << (8 * i)))
}
fn range(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8]> {
    bytes
        .get(offset..offset.checked_add(length).ok_or("range overflow")?)
        .ok_or("truncated KTX2")
}
fn word(bytes: &[u8], offset: usize) -> Result<usize> {
    Ok(le(range(bytes, offset, 4)?))
}
fn wide(bytes: &[u8], offset: usize) -> Result<usize> {
    let value = u64::from_le_bytes(range(bytes, offset, 8)?.try_into().unwrap());
    usize::try_from(value).map_err(|_| "KTX2 offset too large")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Etc1s,
    Uastc,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Rgb,
    Rgba,
    Red,
    RedGreen,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    Linear,
    Srgb,
}

#[derive(Debug)]
pub struct Level<'a> {
    pub bytes: &'a [u8],
    pub uncompressed_length: usize,
}

#[derive(Debug)]
pub struct Ktx2<'a> {
    pub width: usize,
    pub height: usize,
    pub model: Model,
    pub channels: Channels,
    pub transfer: Transfer,
    pub supercompression: usize,
    pub levels: Vec<Level<'a>>,
    pub global_data: &'a [u8],
}

impl<'a> Ktx2<'a> {
    /// Parses a 2D glTF Basis Universal image. Primaries are retained by neither
    /// the decoder nor renderer; transfer is checked against the role separately.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if range(bytes, 0, 12)? != b"\xabKTX 20\xbb\r\n\x1a\n" {
            return Err("invalid KTX2 identifier");
        }
        range(bytes, 0, 80)?;
        if word(bytes, 12)? != 0 {
            return Err("vkFormat must be undefined");
        }
        if word(bytes, 16)? != 1 {
            return Err("typeSize must be 1");
        }
        let width = word(bytes, 20)?;
        let height = word(bytes, 24)?;
        if width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0 {
            return Err("dimensions must be positive multiples of 4");
        }
        if word(bytes, 28)? != 0 || word(bytes, 32)? != 0 || word(bytes, 36)? != 1 {
            return Err("KTX2 must be 2D, not an array or cube");
        }
        let count = word(bytes, 40)?;
        if count == 0 || count > (usize::BITS - width.max(height).leading_zeros()) as usize {
            return Err("invalid levelCount");
        }
        let index_end = 80 + count * 24;
        range(bytes, 80, count * 24)?;
        let supercompression = word(bytes, 44)?;
        let dfd_offset = word(bytes, 48)?;
        let dfd = range(bytes, dfd_offset, word(bytes, 52)?)?;
        if dfd_offset < index_end
            || dfd_offset % 4 != 0
            || dfd.len() < 44
            || word(dfd, 0)? != dfd.len()
        {
            return Err("invalid DFD range or size");
        }
        if word(dfd, 4)? != 0
            || le(range(dfd, 8, 2)?) != 2
            || le(range(dfd, 10, 2)?) + 4 != dfd.len()
            || (dfd.len() - 28) % 16 != 0
        {
            return Err("invalid DFD descriptor");
        }
        let model = match (dfd[12], supercompression) {
            (163, 1) => Model::Etc1s,
            (166, 0 | 2) => Model::Uastc,
            _ => return Err("invalid colorModel or supercompressionScheme"),
        };
        let transfer = match dfd[14] {
            1 => Transfer::Linear,
            2 => Transfer::Srgb,
            _ => return Err("invalid transferFunction"),
        };
        if dfd[15] != 0 {
            return Err("premultiplied alpha is not supported");
        }
        if dfd[16..20] != [3, 3, 0, 0] {
            return Err("invalid DFD block dimensions");
        }
        for (i, sample) in dfd[28..].as_chunks::<16>().0.iter().enumerate() {
            let (offset, bits) = if model == Model::Etc1s {
                (i * 64, 63)
            } else {
                (0, 127)
            };
            if le(&sample[..2]) != offset
                || sample[2] != bits
                || sample[3] & 0xe0 != 0
                || sample[4..8] != [0; 4]
            {
                return Err("invalid DFD sample layout");
            }
        }
        let samples: Vec<_> = dfd[28..]
            .as_chunks::<16>()
            .0
            .iter()
            .map(|s| s[3] & 15)
            .collect();
        let channels = match (model, samples.as_slice()) {
            (Model::Etc1s, [0]) | (Model::Uastc, [0]) => Channels::Rgb,
            (Model::Etc1s, [0, 15]) | (Model::Uastc, [3]) => Channels::Rgba,
            (Model::Etc1s, [3]) | (Model::Uastc, [4]) => Channels::Red,
            (Model::Etc1s, [3, 4]) | (Model::Uastc, [6]) => Channels::RedGreen,
            _ => return Err("invalid DFD channels"),
        };
        if matches!(channels, Channels::Red | Channels::RedGreen) && transfer != Transfer::Linear {
            return Err("red and red-green require linear transferFunction");
        }
        let kvd_offset = word(bytes, 56)?;
        let kvd = range(bytes, kvd_offset, word(bytes, 60)?)?;
        let mut kv = kvd;
        while !kv.is_empty() {
            let size = word(kv, 0)?;
            let entry = range(kv, 4, size)?;
            let end = entry
                .iter()
                .position(|b| *b == 0)
                .ok_or("invalid KTX2 metadata key")?;
            let value = &entry[end + 1..];
            let value = value.strip_suffix(&[0]).unwrap_or(value);
            match &entry[..end] {
                b"KTXswizzle" if value != b"rgba" => return Err("KTXswizzle must be rgba"),
                b"KTXorientation" if value != b"rd" => return Err("KTXorientation must be rd"),
                _ => (),
            }
            kv = kv
                .get(4 + size.checked_add(3).ok_or("metadata overflow")? / 4 * 4..)
                .ok_or("truncated metadata padding")?;
        }
        let global_offset = wide(bytes, 64)?;
        let global_data = range(bytes, global_offset, wide(bytes, 72)?)?;
        let mut spans = vec![(0, index_end), (dfd_offset, dfd_offset + dfd.len())];
        for (offset, data, alignment) in [(kvd_offset, kvd, 4), (global_offset, global_data, 8)] {
            if !data.is_empty() {
                if offset % alignment != 0 {
                    return Err("unaligned metadata");
                }
                spans.push((offset, offset + data.len()));
            }
        }
        if model == Model::Etc1s && global_data.len() < 20 {
            return Err("missing BasisLZ global data");
        }
        if model == Model::Uastc && !global_data.is_empty() {
            return Err("unexpected global data");
        }
        let mut levels = Vec::with_capacity(count);
        for i in 0..count {
            let at = 80 + i * 24;
            let offset = wide(bytes, at)?;
            let length = wide(bytes, at + 8)?;
            let uncompressed_length = wide(bytes, at + 16)?;
            if offset < index_end || length == 0 {
                return Err("invalid level range");
            }
            let data = range(bytes, offset, length)?;
            if supercompression == 0 && offset % 16 != 0 {
                return Err("unaligned UASTC level");
            }
            spans.push((offset, offset + length));
            if model == Model::Uastc {
                let expected = (width >> i)
                    .max(1)
                    .div_ceil(4)
                    .checked_mul((height >> i).max(1).div_ceil(4))
                    .and_then(|n| n.checked_mul(16))
                    .ok_or("level size overflow")?;
                if uncompressed_length != expected || (supercompression == 0 && length != expected)
                {
                    return Err("invalid UASTC level length");
                }
            } else if uncompressed_length != 0 {
                return Err("BasisLZ uncompressedByteLength must be zero");
            }
            levels.push(Level {
                bytes: data,
                uncompressed_length,
            });
        }
        spans.sort_unstable();
        if spans.windows(2).any(|s| s[0].1 > s[1].0) {
            return Err("overlapping KTX2 ranges");
        }
        Ok(Self {
            width,
            height,
            model,
            channels,
            transfer,
            supercompression,
            levels,
            global_data,
        })
    }

    pub fn require_transfer(&self, transfer: Transfer) -> Result<()> {
        if self.transfer != transfer {
            return Err("DFD transferFunction contradicts texture role");
        }
        Ok(())
    }

    /// Returns the BasisLZ payload unchanged or the unpacked UASTC blocks.
    pub fn level(&self, index: usize) -> Result<Vec<u8>> {
        let level = self.levels.get(index).ok_or("invalid level index")?;
        if self.supercompression == 2 {
            zstd::decompress(level.bytes, level.uncompressed_length)
        } else {
            Ok(level.bytes.to_vec())
        }
    }
}
