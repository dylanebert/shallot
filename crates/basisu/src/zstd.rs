//! Scalar port of zstddeclib.c bundled with Basis Universal 073eb6a0.
//! FSE/Huffman tables and sequence order follow the reference; bit readers use
//! checked slices instead of word lookahead and wild copies.
//! Copyright (c) 2016-present, Facebook, Inc. See ../licenses/LICENSE-zstd.
use super::{le, Result};

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let (a, b) = self.0.split_at_checked(n).ok_or("truncated Zstandard")?;
        self.0 = b;
        Ok(a)
    }
    fn number(&mut self, n: usize) -> Result<usize> {
        Ok(le(self.take(n)?))
    }
}
struct Bits<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Bits<'a> {
    fn forward(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn reverse(bytes: &'a [u8]) -> Result<Self> {
        let last = *bytes.last().ok_or("empty bitstream")?;
        if last == 0 {
            return Err("missing bitstream marker");
        }
        Ok(Self {
            bytes,
            pos: (bytes.len() - 1) * 8 + (7 - last.leading_zeros()) as usize,
        })
    }
    fn peek_at(&self, pos: usize, n: usize) -> Result<usize> {
        if n > 31
            || pos
                .checked_add(n)
                .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err("truncated bits");
        }
        let mut value = 0;
        for i in 0..n {
            value |= usize::from((self.bytes[(pos + i) / 8] >> ((pos + i) % 8)) & 1) << i;
        }
        Ok(value)
    }
    fn read(&mut self, n: usize) -> Result<usize> {
        let v = self.peek_at(self.pos, n)?;
        self.pos += n;
        Ok(v)
    }
    fn back(&mut self, n: usize) -> Result<usize> {
        self.pos = self.pos.checked_sub(n).ok_or("bitstream underflow")?;
        self.peek_at(self.pos, n)
    }
}
#[derive(Clone, Copy, Default)]
struct Entry {
    symbol: usize,
    bits: usize,
    base: usize,
}
#[derive(Clone, Default)]
struct Fse {
    log: usize,
    table: Vec<Entry>,
}
impl Fse {
    fn build(norm: &[i16], log: usize) -> Result<Self> {
        if !(5..=9).contains(&log) {
            return Err("invalid FSE table log");
        }
        let size = 1usize << log;
        if norm
            .iter()
            .map(|n| n.unsigned_abs() as usize)
            .sum::<usize>()
            != size
        {
            return Err("invalid FSE counts");
        }
        let mut table = vec![Entry::default(); size];
        let mut next = vec![0; norm.len()];
        let mut high = size;
        for (symbol, &n) in norm.iter().enumerate() {
            if n == -1 {
                high -= 1;
                table[high].symbol = symbol;
                next[symbol] = 1;
            } else if n >= 0 {
                next[symbol] = n as usize;
            } else {
                return Err("invalid FSE count");
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mut pos = 0;
        for (symbol, &n) in norm.iter().enumerate() {
            for _ in 0..n.max(0) {
                table[pos].symbol = symbol;
                pos = (pos + step) & (size - 1);
                while pos >= high {
                    pos = (pos + step) & (size - 1);
                }
            }
        }
        if pos != 0 {
            return Err("invalid FSE spread");
        }
        for entry in &mut table {
            let n = next[entry.symbol];
            next[entry.symbol] += 1;
            entry.bits = log - (usize::BITS - 1 - n.leading_zeros()) as usize;
            entry.base = (n << entry.bits) - size;
        }
        Ok(Self { log, table })
    }
    fn read(r: &mut Reader<'_>, max: usize, max_log: usize) -> Result<Self> {
        let mut b = Bits::forward(r.0);
        let log = b.read(4)? + 5;
        if log > max_log {
            return Err("FSE table log too large");
        }
        let mut remaining = (1usize << log) + 1;
        let mut threshold = 1usize << log;
        let mut bits = log + 1;
        let mut norm = Vec::new();
        while remaining > 1 && norm.len() <= max {
            if norm.last() == Some(&0) {
                loop {
                    let n = b.read(2)?;
                    if norm.len() + n > max {
                        return Err("FSE zero run too long");
                    }
                    norm.resize(norm.len() + n, 0);
                    if n != 3 {
                        break;
                    }
                }
            }
            let max_value = 2 * threshold - 1 - remaining;
            let low = b.peek_at(b.pos, bits - 1)?;
            let value = if low < max_value {
                b.read(bits - 1)?
            } else {
                let v = b.read(bits)?;
                if v >= threshold {
                    v - max_value
                } else {
                    v
                }
            };
            let count = value as i16 - 1;
            remaining = remaining
                .checked_sub(count.unsigned_abs() as usize)
                .ok_or("FSE count overflow")?;
            norm.push(count);
            if remaining < threshold && remaining > 1 {
                bits = (usize::BITS - remaining.leading_zeros()) as usize;
                threshold = 1 << (bits - 1);
            }
        }
        if remaining != 1 || norm.len() > max + 1 {
            return Err("invalid FSE normalized counts");
        }
        r.take(b.pos.div_ceil(8))?;
        Self::build(&norm, log)
    }
    fn entry(&self, state: usize) -> Result<Entry> {
        self.table.get(state).copied().ok_or("invalid FSE state")
    }
    fn mode(
        r: &mut Reader<'_>,
        mode: usize,
        previous: &mut Self,
        default: &[i16],
        log: usize,
        max: usize,
    ) -> Result<()> {
        *previous = match mode {
            0 => Self::build(default, log)?,
            1 => {
                let symbol = r.number(1)?;
                if symbol > max {
                    return Err("invalid RLE symbol");
                }
                Self {
                    log: 0,
                    table: vec![Entry {
                        symbol,
                        bits: 0,
                        base: 0,
                    }],
                }
            }
            2 => Self::read(r, max, if max == 31 { 8 } else { 9 })?,
            3 if !previous.table.is_empty() => return Ok(()),
            _ => return Err("missing repeated FSE table"),
        };
        Ok(())
    }
}

#[derive(Default)]
struct Huffman {
    log: usize,
    table: Vec<(u8, usize)>,
}
impl Huffman {
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let size = r.number(1)?;
        let mut weights = Vec::new();
        if size >= 128 {
            let count = size - 127;
            let data = r.take(count.div_ceil(2))?;
            for i in 0..count {
                weights.push(usize::from(
                    (data[i / 2] >> (if i % 2 == 0 { 4 } else { 0 })) & 15,
                ));
            }
        } else {
            let mut input = Reader(r.take(size)?);
            let fse = Fse::read(&mut input, 255, 6)?;
            let mut bits = Bits::reverse(input.0)?;
            let mut states = [bits.back(fse.log)?, bits.back(fse.log)?];
            let mut turn = 0;
            loop {
                let entry = fse.entry(states[turn])?;
                weights.push(entry.symbol);
                if weights.len() > 255 {
                    return Err("too many Huffman weights");
                }
                if entry.bits > bits.pos {
                    weights.push(fse.entry(states[turn ^ 1])?.symbol);
                    break;
                }
                states[turn] = entry.base + bits.back(entry.bits)?;
                turn ^= 1;
            }
        }
        let mut total = 0usize;
        for &w in &weights {
            if w > 11 {
                return Err("invalid Huffman weight");
            }
            total += (1 << w) >> 1;
        }
        if total == 0 {
            return Err("empty Huffman tree");
        }
        let log = (usize::BITS - total.leading_zeros()) as usize;
        if log > 11 {
            return Err("Huffman table too large");
        }
        let rest = (1usize << log) - total;
        if !rest.is_power_of_two() {
            return Err("invalid implied Huffman weight");
        }
        weights.push((usize::BITS - rest.leading_zeros()) as usize);
        if weights.len() > 256
            || weights.iter().filter(|w| **w == 1).count() < 2
            || weights.iter().filter(|w| **w == 1).count() % 2 != 0
        {
            return Err("invalid Huffman ranks");
        }
        let mut table = Vec::with_capacity(1 << log);
        for weight in 1..=log {
            for (symbol, &w) in weights.iter().enumerate() {
                if w == weight {
                    table.resize(table.len() + (1 << (w - 1)), (symbol as u8, log + 1 - w));
                }
            }
        }
        if table.len() != 1 << log {
            return Err("invalid Huffman table");
        }
        Ok(Self { log, table })
    }
    fn stream(&self, source: &[u8], count: usize, output: &mut Vec<u8>) -> Result<()> {
        if self.table.is_empty() {
            return Err("missing Huffman table");
        }
        let mut bits = Bits::reverse(source)?;
        for _ in 0..count {
            let index = if bits.pos >= self.log {
                bits.peek_at(bits.pos - self.log, self.log)?
            } else {
                bits.peek_at(0, bits.pos)? << (self.log - bits.pos)
            };
            let (symbol, n) = self.table[index];
            bits.back(n)?;
            output.push(symbol);
        }
        if bits.pos != 0 {
            return Err("extra Huffman bits");
        }
        Ok(())
    }
}

const LL_BITS: &[usize] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];
const ML_BITS: &[usize] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
const LL_BASE: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const ML_BASE: &[usize] = &[
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];
const LL_NORM: &[i16] = &[
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const ML_NORM: &[i16] = &[
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_NORM: &[i16] = &[
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

#[derive(Default)]
struct Context {
    huf: Huffman,
    ll: Fse,
    ml: Fse,
    of: Fse,
    reps: [usize; 3],
}
impl Context {
    fn literals(&mut self, r: &mut Reader<'_>) -> Result<Vec<u8>> {
        let first = *r.0.first().ok_or("missing literals")?;
        let kind = first & 3;
        let code = (first >> 2) & 3;
        let mut output = Vec::new();
        if kind < 2 {
            let (header, shift) = match code {
                0 | 2 => (1, 3),
                1 => (2, 4),
                _ => (3, 4),
            };
            let count = r.number(header)? >> shift;
            if count > 131072 {
                return Err("too many literals");
            }
            if kind == 0 {
                output.extend_from_slice(r.take(count)?);
            } else {
                output.resize(count, r.number(1)? as u8);
            }
        } else {
            let (header, n) = match code {
                0 | 1 => (3, 10),
                2 => (4, 14),
                _ => (5, 18),
            };
            // Five-byte headers must also work on wasm32.
            let data = r.take(header)?;
            let mut bits = Bits::forward(data);
            bits.read(4)?;
            let count = bits.read(n)?;
            let size = bits.read(n)?;
            if count > 131072 {
                return Err("too many literals");
            }
            let mut compressed = Reader(r.take(size)?);
            if kind == 2 {
                self.huf = Huffman::read(&mut compressed)?;
            }
            if code == 0 {
                self.huf.stream(compressed.0, count, &mut output)?;
            } else {
                if count < 6 {
                    return Err("four-stream literals too short");
                }
                let sizes = [
                    compressed.number(2)?,
                    compressed.number(2)?,
                    compressed.number(2)?,
                ];
                let part = count.div_ceil(4);
                if part * 3 > count {
                    return Err("invalid literal split");
                }
                for size in sizes {
                    self.huf.stream(compressed.take(size)?, part, &mut output)?;
                }
                self.huf
                    .stream(compressed.0, count - part * 3, &mut output)?;
            }
        }
        Ok(output)
    }
    fn block(
        &mut self,
        source: &[u8],
        output: &mut Vec<u8>,
        limit: usize,
        window: usize,
    ) -> Result<()> {
        let mut r = Reader(source);
        let literals = self.literals(&mut r)?;
        let mut lit = Reader(&literals);
        let n = r.number(1)?;
        let count = match n {
            0..=127 => n,
            128..=254 => ((n - 128) << 8) + r.number(1)?,
            _ => r.number(2)? + 0x7f00,
        };
        if count == 0 {
            if !r.0.is_empty() {
                return Err("extra sequence data");
            }
        } else {
            let modes = r.number(1)?;
            if modes & 3 != 0 {
                return Err("reserved sequence mode bits");
            }
            Fse::mode(&mut r, modes >> 6, &mut self.ll, LL_NORM, 6, 35)?;
            Fse::mode(&mut r, (modes >> 4) & 3, &mut self.of, OF_NORM, 5, 31)?;
            Fse::mode(&mut r, (modes >> 2) & 3, &mut self.ml, ML_NORM, 6, 52)?;
            let mut bits = Bits::reverse(r.0)?;
            let mut ll = bits.back(self.ll.log)?;
            let mut of = bits.back(self.of.log)?;
            let mut ml = bits.back(self.ml.log)?;
            for i in 0..count {
                let l = self.ll.entry(ll)?;
                let o = self.of.entry(of)?;
                let m = self.ml.entry(ml)?;
                let offset = if o.symbol > 1 {
                    let v = (1usize << o.symbol) - 3 + bits.back(o.symbol)?;
                    self.reps = [v, self.reps[0], self.reps[1]];
                    v
                } else if o.symbol == 0 {
                    let zero = usize::from(l.symbol == 0);
                    let v = self.reps[zero];
                    self.reps[1] = self.reps[zero ^ 1];
                    self.reps[0] = v;
                    v
                } else {
                    let v = 1 + usize::from(l.symbol == 0) + bits.back(1)?;
                    let offset = if v == 3 {
                        self.reps[0].checked_sub(1).ok_or("invalid repeat offset")?
                    } else {
                        self.reps[v]
                    };
                    if v != 1 {
                        self.reps[2] = self.reps[1];
                    }
                    self.reps[1] = self.reps[0];
                    self.reps[0] = offset;
                    offset
                };
                let match_length = ML_BASE[m.symbol] + bits.back(ML_BITS[m.symbol])?;
                let literal_length = LL_BASE[l.symbol] + bits.back(LL_BITS[l.symbol])?;
                if literal_length + match_length > limit.saturating_sub(output.len()) {
                    return Err("sequence exceeds output");
                }
                output.extend_from_slice(lit.take(literal_length)?);
                if offset == 0 || offset > output.len().min(window) {
                    return Err("invalid match offset");
                }
                for _ in 0..match_length {
                    output.push(output[output.len() - offset]);
                }
                if i + 1 < count {
                    ll = l.base + bits.back(l.bits)?;
                    ml = m.base + bits.back(m.bits)?;
                    of = o.base + bits.back(o.bits)?;
                }
            }
            if bits.pos != 0 {
                return Err("extra sequence bits");
            }
        }
        if lit.0.len() > limit.saturating_sub(output.len()) {
            return Err("literals exceed output");
        }
        output.extend_from_slice(lit.0);
        Ok(())
    }
}

pub(super) fn decompress(source: &[u8], length: usize) -> Result<Vec<u8>> {
    let mut r = Reader(source);
    if r.take(4)? != [0x28, 0xb5, 0x2f, 0xfd] {
        return Err("invalid Zstandard magic");
    }
    let flags = r.number(1)?;
    // RFC 8878 §3.1.1.1.1: bit 4 is unused and ignored; bit 3 is reserved.
    if flags & 0x08 != 0 {
        return Err("reserved Zstandard frame bits");
    }
    let single = flags & 0x20 != 0;
    let window = if single {
        length
    } else {
        let descriptor = r.number(1)?;
        let log = 10 + (descriptor >> 3);
        if log >= usize::BITS as usize {
            return Err("Zstandard window too large");
        }
        let base = 1usize << log;
        base + (base / 8) * (descriptor & 7)
    };
    let dict_size = [0, 1, 2, 4][flags & 3];
    if r.number(dict_size)? != 0 {
        return Err("Zstandard dictionaries are not supported");
    }
    let fcs_size = match flags >> 6 {
        0 => usize::from(single),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    if fcs_size != 0 {
        let data = r.take(fcs_size)?;
        let mut value = [0; 8];
        value[..fcs_size].copy_from_slice(data);
        let size = u64::from_le_bytes(value) + if fcs_size == 2 { 256 } else { 0 };
        if size != length as u64 {
            return Err("Zstandard content size mismatch");
        }
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| "Zstandard output allocation failed")?;
    let mut context = Context {
        reps: [1, 4, 8],
        ..Default::default()
    };
    loop {
        let header = r.number(3)?;
        let size = header >> 3;
        if size > 131072.min(window) {
            return Err("Zstandard block too large");
        }
        let start = output.len();
        let limit = length.min(start.saturating_add(131072.min(window)));
        match (header >> 1) & 3 {
            0 => {
                if size > limit.saturating_sub(start) {
                    return Err("raw block exceeds output");
                }
                output.extend_from_slice(r.take(size)?);
            }
            1 => {
                if size > limit.saturating_sub(start) {
                    return Err("RLE block exceeds output");
                }
                output.resize(start + size, r.number(1)? as u8);
            }
            2 => context.block(r.take(size)?, &mut output, limit, window)?,
            _ => return Err("reserved Zstandard block type"),
        }
        if header & 1 != 0 {
            break;
        }
    }
    if flags & 4 != 0 {
        let expected = r.number(4)? as u32;
        if xxhash64(&output) as u32 != expected {
            return Err("Zstandard checksum mismatch");
        }
    }
    if !r.0.is_empty() || output.len() != length {
        return Err("Zstandard length mismatch");
    }
    Ok(output)
}

// XXH64, used by the reference's optional frame checksum.
fn xxhash64(bytes: &[u8]) -> u64 {
    const P1: u64 = 11400714785074694791;
    const P2: u64 = 14029467366897019727;
    const P3: u64 = 1609587929392839161;
    const P4: u64 = 9650029242287828579;
    const P5: u64 = 2870177450012600261;
    fn round(a: u64, b: u64) -> u64 {
        a.wrapping_add(b.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    }
    fn u64le(b: &[u8]) -> u64 {
        u64::from_le_bytes(b[..8].try_into().unwrap())
    }
    let mut tail = bytes;
    let mut hash = if bytes.len() >= 32 {
        let mut v = [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)];
        while tail.len() >= 32 {
            for i in 0..4 {
                v[i] = round(v[i], u64le(&tail[i * 8..]));
            }
            tail = &tail[32..];
        }
        let mut h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        for a in v {
            h = (h ^ round(0, a)).wrapping_mul(P1).wrapping_add(P4);
        }
        h
    } else {
        P5
    };
    hash = hash.wrapping_add(bytes.len() as u64);
    while tail.len() >= 8 {
        hash = (hash ^ round(0, u64le(tail)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        tail = &tail[8..];
    }
    if tail.len() >= 4 {
        hash = (hash ^ (le(&tail[..4]) as u64).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        tail = &tail[4..];
    }
    for b in tail {
        hash = (hash ^ u64::from(*b).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(P2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(P3);
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    const RAW: &[u8] = b"\x28\xb5\x2f\xfd\x20\x03\x19\x00\x00abc";
    const RLE: &[u8] = b"\x28\xb5\x2f\xfd\x20\x20\x03\x01\x00a";

    #[test]
    fn raw_block() {
        assert_eq!(decompress(RAW, 3).unwrap(), b"abc");
    }
    #[test]
    fn rle_block() {
        assert_eq!(decompress(RLE, 32).unwrap(), vec![b'a'; 32]);
    }
    #[test]
    fn compressed_block() {
        let bytes = include_bytes!("../tests/fixtures/uastc-red.ktx2");
        let image = crate::Ktx2::parse(bytes).unwrap();
        assert_eq!(image.supercompression, 2);
        assert_red_record(&image.level(0).unwrap());
    }
    fn assert_red_record(actual: &[u8]) {
        use sha2::{Digest, Sha256};
        let record = include_str!("../tests/records/levels.tsv")
            .lines()
            .find(|line| line.starts_with("fixtures/uastc-red\t0\t"))
            .unwrap();
        let fields: Vec<_> = record.split('\t').collect();
        let digest = format!("{:x}", Sha256::digest(actual));
        assert!(
            actual.len() == fields[2].parse::<usize>().unwrap() && digest == fields[3],
            "fixtures/uastc-red level 0: expected length {} SHA-256 {}, actual length {} SHA-256 {digest}",
            fields[2], fields[3], actual.len()
        );
    }
    #[test]
    fn unused_frame_bit_matches_reference() {
        let image = crate::Ktx2::parse(include_bytes!("../tests/fixtures/uastc-red.ktx2")).unwrap();
        let mut source = image.levels[0].bytes.to_vec();
        source[4] |= 0x10;
        assert_red_record(&decompress(&source, image.levels[0].uncompressed_length).unwrap());
    }
    #[test]
    fn reserved_frame_bit_is_refused() {
        let image = crate::Ktx2::parse(include_bytes!("../tests/fixtures/uastc-red.ktx2")).unwrap();
        let mut source = image.levels[0].bytes.to_vec();
        source[4] |= 0x08;
        assert!(decompress(&source, image.levels[0].uncompressed_length).is_err());
    }
    #[test]
    fn reserved_block() {
        let mut bytes = RAW.to_vec();
        bytes[6] |= 6;
        assert!(decompress(&bytes, 3).is_err());
    }
    #[test]
    fn truncated_and_corrupt_frames() {
        for (source, size) in [(RAW, 3), (RLE, 32)] {
            for end in 0..source.len() {
                assert!(decompress(&source[..end], size).is_err());
            }
            let mut extra = source.to_vec();
            extra.push(0);
            assert!(decompress(&extra, size).is_err());
            for (at, mask) in [(0, 1), (4, 8), (5, 1), (8, 255)] {
                let mut corrupt = source.to_vec();
                corrupt[at] ^= mask;
                assert!(decompress(&corrupt, size).is_err(), "at {at}");
            }
        }
    }
    #[test]
    fn checksum() {
        assert_eq!(xxhash64(b""), 0xef46db3751d8e999);
        assert_eq!(xxhash64(b"abc"), 0x44bc2cf5ad770999);
        let mut bytes = RAW.to_vec();
        bytes[4] |= 4;
        bytes.extend_from_slice(&0xad770999u32.to_le_bytes());
        assert_eq!(decompress(&bytes, 3).unwrap(), b"abc");
        *bytes.last_mut().unwrap() ^= 1;
        assert!(decompress(&bytes, 3).is_err());
    }
    #[test]
    fn dictionary_and_window_headers() {
        let mut bytes = RAW.to_vec();
        bytes[4] |= 1;
        bytes.insert(5, 1);
        assert!(decompress(&bytes, 3).is_err());
        bytes[5] = 0;
        assert_eq!(decompress(&bytes, 3).unwrap(), b"abc");
        let mut bytes = RAW.to_vec();
        bytes[4] &= !0x20;
        bytes[5] = 0;
        assert_eq!(decompress(&bytes, 3).unwrap(), b"abc");
    }
    #[test]
    fn arbitrary_small_inputs_never_panic() {
        let mut seed = 123456789u32;
        for length in 0..256 {
            for _ in 0..16 {
                let mut bytes = vec![0; length];
                for byte in &mut bytes {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    *byte = seed as u8;
                }
                if length >= 6 {
                    bytes[..6].copy_from_slice(b"\x28\xb5\x2f\xfd\x20\x40");
                }
                let _ = decompress(&bytes, 64);
                // Exercise FSE and Huffman input parsing without a valid frame.
                let _ = Fse::read(&mut Reader(&bytes), 52, 9);
                let _ = Huffman::read(&mut Reader(&bytes));
            }
        }
    }
}
