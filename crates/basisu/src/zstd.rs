//! Scalar port of zstddeclib.c bundled with Basis Universal 073eb6a0.
//! FSE/Huffman tables, word lookahead and sequence order follow the reference;
//! lookahead pads the final word and match copies stay inside initialized slices.
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
    #[inline(always)]
    fn peek_at(&mut self, pos: usize, n: usize) -> Result<usize> {
        if n > 31
            || pos
                .checked_add(n)
                .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err("truncated bits");
        }
        if n == 0 {
            return Ok(0);
        }
        Ok(((lookahead(&self.bytes[pos / 8..]) >> (pos & 7)) & ((1u64 << n) - 1)) as usize)
    }
    fn read(&mut self, n: usize) -> Result<usize> {
        let v = self.peek_at(self.pos, n)?;
        self.pos += n;
        Ok(v)
    }
}
#[inline(always)]
fn lookahead(tail: &[u8]) -> u64 {
    if let Some(bytes) = tail.first_chunk::<8>() {
        u64::from_le_bytes(*bytes)
    } else {
        let mut bytes = [0u8; 8];
        bytes[..tail.len()].copy_from_slice(tail);
        u64::from_le_bytes(bytes)
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

// HUF's BIT_DStream state: a high-end lookahead word and consumed-bit count.
// Reloads are bounds checked; a short first word is padded locally.
struct DStream<'a> {
    bytes: &'a [u8],
    word: u64,
    consumed: usize,
    remaining: usize,
}
impl<'a> DStream<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        let last = *bytes.last().ok_or("empty bitstream")?;
        if last == 0 {
            return Err("missing bitstream marker");
        }
        let mut stream = Self {
            bytes,
            word: 0,
            consumed: 0,
            remaining: (bytes.len() - 1) * 8 + (7 - last.leading_zeros()) as usize,
        };
        stream.reload();
        Ok(stream)
    }
    #[inline(always)]
    fn reload(&mut self) {
        let start = self.remaining.div_ceil(8).saturating_sub(8);
        self.word = lookahead(&self.bytes[start..]);
        self.consumed = 64 - (self.remaining - start * 8);
    }
    #[inline(always)]
    fn peek(&mut self, n: usize) -> usize {
        if self.remaining < n {
            if self.consumed + n > 64 {
                self.reload();
            }
            ((self.word & ((1u64 << self.remaining) - 1)) << (n - self.remaining)) as usize
        } else {
            self.look(n)
        }
    }
    #[inline(always)]
    fn look(&mut self, n: usize) -> usize {
        if self.consumed + n > 64 {
            self.reload();
        }
        // Wasm shifts mask their count. wrapping_shr preserves that defined
        // operation for a zero-width read at bit 64; its value mask is zero.
        (self.word.wrapping_shr((64 - self.consumed - n) as u32) & ((1u64 << n) - 1)) as usize
    }
    #[inline(always)]
    fn skip(&mut self, n: usize) -> Result<()> {
        self.remaining = self.remaining.checked_sub(n).ok_or("bitstream underflow")?;
        self.consumed += n;
        Ok(())
    }
    #[inline(always)]
    fn back(&mut self, n: usize) -> Result<usize> {
        // FSE logs are <=9 and sequence extra widths are <=31, validated
        // when their tables are built. Payload bits cannot change the width.
        debug_assert!(n <= 31);
        if n > self.remaining {
            return Err("bitstream underflow");
        }
        Ok(self.take(n))
    }
    #[inline(always)]
    fn take(&mut self, n: usize) -> usize {
        debug_assert!(n <= 31 && n <= self.remaining);
        let value = self.look(n);
        self.remaining -= n;
        self.consumed += n;
        value
    }
}
struct Huffman {
    log: usize,
    // HUF_DEltX1 is two bytes: consumed bits, then the decoded byte.
    table: Box<[u16; 2048]>,
}
impl Default for Huffman {
    fn default() -> Self {
        Self {
            log: 0,
            table: Box::new([0; 2048]),
        }
    }
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
            let mut bits = DStream::new(input.0)?;
            let mut states = [bits.back(fse.log)?, bits.back(fse.log)?];
            let mut turn = 0;
            loop {
                let entry = fse.entry(states[turn])?;
                weights.push(entry.symbol);
                if weights.len() > 255 {
                    return Err("too many Huffman weights");
                }
                if entry.bits > bits.remaining {
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
        let mut table = Box::new([0; 2048]);
        let mut at = 0;
        for weight in 1..=log {
            for (symbol, &w) in weights.iter().enumerate() {
                if w == weight {
                    let end = at + (1 << (w - 1));
                    table
                        .get_mut(at..end)
                        .ok_or("invalid Huffman table")?
                        .fill(((symbol as u16) << 8) | (log + 1 - w) as u16);
                    at = end;
                }
            }
        }
        if at != 1 << log {
            return Err("invalid Huffman table");
        }
        Ok(Self { log, table })
    }
    #[inline(always)]
    fn symbol(&self, bits: &mut DStream<'_>) -> Result<u8> {
        let index = bits.peek(self.log);
        // log <= 11; masking exposes the fixed HUF table's bound to wasm.
        let entry = self.table[index & 2047];
        let n = usize::from(entry & 255);
        bits.skip(n)?;
        Ok((entry >> 8) as u8)
    }
    #[inline(always)]
    fn symbol_fast(&self, bits: &mut DStream<'_>) -> u8 {
        debug_assert!(self.log > 0 && self.log <= 11);
        debug_assert!(bits.consumed + self.log <= 64 && bits.remaining >= self.log);
        let index = bits
            .word
            .wrapping_shr((64 - bits.consumed - self.log) as u32);
        let entry = self.table[(index as usize) & ((1 << self.log) - 1) & 2047];
        let used = usize::from(entry & 255);
        bits.remaining -= used;
        bits.consumed += used;
        (entry >> 8) as u8
    }
    fn stream(&self, source: &[u8], output: &mut [u8]) -> Result<()> {
        if self.log == 0 {
            return Err("missing Huffman table");
        }
        let mut bits = DStream::new(source)?;
        for byte in output {
            *byte = self.symbol(&mut bits)?;
        }
        if bits.remaining != 0 {
            return Err("extra Huffman bits");
        }
        Ok(())
    }
    fn streams4(&self, sources: [&[u8]; 4], output: &mut [u8], part: usize) -> Result<()> {
        if self.log == 0 {
            return Err("missing Huffman table");
        }
        let mut a = DStream::new(sources[0])?;
        let mut b = DStream::new(sources[1])?;
        let mut c = DStream::new(sources[2])?;
        let mut d = DStream::new(sources[3])?;
        let (out_a, tail) = output.split_at_mut(part);
        let (out_b, tail) = tail.split_at_mut(part);
        let (out_c, out_d) = tail.split_at_mut(part);
        // HUF_decompress4X1_usingDTable_internal_body interleaves four
        // independent reverse streams into their preallocated output spans.
        let mut at = 0;
        // HUF_decompress4X1 decodes four symbols per stream between reloads.
        // Each code uses <=11 bits, so 44 real bits and four output bytes
        // bound the entire group. The checked tail handles shorter streams.
        while at + 4 <= out_d.len()
            && a.remaining >= 44
            && b.remaining >= 44
            && c.remaining >= 44
            && d.remaining >= 44
        {
            a.reload();
            b.reload();
            c.reload();
            d.reload();
            for i in at..at + 4 {
                out_a[i] = self.symbol_fast(&mut a);
                out_b[i] = self.symbol_fast(&mut b);
                out_c[i] = self.symbol_fast(&mut c);
                out_d[i] = self.symbol_fast(&mut d);
            }
            at += 4;
        }
        for i in at..out_d.len() {
            out_a[i] = self.symbol(&mut a)?;
            out_b[i] = self.symbol(&mut b)?;
            out_c[i] = self.symbol(&mut c)?;
            out_d[i] = self.symbol(&mut d)?;
        }
        for i in out_d.len()..part {
            out_a[i] = self.symbol(&mut a)?;
            out_b[i] = self.symbol(&mut b)?;
            out_c[i] = self.symbol(&mut c)?;
        }
        if a.remaining | b.remaining | c.remaining | d.remaining != 0 {
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

// ZSTD_seqSymbol packs the next-state base, both bit counts and the
// additional-bit base value into the decode table, not separate hot lookups.
#[derive(Clone, Copy, Default)]
struct SeqSymbol {
    next: u16,
    extra: u8,
    bits: u8,
    value: u32,
}
struct SeqTable {
    log: usize,
    initialized: bool,
    table: Box<[SeqSymbol; 512]>,
}
impl Default for SeqTable {
    fn default() -> Self {
        Self {
            log: 0,
            initialized: false,
            table: Box::new([SeqSymbol::default(); 512]),
        }
    }
}
impl SeqTable {
    fn mode(
        &mut self,
        r: &mut Reader<'_>,
        mode: usize,
        norm: &[i16],
        log: usize,
        values: &[usize],
        extras: &[usize],
    ) -> Result<()> {
        if mode == 3 {
            return if !self.initialized {
                Err("missing repeated FSE table")
            } else {
                Ok(())
            };
        }
        let mut fse = Fse::default();
        Fse::mode(r, mode, &mut fse, norm, log, values.len() - 1)?;
        self.log = fse.log;
        if fse.table.len() > 512 {
            return Err("FSE table too large");
        }
        for (dst, entry) in self.table.iter_mut().zip(&fse.table) {
            if entry.base + (1 << entry.bits) > fse.table.len() {
                return Err("invalid FSE next-state range");
            }
            *dst = SeqSymbol {
                next: entry.base as u16,
                bits: entry.bits as u8,
                extra: extras[entry.symbol] as u8,
                value: values[entry.symbol] as u32,
            };
        }
        self.initialized = true;
        Ok(())
    }
    #[inline(always)]
    fn entry(&self, state: usize) -> SeqSymbol {
        // Initial states have log<=9 bits, and mode validates every next-state
        // range. This exposes the fixed table bound without unchecked reads.
        self.table[state & 511]
    }
}
const OF_BITS: [usize; 32] = {
    let mut values = [0; 32];
    let mut i = 0;
    while i < 32 {
        values[i] = i;
        i += 1;
    }
    values
};
const OF_BASE: [usize; 32] = {
    let mut values = OF_BITS;
    let mut i = 2;
    while i < 32 {
        values[i] = (1usize << i) - 3;
        i += 1;
    }
    values
};
// ZSTD_decompressSequences writes through a cursor into one frame allocation.
// All bytes are initialized, but only bytes before pos are match history.
struct Output {
    bytes: Vec<u8>,
    pos: usize,
}
impl Output {
    #[inline(always)]
    fn len(&self) -> usize {
        self.pos
    }
    #[inline(always)]
    fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.bytes[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
    }
    #[inline(always)]
    fn copy_match(&mut self, offset: usize, length: usize) {
        let end = self.pos + length;
        let mut dst = self.pos;
        let mut distance = offset;
        // ZSTD_overlapCopy8 expands short-distance patterns before wildcopy.
        // The initialized frame allocation has 8 bytes of writable padding;
        // logical output bounds and match-history bounds are checked by block.
        if distance < 8 {
            for i in 0..8 {
                self.bytes[dst + i] = self.bytes[dst + i - distance];
            }
            dst += 8;
            distance *= 8usize.div_ceil(distance);
        }
        while dst < end {
            let src = dst - distance;
            let chunk: [u8; 8] = self.bytes[src..src + 8].try_into().unwrap();
            self.bytes[dst..dst + 8].copy_from_slice(&chunk);
            dst += 8;
        }
        self.pos = end;
    }
    fn resize(&mut self, length: usize, byte: u8) {
        self.bytes[self.pos..length].fill(byte);
        self.pos = length;
    }
}
#[derive(Default)]
struct Context {
    huf: Huffman,
    ll: SeqTable,
    ml: SeqTable,
    of: SeqTable,
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
            output.resize(count, 0);
            if code == 0 {
                self.huf.stream(compressed.0, &mut output)?;
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
                let sources = [
                    compressed.take(sizes[0])?,
                    compressed.take(sizes[1])?,
                    compressed.take(sizes[2])?,
                    compressed.0,
                ];
                self.huf.streams4(sources, &mut output, part)?;
            }
        }
        Ok(output)
    }
    fn block(
        &mut self,
        source: &[u8],
        output: &mut Output,
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
            self.ll
                .mode(&mut r, modes >> 6, LL_NORM, 6, LL_BASE, LL_BITS)?;
            self.of
                .mode(&mut r, (modes >> 4) & 3, OF_NORM, 5, &OF_BASE, &OF_BITS)?;
            self.ml
                .mode(&mut r, (modes >> 2) & 3, ML_NORM, 6, ML_BASE, ML_BITS)?;
            let mut bits = DStream::new(r.0)?;
            let mut ll = bits.back(self.ll.log)?;
            let mut of = bits.back(self.of.log)?;
            let mut ml = bits.back(self.ml.log)?;
            for i in 0..count {
                let l = self.ll.entry(ll);
                let o = self.of.entry(of);
                let m = self.ml.entry(ml);
                // Bound the complete sequence before consuming any fields.
                // Tables validate all widths; reads retain their word reloads
                // but need not recheck this same stream budget six times.
                let state_bits = if i + 1 < count {
                    usize::from(l.bits) + usize::from(m.bits) + usize::from(o.bits)
                } else {
                    0
                };
                let total =
                    usize::from(o.extra) + usize::from(m.extra) + usize::from(l.extra) + state_bits;
                if total > bits.remaining {
                    return Err("bitstream underflow");
                }
                let offset = if o.extra > 1 {
                    let v = o.value as usize + bits.take(usize::from(o.extra));
                    self.reps = [v, self.reps[0], self.reps[1]];
                    v
                } else if o.extra == 0 {
                    let zero = usize::from(l.value == 0);
                    let v = self.reps[zero];
                    self.reps[1] = self.reps[zero ^ 1];
                    self.reps[0] = v;
                    v
                } else {
                    let v = 1 + usize::from(l.value == 0) + bits.take(1);
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
                let match_length = m.value as usize
                    + if m.extra != 0 {
                        bits.take(usize::from(m.extra))
                    } else {
                        0
                    };
                let literal_length = l.value as usize
                    + if l.extra != 0 {
                        bits.take(usize::from(l.extra))
                    } else {
                        0
                    };
                if literal_length + match_length > limit.saturating_sub(output.len()) {
                    return Err("sequence exceeds output");
                }
                output.extend_from_slice(lit.take(literal_length)?);
                if offset == 0 || offset > output.len().min(window) {
                    return Err("invalid match offset");
                }
                output.copy_match(offset, match_length);
                if i + 1 < count {
                    ll = usize::from(l.next) + bits.take(usize::from(l.bits));
                    ml = usize::from(m.next) + bits.take(usize::from(m.bits));
                    of = usize::from(o.next) + bits.take(usize::from(o.bits));
                }
            }
            if bits.remaining != 0 {
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
    let mut bytes = Vec::new();
    let padded = length
        .checked_add(8)
        .ok_or("Zstandard output size overflow")?;
    bytes
        .try_reserve_exact(padded)
        .map_err(|_| "Zstandard output allocation failed")?;
    bytes.resize(padded, 0);
    let mut output = Output { bytes, pos: 0 };
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
        if xxhash64(&output.bytes[..output.pos]) as u32 != expected {
            return Err("Zstandard checksum mismatch");
        }
    }
    if !r.0.is_empty() || output.len() != length {
        return Err("Zstandard length mismatch");
    }
    output.bytes.truncate(output.pos);
    Ok(output.bytes)
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
    fn reverse_lookahead_matches_bits_at_every_word_and_tail_boundary() {
        let mut source = std::array::from_fn::<_, 16, _>(|i| (i * 37 + 11) as u8);
        source[15] = 0x80;
        for consumed in 0..=127 {
            for width in 0..=31 {
                let mut stream = DStream::new(&source).unwrap();
                stream.skip(consumed).unwrap();
                let remaining = 127 - consumed;
                if width > remaining {
                    assert!(stream.back(width).is_err());
                } else {
                    let start = remaining - width;
                    let expected = (0..width).fold(0usize, |v, i| {
                        v | (usize::from((source[(start + i) / 8] >> ((start + i) % 8)) & 1) << i)
                    });
                    assert_eq!(
                        stream.back(width).unwrap(),
                        expected,
                        "consumed {consumed}, width {width}"
                    );
                }
            }
        }
    }
    #[test]
    fn overlap_copy_matches_forward_bytes_with_only_eight_padding_bytes() {
        for offset in 1..=32 {
            for length in 1..=96 {
                let mut expected: Vec<u8> = (0..32).map(|i| (i * 17 + 3) as u8).collect();
                let mut output = Output {
                    bytes: vec![0; 32 + length + 8],
                    pos: 32,
                };
                output.bytes[..32].copy_from_slice(&expected);
                for _ in 0..length {
                    expected.push(expected[expected.len() - offset]);
                }
                output.copy_match(offset, length);
                assert_eq!(
                    &output.bytes[..output.pos],
                    expected,
                    "offset {offset}, length {length}"
                );
                assert_eq!(output.bytes[32 + length + 7], 0);
            }
        }
    }
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
