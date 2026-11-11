//! Basis Universal 073eb6a0's LSB-first bit reader and canonical Huffman tables.
//! Altered source: checked word lookahead, with local zero padding at the tail.
//! The reference's 10-bit Huffman lookup precedes canonical long-code decoding.
//! Copyright 2016–2026 Binomial LLC. See ../licenses/{LICENSE,NOTICE}-basis-universal.
use crate::Result;

pub struct Bits<'a> {
    bytes: &'a [u8],
    position: usize,
    word: u64,
    available: usize,
}
impl<'a> Bits<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            word: 0,
            available: 0,
        }
    }
    #[inline(always)]
    pub fn get(&mut self, count: usize) -> Result<usize> {
        if count > 32 {
            return Err("invalid BasisLZ bit width");
        }
        let value = (self.lookahead(count) & ((1u64 << count) - 1)) as usize;
        self.consume(count)?;
        Ok(value)
    }
    #[inline(always)]
    fn lookahead(&mut self, count: usize) -> u64 {
        // bitwise_decoder retains and shifts its bit buffer between symbols.
        // Refill only when the requested width exceeds its available bits.
        if self.available < count {
            let start = self.position / 8;
            let tail = &self.bytes[start..];
            self.word = if let Some(bytes) = tail.first_chunk::<8>() {
                u64::from_le_bytes(*bytes)
            } else {
                let mut bytes = [0u8; 8];
                bytes[..tail.len()].copy_from_slice(tail);
                u64::from_le_bytes(bytes)
            };
            let shift = self.position & 7;
            self.word >>= shift;
            self.available = 64 - shift;
        }
        self.word
    }
    #[inline(always)]
    fn consume(&mut self, count: usize) -> Result<()> {
        let next = self.position.checked_add(count).ok_or("bit overflow")?;
        if next > self.bytes.len() * 8 {
            return Err("truncated BasisLZ bits");
        }
        debug_assert!(count <= self.available);
        self.word >>= count;
        self.available -= count;
        self.position = next;
        Ok(())
    }
    pub fn vlc(&mut self, chunk: usize) -> Result<usize> {
        let mut value = 0u32;
        for offset in (0..32).step_by(chunk) {
            let s = self.get(chunk + 1)? as u32;
            let part = s & ((1 << chunk) - 1);
            if part > (u32::MAX >> offset) {
                return Err("BasisLZ VLC overflow");
            }
            value |= part << offset;
            if s & (1 << chunk) == 0 {
                return Ok(value as usize);
            }
        }
        Err("BasisLZ VLC overflow")
    }
    pub fn table(&mut self) -> Result<Huffman> {
        let total = self.get(14)?;
        if total == 0 {
            return Huffman::new(&[]);
        }
        let count = self.get(5)?;
        if !(1..=21).contains(&count) {
            return Err("invalid Huffman code length count");
        }
        let order = [
            17, 18, 19, 20, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 16,
        ];
        let mut sizes = [0; 21];
        for &i in &order[..count] {
            sizes[i] = self.get(3)?;
        }
        let lengths = Huffman::new(&sizes)?;
        let mut codes = Vec::with_capacity(total);
        while codes.len() < total {
            let c = lengths.decode(self)?;
            if c <= 16 {
                codes.push(c);
                continue;
            }
            let (extra, min) = match c {
                17 => (3, 3),
                18 => (7, 11),
                19 => (2, 3),
                20 => (7, 7),
                _ => return Err("invalid Huffman length"),
            };
            let n = self.get(extra)? + min;
            let value = if c <= 18 {
                0
            } else {
                *codes
                    .last()
                    .filter(|&&v| v != 0)
                    .ok_or("invalid Huffman repeat")?
            };
            if codes.len() + n > total {
                return Err("Huffman length run overflow");
            }
            codes.resize(codes.len() + n, value);
        }
        Huffman::new(&codes)
    }
}

pub struct Huffman {
    // Canonical codes are read most-significant bit first from the LSB bitstream.
    counts: [usize; 17],
    first: [usize; 17],
    offsets: [usize; 17],
    symbols: Vec<usize>,
    lookup: Box<[u32; 1024]>,
}
impl Huffman {
    fn new(sizes: &[usize]) -> Result<Self> {
        let mut counts = [0; 17];
        for &size in sizes {
            if size > 16 {
                return Err("invalid Huffman code size");
            }
            if size != 0 {
                counts[size] += 1;
            }
        }
        let used = counts.iter().sum::<usize>();
        let mut first = [0; 17];
        let mut offsets = [0; 17];
        for i in 1..=16 {
            first[i] = (first[i - 1] + counts[i - 1]) * 2;
            offsets[i] = offsets[i - 1] + counts[i - 1];
            if first[i] + counts[i] > 1 << i {
                return Err("oversubscribed Huffman table");
            }
        }
        if used != 0 && used != 1 && first[16] + counts[16] != 65536 {
            return Err("incomplete Huffman table");
        }
        let mut symbols = vec![0; used];
        let mut next = offsets;
        for (sym, &size) in sizes.iter().enumerate() {
            if size != 0 {
                symbols[next[size]] = sym;
                next[size] += 1;
            }
        }
        // Basis Universal's cHuffmanFastLookupBits is 10. Low bits contain
        // the symbol; high bits contain the number of consumed input bits.
        let mut lookup = Box::new([0; 1024]);
        for size in 1..=10 {
            for rank in 0..counts[size] {
                let code = first[size] + rank;
                let reversed = code.reverse_bits() >> (usize::BITS as usize - size);
                let entry = symbols[offsets[size] + rank] as u32 | ((size as u32) << 16);
                for suffix in 0..(1 << (10 - size)) {
                    lookup[reversed | (suffix << size)] = entry;
                }
            }
        }
        Ok(Self {
            counts,
            first,
            offsets,
            symbols,
            lookup,
        })
    }
    pub fn require(&self) -> Result<()> {
        if self.symbols.is_empty() {
            Err("empty BasisLZ Huffman table")
        } else {
            Ok(())
        }
    }
    #[inline(always)]
    pub fn decode(&self, bits: &mut Bits<'_>) -> Result<usize> {
        let entry = self.lookup[(bits.lookahead(10) & 1023) as usize];
        let size = (entry >> 16) as usize;
        if size != 0 {
            bits.consume(size)?;
            return Ok((entry & 65535) as usize);
        }
        self.require()?;
        let mut code = 0;
        for size in 1..=16 {
            code = code * 2 + bits.get(1)?;
            if code >= self.first[size] && code - self.first[size] < self.counts[size] {
                return Ok(self.symbols[self.offsets[size] + code - self.first[size]]);
            }
        }
        Err("invalid BasisLZ Huffman code")
    }
}
