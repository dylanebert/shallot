// Port of Draco's compression/entropy/{ans.h,rans_symbol_decoder.h,symbol_decoding.cc}.
use crate::{
    allocation::filled,
    buffer::{Bits, Reader},
    Result,
};

struct State<'a> {
    bytes: &'a [u8],
    offset: usize,
    value: u32,
    base: u32,
}
impl<'a> State<'a> {
    fn new(bytes: &'a [u8], base: u32, max_bytes: usize) -> Result<Self> {
        let last = *bytes.last().ok_or("empty Draco rANS data")?;
        let size = usize::from(last >> 6) + 1;
        if size > max_bytes || size > bytes.len() {
            return Err("invalid Draco rANS state size");
        }
        let offset = bytes.len() - size;
        let mut value = 0u32;
        for (i, byte) in bytes[offset..].iter().enumerate() {
            value |= u32::from(*byte) << (8 * i);
        }
        value &= (1u32 << (size * 8 - 2)) - 1;
        value = value.checked_add(base).ok_or("Draco rANS state overflow")?;
        if u64::from(value) >= u64::from(base) * 256 {
            return Err("invalid Draco rANS state");
        }
        Ok(Self {
            bytes,
            offset,
            value,
            base,
        })
    }
    fn normalize(&mut self) {
        while self.value < self.base && self.offset > 0 {
            self.offset -= 1;
            self.value = self.value * 256 + u32::from(self.bytes[self.offset]);
        }
    }
}

pub(crate) struct Binary<'a> {
    state: State<'a>,
    zero: u32,
}
impl<'a> Binary<'a> {
    pub fn new(r: &mut Reader<'a>) -> Result<Self> {
        let zero = u32::from(r.byte()?);
        let size = r.count()?;
        Ok(Self {
            state: State::new(r.take(size)?, 4096, 3)?,
            zero,
        })
    }
    pub fn read(&mut self) -> Result<bool> {
        if self.state.value < self.state.base && self.state.offset > 0 {
            self.state.offset -= 1;
            self.state.value =
                self.state.value * 256 + u32::from(self.state.bytes[self.state.offset]);
        }
        let x = self.state.value;
        let p = u32::from((256 - self.zero) as u8);
        let remainder = x % 256;
        let product = (x / 256) * p;
        let value = remainder < p;
        self.state.value = if value {
            product + remainder
        } else {
            x - product - p
        };
        Ok(value)
    }
}

struct Symbols<'a> {
    state: State<'a>,
    precision_bits: u32,
    probabilities: Vec<(u32, u32)>,
    lookup: Vec<u32>,
}
impl<'a> Symbols<'a> {
    fn new(r: &mut Reader<'a>, bits: u32) -> Result<Self> {
        let precision_bits = ((3 * bits) / 2).clamp(12, 20);
        let precision = 1u32 << precision_bits;
        let count = r.count()?;
        if count / 64 > r.remaining() {
            return Err("invalid Draco rANS alphabet size");
        }
        let mut probabilities = Vec::new();
        let mut lookup = Vec::new();
        // At most 2^20 entries; reserved once rather than regrown per symbol.
        lookup
            .try_reserve_exact(precision as usize)
            .map_err(|_| "Draco entropy allocation failed")?;
        let mut cumulative = 0u32;
        let mut i = 0;
        while i < count {
            let first = r.byte()?;
            let token = first & 3;
            if token == 3 {
                let run = usize::from(first >> 2) + 1;
                if run > count - i {
                    return Err("invalid Draco rANS zero run");
                }
                probabilities
                    .try_reserve(run)
                    .map_err(|_| "Draco entropy allocation failed")?;
                probabilities.resize(probabilities.len() + run, (0, cumulative));
                i += run;
            } else {
                let mut probability = u32::from(first >> 2);
                for b in 0..token {
                    probability |= u32::from(r.byte()?) << (8 * (u32::from(b) + 1) - 2);
                }
                let next = cumulative
                    .checked_add(probability)
                    .ok_or("Draco rANS probability overflow")?;
                if next > precision {
                    return Err("invalid Draco rANS probability sum");
                }
                probabilities
                    .try_reserve(1)
                    .map_err(|_| "Draco entropy allocation failed")?;
                probabilities.push((probability, cumulative));
                lookup
                    .try_reserve(next as usize - lookup.len())
                    .map_err(|_| "Draco entropy allocation failed")?;
                lookup.resize(next as usize, i as u32);
                cumulative = next;
                i += 1;
            }
        }
        if count == 0 || cumulative != precision {
            return Err("invalid Draco rANS probability sum");
        }
        let size = usize::try_from(r.varint(64)?).map_err(|_| "Draco rANS size overflow")?;
        let state = State::new(r.take(size)?, precision * 4, 4)?;
        Ok(Self {
            state,
            precision_bits,
            probabilities,
            lookup,
        })
    }
    #[inline(always)]
    fn read(&mut self) -> u32 {
        self.state.normalize();
        let remainder = self.state.value & ((1 << self.precision_bits) - 1);
        let symbol = self.lookup[remainder as usize];
        let (probability, cumulative) = self.probabilities[symbol as usize];
        self.state.value =
            (self.state.value >> self.precision_bits) * probability + remainder - cumulative;
        symbol
    }
}

pub(crate) fn symbols(r: &mut Reader<'_>, count: usize, components: usize) -> Result<Vec<u32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if components == 0 || !count.is_multiple_of(components) {
        return Err("invalid Draco symbol components");
    }
    // `count` comes from the caller's counts; it is allocated only once the
    // symbol tables decode.
    let mut values;
    match r.byte()? {
        0 => {
            let mut decoder = Symbols::new(r, 5)?;
            values = filled(count, 0)?;
            let mut bits = Bits::new(&r.bytes[r.offset..]);
            for group in values.chunks_exact_mut(components) {
                let width = decoder.read();
                for value in group {
                    *value = bits.read(width)?;
                }
            }
            r.take(bits.consumed())?;
        }
        1 => {
            let width = u32::from(r.byte()?);
            if !(1..=18).contains(&width) {
                return Err("invalid Draco raw symbol bit length");
            }
            let mut decoder = Symbols::new(r, width)?;
            values = filled(count, 0)?;
            for value in &mut values {
                *value = decoder.read();
            }
        }
        _ => return Err("invalid Draco symbol coding scheme"),
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_binary_probability_uses_pinned_uint8_complement() {
        let mut decoder = Binary::new(&mut Reader::new(&[0, 1, 0])).unwrap();
        for _ in 0..8 {
            assert!(!decoder.read().unwrap());
        }
    }

    #[test]
    fn raw_symbols_match_pinned_encoder() {
        let bytes = include_bytes!("../tests/fixtures/sequential-compressed-raw.drc");
        let mut r = Reader::new(bytes);
        r.take(11).unwrap();
        assert_eq!(r.count().unwrap(), 3);
        assert_eq!(r.count().unwrap(), 3);
        assert_eq!(r.byte().unwrap(), 0);
        assert_eq!(r.bytes[r.offset], 1);
        assert_eq!(symbols(&mut r, 9, 1).unwrap(), [0, 2, 2, 5, 2, 2, 5, 2, 2]);
    }

    #[test]
    fn tagged_groups_are_packed_continuously_by_pinned_encoder() {
        let bytes = include_bytes!("../tests/fixtures/sequential-compressed-tagged.drc");
        let mut r = Reader::new(bytes);
        r.take(11).unwrap();
        assert_eq!(r.count().unwrap(), 64);
        assert_eq!(r.count().unwrap(), 131072);
        assert_eq!(r.byte().unwrap(), 0);
        assert_eq!(r.bytes[r.offset], 0);
        let values = symbols(&mut r, 192, 1).unwrap();
        let mut state = 42u32;
        let mut previous = 0i32;
        for value in values {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let index = ((state >> 8) & 131071) as i32;
            let delta = index - previous;
            assert_eq!(value, (delta.unsigned_abs() << 1) | u32::from(delta < 0));
            previous = index;
        }
    }

    #[test]
    fn malformed_probability_tables_and_states_are_refused() {
        for bytes in [
            vec![0, 1, 1, 16, 1, 0],    // Underfull probability table with a valid state.
            vec![0, 1, 0],              // Zero total probability.
            vec![0, 1, 7],              // Zero run exceeds alphabet.
            vec![0, 1, 1],              // Missing extra probability byte.
            vec![0, 1, 1, 64, 0],       // Valid total, empty state.
            vec![0, 1, 1, 64, 1, 0xc0], // Truncated four-byte state.
            vec![1, 0],                 // Raw bit length zero.
            vec![1, 19],                // Raw bit length above eighteen.
            vec![2],                    // Unknown scheme.
        ] {
            assert!(
                symbols(&mut Reader::new(&bytes), 1, 1).is_err(),
                "{bytes:?}"
            );
        }
        for bytes in [
            vec![],
            vec![128],
            vec![128, 0],
            vec![128, 1, 0xc0],
            vec![128, 1, 0x40],
            vec![128, 3, 0xff, 0xff, 0xbf],
        ] {
            assert!(Binary::new(&mut Reader::new(&bytes)).is_err(), "{bytes:?}");
        }
        assert!(symbols(&mut Reader::new(&[]), 0, 1).unwrap().is_empty());
        assert!(symbols(&mut Reader::new(&[]), 1, 0).is_err());
        assert!(symbols(&mut Reader::new(&[]), 3, 2).is_err());
    }
}
