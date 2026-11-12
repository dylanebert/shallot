use crate::Result;

pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
    pub offset: usize,
}
impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    pub fn take(&mut self, size: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(size)
            .ok_or("Draco range overflow")?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or("truncated Draco stream")?;
        self.offset = end;
        Ok(bytes)
    }
    pub fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn varint(&mut self, width: u32) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..width).step_by(7) {
            let byte = self.byte()?;
            let part = u64::from(byte & 127);
            value |= part.wrapping_shl(shift);
            if byte & 128 == 0 {
                return Ok(if width == 32 {
                    u64::from(value as u32)
                } else {
                    value
                });
            }
        }
        Err("Draco varint overflow")
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn count(&mut self) -> Result<usize> {
        usize::try_from(self.varint(32)?).map_err(|_| "Draco count overflow")
    }
}

pub(crate) struct Bits<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Bits<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    pub fn read(&mut self, count: u32) -> Result<u32> {
        if count > 32
            || self
                .offset
                .checked_add(count as usize)
                .is_none_or(|end| end > self.bytes.len().saturating_mul(8))
        {
            return Err("truncated Draco bits");
        }
        let mut value = 0;
        for bit in 0..count {
            value |= u32::from((self.bytes[self.offset / 8] >> (self.offset % 8)) & 1) << bit;
            self.offset += 1;
        }
        Ok(value)
    }
    pub fn consumed(&self) -> usize {
        self.offset.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_fields_cross_bytes_without_alignment() {
        let mut bits = Bits::new(&[0b1011_0010, 0b0110_1011, 0xff, 0x31, 0xab]);
        assert_eq!(bits.read(3).unwrap(), 2);
        assert_eq!(bits.read(7).unwrap(), 118);
        assert_eq!(bits.read(0).unwrap(), 0);
        assert_eq!(bits.read(6).unwrap(), 26);
        assert_eq!(bits.consumed(), 2);
        assert!(bits.read(33).is_err());
        assert_eq!(bits.read(24).unwrap(), 0xab31ff);
        assert!(bits.read(1).is_err());
    }

    #[test]
    fn ranges_and_varints_are_bounded() {
        let mut r = Reader::new(&[0x80, 0x01]);
        assert_eq!(r.varint(32).unwrap(), 128);
        assert!(r.take(usize::MAX).is_err());
        assert_eq!(r.offset, 2);
        for width in [32, 64] {
            let bytes = [0x80; 11];
            assert!(Reader::new(&bytes).varint(width).is_err());
            for size in 0..if width == 32 { 5 } else { 10 } {
                assert!(Reader::new(&bytes[..size]).varint(width).is_err());
            }
        }
        // DecodeVarintUnsigned shifts in the destination type, retaining only its bits.
        assert_eq!(
            Reader::new(&[0xff; 4].into_iter().chain([0x7f]).collect::<Vec<_>>())
                .varint(32)
                .unwrap(),
            u64::from(u32::MAX)
        );
    }
}
