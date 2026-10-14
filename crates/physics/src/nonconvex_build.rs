//! Native Box3D byte images for mesh and height-field authoring.
use crate::math::Vec3;
pub struct Image(pub Vec<u64>);
impl Image {
    pub(crate) fn new(version: u64, size: usize) -> Self {
        let mut image = Self(vec![0; size / 8]);
        image.0[0] = version;
        image.word(16, size as u32);
        image
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.0.as_ptr().cast(), self.0.len() * 8) }
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), self.0.len() * 8) }
    }
    pub(crate) fn word(&mut self, offset: usize, value: u32) {
        self.bytes_mut()[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn float(&mut self, offset: usize, value: f32) {
        self.word(offset, value.to_bits());
    }
    pub(crate) fn vector(&mut self, offset: usize, value: Vec3) {
        self.float(offset, value.x);
        self.float(offset + 4, value.y);
        self.float(offset + 8, value.z);
    }
    pub(crate) fn finish(&mut self) {
        self.0[1] = crate::hull_build::hash(self.bytes());
    }
}
pub(crate) fn align8(n: usize) -> usize {
    (n + 7) & !7
}
