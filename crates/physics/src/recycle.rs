//! Angular threshold for Box3D's collide-task recycle gate.

pub(crate) const RECYCLE_ANGULAR_DISTANCE: f32 = 0.99240388;

#[cfg(test)]
mod c_parity {
    #[test]
    fn recycle_angular_distance() {
        assert_eq!(
            super::RECYCLE_ANGULAR_DISTANCE.to_bits(),
            0.99240388f32.to_bits()
        );
    }
}
