//! Box3D aabb.c ray-AABB primitive.
use crate::math::{absf, clampf, maxf, minf, Vec3, FLT_EPSILON};

pub(crate) fn ray_cast(lower: Vec3, upper: Vec3, p1: Vec3, p2: Vec3) -> Option<(f32, f32)> {
    let d = p2.sub(p1);
    let length = d.length();
    if length < FLT_EPSILON {
        return if p1.x >= lower.x
            && p1.x <= upper.x
            && p1.y >= lower.y
            && p1.y <= upper.y
            && p1.z >= lower.z
            && p1.z <= upper.z
        {
            Some((0.0, 0.0))
        } else {
            None
        };
    }
    let dir = d.scale(1.0 / length);
    let mut t_min = 0.0;
    let mut t_max = length;
    for (c, start, lo, hi) in [
        (dir.x, p1.x, lower.x, upper.x),
        (dir.y, p1.y, lower.y, upper.y),
        (dir.z, p1.z, lower.z, upper.z),
    ] {
        if absf(c) < FLT_EPSILON {
            if start < lo || start > hi {
                return None;
            }
        } else {
            let mut t1 = (lo - start) / c;
            let mut t2 = (hi - start) / c;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            t_min = maxf(t_min, t1);
            t_max = minf(t_max, t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max < 0.0 {
        return None;
    }
    Some((
        clampf(t_min / length, 0.0, 1.0),
        clampf(t_max / length, 0.0, 1.0),
    ))
}
