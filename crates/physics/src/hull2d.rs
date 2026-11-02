//! Box3D hull.c two-dimensional hull construction and simplification (Erin Catto, MIT).
use crate::math::Vec2;
const SLOP: f32 = 0.005;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D {
    pub p: Vec2,
    pub separation: f32,
    pub original_index: i32,
}
impl Point2D {
    pub const ZERO: Self = Self {
        p: Vec2 { x: 0.0, y: 0.0 },
        separation: 0.0,
        original_index: 0,
    };
}
fn cross(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}
/// Andrew's monotone chain; output reserves twice the input length. Input sorts and welds in place.
pub fn hull(points: &mut [Point2D], output: &mut [Point2D]) -> usize {
    let mut count = points.len();
    if count <= 1 {
        if count == 1 {
            output[0] = points[0];
        }
        return count;
    }
    for i in 1..count {
        let base = points[i];
        let mut j = i;
        while j > 0
            && (points[j - 1].p.x > base.p.x
                || (points[j - 1].p.x == base.p.x && points[j - 1].p.y > base.p.y))
        {
            points[j] = points[j - 1];
            j -= 1;
        }
        points[j] = base;
    }
    let slop = 0.25 * SLOP;
    let tolerance = slop * slop;
    let mut welded = 1;
    let mut base = 0;
    for i in 1..count {
        let d = points[i].p.sub(points[base].p);
        if d.dot(d) < tolerance {
            if points[i].original_index < points[base].original_index {
                points[base] = points[i];
            }
        } else {
            points[welded] = points[i];
            base = welded;
            welded += 1;
        }
    }
    count = welded;
    if count <= 2 {
        output[..count].copy_from_slice(&points[..count]);
        return count;
    }
    let mut n = 0;
    for point in points.iter().take(count) {
        while n >= 2 {
            if cross(output[n - 2].p, output[n - 1].p, point.p) > 0.0 {
                break;
            }
            n -= 1;
        }
        output[n] = *point;
        n += 1;
    }
    let lower = n + 1;
    for i in (0..count - 1).rev() {
        while n >= lower {
            if cross(output[n - 2].p, output[n - 1].p, points[i].p) > 0.0 {
                break;
            }
            n -= 1;
        }
        output[n] = points[i];
        n += 1;
    }
    n - 1
}
/// Remove least-area vertices, preserving Box3D's tolerance-biased tie order.
pub fn simplify(points: &mut [Point2D], target: usize) -> usize {
    assert!(target >= 3);
    let mut count = points.len();
    if count <= 3 {
        return count;
    }
    let tolerance = 0.25 * SLOP * SLOP;
    loop {
        let mut min_area = f32::MAX;
        let mut min_index = 0;
        for i in 0..count {
            let area = cross(
                points[(i + count - 1) % count].p,
                points[i].p,
                points[(i + 1) % count].p,
            );
            debug_assert!(area >= -f32::EPSILON);
            if area + tolerance < min_area {
                min_area = area;
                min_index = i;
            }
        }
        if count <= 3 || (min_area > tolerance && count <= target) {
            break;
        }
        count -= 1;
        if min_index < count {
            points.copy_within(min_index + 1..count + 1, min_index);
        }
    }
    count
}

#[cfg(target_arch = "wasm32")]
mod abi {
    use super::*;
    static mut INPUT: Vec<Point2D> = Vec::new();
    static mut OUTPUT: Vec<Point2D> = Vec::new();
    static mut HEADER: [u32; 2] = [0; 2];
    #[export_name = "hull2DInputPtr"]
    pub unsafe extern "C" fn input(count: usize) -> *mut Point2D {
        let input = &mut *(&raw mut INPUT);
        input.resize(count, Point2D::ZERO);
        input.as_mut_ptr()
    }
    #[export_name = "hull2D"]
    pub unsafe extern "C" fn compute(operation: u32, count: usize, target: usize) -> *const u32 {
        let input = &mut *(&raw mut INPUT);
        let output = &mut *(&raw mut OUTPUT);
        output.resize(2 * count, Point2D::ZERO);
        let n = if operation == 0 {
            hull(&mut input[..count], output)
        } else {
            output[..count].copy_from_slice(&input[..count]);
            simplify(&mut output[..count], target)
        };
        let header = &mut *(&raw mut HEADER);
        *header = [output.as_ptr() as u32, n as u32];
        header.as_ptr()
    }
}
