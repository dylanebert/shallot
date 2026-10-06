//! Per-shape geometry operations from Box3D shape.c and geometry.c (Erin Catto, MIT).
use crate::math::{compute_quat_between_unit_vectors, Mat3, Vec3, FLT_MIN};

#[derive(Clone, Copy)]
pub(crate) struct MassData {
    pub mass: f32,
    pub center: Vec3,
    pub inertia: Mat3,
}
impl MassData {
    const ZERO: Self = Self {
        mass: 0.0,
        center: Vec3::ZERO,
        inertia: Mat3::ZERO,
    };
}
fn sphere_mass(center: Vec3, r: f32, density: f32) -> MassData {
    let volume = (((4.0f32 / 3.0) * core::f32::consts::PI * r) * r) * r;
    let mass = volume * density;
    let i = ((0.4f32 * mass) * r) * r;
    MassData {
        mass,
        center,
        inertia: Mat3 {
            cx: Vec3::new(i, 0.0, 0.0),
            cy: Vec3::new(0.0, i, 0.0),
            cz: Vec3::new(0.0, 0.0, i),
        },
    }
}
fn capsule_mass(a: Vec3, b: Vec3, r: f32, density: f32) -> MassData {
    let h = a.distance(b);
    let cm = (((core::f32::consts::PI * r) * r) * h) * density;
    let sm = sphere_mass(Vec3::ZERO, r, density);
    let x = cm * ((3.0 * r) * r + h * h) / 12.0 + sm.inertia.cx.x;
    let y = ((0.5 * cm) * r) * r + sm.inertia.cy.y;
    let shift = ((0.125 * sm.mass) * (3.0 * r + 2.0 * h)) * h;
    let inertia = Mat3 {
        cx: Vec3::new(x + shift, 0.0, 0.0),
        cy: Vec3::new(0.0, y, 0.0),
        cz: Vec3::new(0.0, 0.0, x + shift),
    };
    let rotation = if h * h > 1000.0 * FLT_MIN {
        let direction = b.sub(a);
        Mat3::from_quat(compute_quat_between_unit_vectors(
            Vec3::new(0.0, 1.0, 0.0),
            direction.scale(1.0 / direction.length()),
        ))
    } else {
        Mat3::from_quat(crate::math::Quat::IDENTITY)
    };
    MassData {
        mass: sm.mass + cm,
        center: a.add(b).scale(0.5),
        inertia: rotation.mul(inertia.mul(rotation.transpose())),
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use crate::{
        geo,
        math::{maxf, minf, Quat, Transform},
        regions, shapes,
    };
    unsafe fn vector(id: usize, lane: usize) -> Vec3 {
        let f = shapes::col_f();
        let o = id * shapes::SHAPE_STRIDE + lane;
        Vec3::new(f.get(o), f.get(o + 1), f.get(o + 2))
    }
    fn kind(id: usize) -> u32 {
        shapes::col().get(id * shapes::SHAPE_STRIDE)
    }
    fn reference(id: usize) -> usize {
        shapes::col().get(id * shapes::SHAPE_STRIDE + 8) as usize
    }
    pub(crate) unsafe fn mass(id: usize) -> MassData {
        let f = shapes::col_f();
        let o = id * shapes::SHAPE_STRIDE;
        let density = f.get(o + shapes::S_DENSITY);
        match kind(id) {
            0 => capsule_mass(vector(id, 2), vector(id, 5), f.get(o + 8), density),
            5 => sphere_mass(vector(id, 2), f.get(o + 5), density),
            3 => {
                let h = geo::hull_record(reference(id));
                MassData {
                    mass: density * h.volume,
                    center: h.center,
                    inertia: Mat3 {
                        cx: h.central_inertia.cx.scale(density),
                        cy: h.central_inertia.cy.scale(density),
                        cz: h.central_inertia.cz.scale(density),
                    },
                }
            }
            _ => MassData::ZERO,
        }
    }
    pub(crate) unsafe fn bounds(id: usize, pose: Transform) -> [f32; 6] {
        if kind(id) == 3 {
            let h = geo::hull_record(reference(id));
            let center = pose.point(h.bounds[0].add(h.bounds[1]).scale(0.5));
            let extent = Mat3::from_quat(pose.q)
                .abs()
                .mul_v(h.bounds[1].sub(h.bounds[0]).scale(0.5));
            let lo = center.sub(extent);
            let hi = center.add(extent);
            [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
        } else {
            crate::continuous::bounds(id, pose)
        }
    }
    pub(crate) unsafe fn centroid(id: usize) -> Vec3 {
        match kind(id) {
            0 => vector(id, 2).lerp(vector(id, 5), 0.5),
            5 => vector(id, 2),
            3 => geo::hull_record(reference(id)).center,
            _ => {
                let b = bounds(id, Transform::IDENTITY);
                Vec3::new(b[0] + b[3], b[1] + b[4], b[2] + b[5]).scale(0.5)
            }
        }
    }
    pub(crate) unsafe fn extent(id: usize, center: Vec3) -> (f32, Vec3) {
        let o = id * shapes::SHAPE_STRIDE;
        let f = shapes::col_f();
        match kind(id) {
            0 => {
                let r = f.get(o + 8);
                let a = vector(id, 2).sub(center).abs();
                let b = vector(id, 5).sub(center).abs();
                (
                    r,
                    Vec3::new(maxf(a.x, b.x) + r, maxf(a.y, b.y) + r, maxf(a.z, b.z) + r),
                )
            }
            5 => {
                let r = f.get(o + 5);
                let h = vector(id, 2).sub(center).abs();
                (r, h.add(Vec3::new(r, r, r)))
            }
            3 => {
                let h = geo::hull_record(reference(id));
                let view = geo::hull_view(reference(id));
                let mut maximum = Vec3::ZERO;
                for p in view.points {
                    let d = p.sub(center).abs();
                    maximum = Vec3::new(
                        maxf(maximum.x, d.x),
                        maxf(maximum.y, d.y),
                        maxf(maximum.z, d.z),
                    );
                }
                (h.inner_radius, maximum)
            }
            1 | 4 => {
                let b = bounds(id, Transform::IDENTITY);
                let lo = Vec3::new(b[0], b[1], b[2]);
                let hi = Vec3::new(b[3], b[4], b[5]);
                let p = Vec3::new(
                    if center.x - lo.x > hi.x - center.x {
                        lo.x
                    } else {
                        hi.x
                    },
                    if center.y - lo.y > hi.y - center.y {
                        lo.y
                    } else {
                        hi.y
                    },
                    if center.z - lo.z > hi.z - center.z {
                        lo.z
                    } else {
                        hi.z
                    },
                );
                (
                    minf(lo.sub(center).length(), hi.sub(center).length()),
                    p.sub(center).abs(),
                )
            }
            _ => (0.0, Vec3::ZERO),
        }
    }
    unsafe fn margin(id: usize) -> f32 {
        let o = id * shapes::SHAPE_STRIDE;
        let f = shapes::col_f();
        let radius = match kind(id) {
            0 => 0.5 * vector(id, 5).distance(vector(id, 2)) + f.get(o + 8),
            5 => f.get(o + 5),
            3 => {
                let h = geo::hull_view(reference(id));
                let mut r = 0.0;
                for p in h.points {
                    r = maxf(r, p.sub(h.center).length_sq());
                }
                r.sqrt()
            }
            _ => return 0.05,
        };
        minf(0.05, 0.125 * radius)
    }
    static mut OUTPUT: [f32; 13] = [0.0; 13];
    #[export_name = "shapeGeometryOutputPtr"]
    pub extern "C" fn output_ptr() -> *mut f32 {
        (&raw mut OUTPUT) as *mut f32
    }
    unsafe fn put_vector(lane: usize, p: Vec3) {
        OUTPUT[lane] = p.x;
        OUTPUT[lane + 1] = p.y;
        OUTPUT[lane + 2] = p.z;
    }
    #[export_name = "shapeComputeMass"]
    pub unsafe extern "C" fn compute_mass(world: usize, id: usize) {
        regions::select(world as u32);
        let m = mass(id);
        OUTPUT[0] = m.mass;
        put_vector(1, m.center);
        put_vector(4, m.inertia.cx);
        put_vector(7, m.inertia.cy);
        put_vector(10, m.inertia.cz);
    }
    #[export_name = "shapeComputeExtent"]
    pub unsafe extern "C" fn compute_extent(world: usize, id: usize, x: f32, y: f32, z: f32) {
        regions::select(world as u32);
        let (minimum, maximum) = extent(id, Vec3::new(x, y, z));
        OUTPUT[0] = minimum;
        put_vector(1, maximum);
    }
    #[export_name = "shapeComputeAABB"]
    pub unsafe extern "C" fn compute_aabb(
        world: usize,
        id: usize,
        x: f32,
        y: f32,
        z: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
        extra: f32,
    ) {
        regions::select(world as u32);
        let b = bounds(
            id,
            Transform {
                p: Vec3::new(x, y, z),
                q: Quat {
                    v: Vec3::new(qx, qy, qz),
                    s: qs,
                },
            },
        );
        for i in 0..3 {
            OUTPUT[i] = b[i] - extra;
            OUTPUT[i + 3] = b[i + 3] + extra;
        }
    }
    #[export_name = "shapeGetCentroid"]
    pub unsafe extern "C" fn get_centroid(world: usize, id: usize) {
        regions::select(world as u32);
        put_vector(0, centroid(id));
    }
    #[export_name = "shapeFinishGeometry"]
    pub unsafe extern "C" fn finish_geometry(world: usize, id: usize) {
        regions::select(world as u32);
        let c = centroid(id);
        let f = shapes::col_f();
        let o = id * shapes::SHAPE_STRIDE;
        f.set(o + 65, c.x);
        f.set(o + 66, c.y);
        f.set(o + 67, c.z);
        f.set(o + 40, margin(id));
        f.set(
            o + 43,
            if kind(id) == 3 {
                geo::hull_record(reference(id)).inner_radius
            } else {
                0.0
            },
        );
    }
    #[export_name = "shapeCanCreate"]
    pub unsafe extern "C" fn can_create(world: usize, body: usize, kind: u32) -> bool {
        regions::select(world as u32);
        crate::bodies::record(world, body).body_type == 0 || (kind != 1 && kind != 2)
    }
    #[export_name = "shapeSetGeometry"]
    pub unsafe extern "C" fn set_geometry(
        world: usize,
        id: usize,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        e: f32,
        f: f32,
        g: f32,
    ) -> u32 {
        regions::select(world as u32);
        let o = id * shapes::SHAPE_STRIDE;
        let floats = shapes::col_f();
        let mut t = kind(id);
        if t == 0 && Vec3::new(d - a, e - b, f - c).length_sq() <= 0.005f32 * 0.005f32 {
            t = 5;
            shapes::col().set(o, t);
            let center = Vec3::new(a, b, c).lerp(Vec3::new(d, e, f), 0.5);
            for (lane, value) in [center.x, center.y, center.z, g].into_iter().enumerate() {
                floats.set(o + 2 + lane, value);
            }
        } else if t == 4 {
            for (lane, value) in [a, b, c].into_iter().enumerate() {
                let sign = if value >= 0.0 { 1.0 } else { -1.0 };
                floats.set(o + 2 + lane, sign * maxf(value.abs(), 0.01));
            }
        } else {
            for (lane, value) in [a, b, c, d, e, f, g].into_iter().enumerate() {
                floats.set(o + 2 + lane, value);
            }
        }
        t
    }
}
#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::{bounds, extent, mass};

#[cfg(test)]
#[path = "shape_geometry_tests.rs"]
mod tests;
