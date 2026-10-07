//! Continuous body sweeps from Box3D solver.c (Erin Catto, MIT).
use crate::{
    bodies, body, broad,
    col::Col,
    continuous_shape::shape_time_of_impact,
    distance::Sweep,
    math::{Quat, Transform, Vec3},
    query::Shape,
    shapes, tree,
};
pub(crate) const IS_FAST: u32 = 0x40;
pub(crate) const IS_BULLET: u32 = 0x80;
const HAD_TIME_OF_IMPACT: u32 = 0x200;
pub(crate) const ENLARGE_BOUNDS: u32 = 0x800;
static mut BULLET_BODIES: *mut u32 = core::ptr::null_mut();
static BULLET_COUNT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

pub(crate) unsafe fn reserve_bullets(base: usize) {
    BULLET_BODIES = base as *mut u32;
    BULLET_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
}

pub(crate) unsafe fn add_bullet(sim: usize) {
    let index = BULLET_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    *BULLET_BODIES.add(index) = sim as u32;
}

pub(crate) fn bullet_count() -> usize {
    BULLET_COUNT.load(core::sync::atomic::Ordering::Relaxed)
}

pub(crate) unsafe fn bullet_body(index: usize) -> usize {
    *BULLET_BODIES.add(index) as usize
}

static mut ROOTS: [i32; 3] = [-1; 3];
static mut ENABLE_SLEEP: bool = true;
#[export_name = "continuousRoots"]
pub extern "C" fn roots(s: i32, k: i32, d: i32, enable_sleep: bool) {
    unsafe {
        ROOTS = [s, k, d];
        ENABLE_SLEEP = enable_sleep;
    }
}
unsafe fn sim(world_index: usize) -> Col<'static, f32> {
    Col::new(
        bodies::sim_base(world_index) as *mut f32,
        (bodies::body_cap_in_world(world_index) + 8) * body::SIM_STRIDE,
    )
}
unsafe fn fin(world_index: usize) -> Col<'static, f32> {
    Col::new(
        bodies::fin_base(world_index) as *mut f32,
        (bodies::body_cap_in_world(world_index) + 8) * body::SIM_STRIDE,
    )
}
unsafe fn sim2(world_index: usize) -> Col<'static, u32> {
    Col::new(
        bodies::sim2_base(world_index) as *mut u32,
        (bodies::body_cap_in_world(world_index) + 8) * body::SIM2_STRIDE,
    )
}
fn v(c: Col<f32>, o: usize) -> Vec3 {
    Vec3::new(c.get(o), c.get(o + 1), c.get(o + 2))
}
fn put(c: Col<f32>, o: usize, v: Vec3) {
    c.set(o, v.x);
    c.set(o + 1, v.y);
    c.set(o + 2, v.z);
}
fn q(c: Col<f32>, o: usize) -> Quat {
    Quat {
        v: v(c, o),
        s: c.get(o + 3),
    }
}
unsafe fn sweep(world_index: usize, i: usize, base: Vec3) -> Sweep {
    let f = fin(world_index);
    let s = sim(world_index);
    let s2 = Col::new(
        bodies::sim2_base(world_index) as *mut f32,
        (bodies::body_cap_in_world(world_index) + 8) * body::SIM2_STRIDE,
    );
    Sweep {
        local_center: v(f, i * body::SIM_STRIDE + body::LOCAL_CENTER),
        c1: v(s2, i * body::SIM2_STRIDE + body::S2_CENTER0).sub(base),
        c2: v(f, i * body::SIM_STRIDE + body::CENTER).sub(base),
        q1: q(s2, i * body::SIM2_STRIDE + body::S2_ROTATION0),
        q2: q(s, i * body::SIM_STRIDE + body::ROTATION),
    }
}
fn transform(s: Sweep, t: f32) -> Transform {
    let q = s.q1.nlerp(s.q2, t);
    let c = s.c1.lerp(s.c2, t);
    Transform {
        p: c.sub(q.rotate(s.local_center)),
        q,
    }
}
fn start(s: Sweep) -> Transform {
    Transform {
        p: s.c1.sub(s.q1.rotate(s.local_center)),
        q: s.q1,
    }
}
fn end(s: Sweep) -> Transform {
    Transform {
        p: s.c2.sub(s.q2.rotate(s.local_center)),
        q: s.q2,
    }
}
fn box_transform(lower: Vec3, upper: Vec3, xf: Transform) -> [f32; 6] {
    let center = xf.point(lower.add(upper).scale(0.5));
    let extent = crate::math::Mat3::from_quat(xf.q)
        .abs()
        .mul_v(upper.sub(lower).scale(0.5));
    let a = center.sub(extent);
    let b = center.add(extent);
    [a.x, a.y, a.z, b.x, b.y, b.z]
}
pub(crate) fn bounds(world_index: usize, id: usize, xf: Transform) -> [f32; 6] {
    let r = shapes::col_f(world_index);
    let o = id * shapes::SHAPE_STRIDE;
    let kind = shapes::col(world_index).get(o + shapes::S_TYPE);
    if kind == 3 {
        return unsafe { crate::shape_geometry::bounds(world_index, id, xf) };
    }
    let geom = [
        r.get(o + 48),
        r.get(o + 49),
        r.get(o + 50),
        r.get(o + 51),
        r.get(o + 52),
        r.get(o + 53),
        r.get(o + 54),
    ];
    if crate::finalize::is_convex_refit(kind) {
        return crate::finalize::convex_bounds(kind, &geom, xf);
    }
    let shape = unsafe { crate::query_abi::active_shape(world_index, id).0 };
    let (lower, upper) = match shape {
        Shape::Mesh(m) => {
            let a = crate::mesh_query::mul(m.nodes[0].lower, m.scale);
            let b = crate::mesh_query::mul(m.nodes[0].upper, m.scale);
            (crate::mesh_query::min(a, b), crate::mesh_query::max(a, b))
        }
        Shape::Height(h) => (h.lower, h.upper),
        Shape::Compound(c) => {
            let n = c.root as usize * 12;
            (
                Vec3::new(
                    f32::from_bits(c.nodes[n]),
                    f32::from_bits(c.nodes[n + 1]),
                    f32::from_bits(c.nodes[n + 2]),
                ),
                Vec3::new(
                    f32::from_bits(c.nodes[n + 3]),
                    f32::from_bits(c.nodes[n + 4]),
                    f32::from_bits(c.nodes[n + 5]),
                ),
            )
        }
        _ => unreachable!(),
    };
    box_transform(lower, upper, xf)
}
fn offset(mut b: [f32; 6], p: Vec3) -> [f32; 6] {
    b[0] += p.x;
    b[1] += p.y;
    b[2] += p.z;
    b[3] += p.x;
    b[4] += p.y;
    b[5] += p.z;
    b
}
fn union(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        crate::math::minf(a[0], b[0]),
        crate::math::minf(a[1], b[1]),
        crate::math::minf(a[2], b[2]),
        crate::math::maxf(a[3], b[3]),
        crate::math::maxf(a[4], b[4]),
        crate::math::maxf(a[5], b[5]),
    ]
}
fn lo(b: [f32; 6]) -> Vec3 {
    Vec3::new(b[0], b[1], b[2])
}
fn hi(b: [f32; 6]) -> Vec3 {
    Vec3::new(b[3], b[4], b[5])
}
unsafe fn target_sweep(world_index: usize, id: usize, base: Vec3) -> Sweep {
    let u = shapes::col(world_index);
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 1) as usize;
    let record = bodies::record(world_index, body_id);
    if record.set_index == 2 {
        return sweep(world_index, record.local_index as usize, base);
    }
    let sim = bodies::column(world_index, body_id, 1, body::SIM_STRIDE);
    let c = v(sim, body::CENTER).sub(base);
    let q = q(sim, body::ROTATION);
    Sweep {
        local_center: v(sim, body::LOCAL_CENTER),
        c1: c,
        c2: c,
        q1: q,
        q2: q,
    }
}
fn filtered(world_index: usize, a: usize, b: usize) -> bool {
    let r = shapes::col(world_index);
    let a = a * shapes::SHAPE_STRIDE;
    let b = b * shapes::SHAPE_STRIDE;
    let g = r.get(a + 42) as i32;
    if g != 0 && g == r.get(b + 42) as i32 {
        return g < 0;
    }
    ((r.get(a + 41) & r.get(b + 39)) | (r.get(a + 40) & r.get(b + 38))) == 0
        || ((r.get(b + 41) & r.get(a + 39)) | (r.get(b + 40) & r.get(a + 38))) == 0
}
/// # Safety
/// The body columns must be reserved for the active world, no other thread may write the bodies in
/// `[start, end)`, and no thread may grow memory while this runs.
pub unsafe fn sleep_enabled() -> bool {
    ENABLE_SLEEP
}
/// # Safety
/// As `finalize`, and `reserve_at` must have reserved continuous rows for every body in `[start, end)`.
pub unsafe fn bullets(world_index: usize, worker: usize, start: usize, end: usize) {
    for index in start..end {
        solve(world_index, worker, bullet_body(index));
    }
}
pub(crate) unsafe fn solve(world_index: usize, worker: usize, i: usize) {
    let u = shapes::col(world_index);
    let f = shapes::col_f(world_index);
    let s2 = sim2(world_index);
    let sf = sim(world_index);
    let ff = fin(world_index);
    let f2 = Col::new(
        bodies::sim2_base(world_index) as *mut f32,
        (bodies::body_cap_in_world(world_index) + 8) * body::SIM2_STRIDE,
    );
    let base = v(f2, i * body::SIM2_STRIDE + body::S2_CENTER0);
    let sw = sweep(world_index, i, base);
    let end = end(sw);
    let bullet = s2.atomic_get(i * body::SIM2_STRIDE + body::S2_FLAGS) & IS_BULLET != 0;
    let body_id = s2.get(i * body::SIM2_STRIDE + body::S2_BODY_ID);
    let mut fraction = 1.0;
    let mut hits = [(0u32, 0u32, 0.0f32); 8];
    let mut hit_count = 0;
    let head = bodies::record(world_index, body_id as usize).head_shape_id as u32;
    let mut id = head;
    while id != u32::MAX {
        let fast = id as usize;
        let o = fast * shapes::SHAPE_STRIDE;
        id = u.get(o + 3);
        let old = [
            f.get(o + 10),
            f.get(o + 11),
            f.get(o + 12),
            f.get(o + 13),
            f.get(o + 14),
            f.get(o + 15),
        ];
        let box2 = offset(bounds(world_index, fast, end), base);
        for n in 0..6 {
            f.set(o + 10 + n, box2[n]);
        }
        if matches!(u.get(o + shapes::S_TYPE), 2 | 4) {
            continue;
        }
        if u.get(o + 4) != u32::MAX {
            continue;
        }
        let shape = crate::query_abi::active_shape(world_index, fast).0;
        if !matches!(shape, Shape::Sphere(_) | Shape::Capsule(_) | Shape::Hull(_)) {
            continue;
        }
        let swept = union(old, box2);
        for t in 0..if bullet { 3 } else { 1 } {
            if ROOTS[t] == -1 {
                continue;
            }
            let pool = core::slice::from_raw_parts(
                broad::tree_ptr(world_index, t),
                broad::tree_cap(world_index, t) * tree::STRIDE,
            );
            let mut stack = [0; tree::STACK_SIZE];
            tree::query(
                pool,
                ROOTS[t],
                broad::tree_cap(world_index, t),
                [swept[0], swept[1], swept[2]],
                [swept[3], swept[4], swept[5]],
                u32::MAX,
                u32::MAX,
                false,
                &mut stack,
                |_, target| {
                    let target = target as usize;
                    let a = target * shapes::SHAPE_STRIDE;
                    if target == fast || u.get(a + 1) == body_id {
                        return true;
                    }
                    let sensor = u.get(a + 4) != u32::MAX;
                    if sensor
                        && (u.get(a + shapes::S_FLAGS) & shapes::SENSOR_FLAG == 0
                            || u.get(o + shapes::S_FLAGS) & shapes::SENSOR_FLAG == 0)
                        || filtered(world_index, fast, target)
                    {
                        return true;
                    }
                    let target_body = u.get(a + 1) as usize;
                    let record = bodies::record(world_index, target_body);
                    let target_flags = if record.set_index == 2 {
                        s2.atomic_get(
                            record.local_index as usize * body::SIM2_STRIDE + body::S2_FLAGS,
                        )
                    } else {
                        bodies::column(world_index, target_body, 5, body::SIM2_STRIDE)
                            .get(body::S2_FLAGS)
                            .to_bits()
                    };
                    if target_flags & IS_BULLET != 0 {
                        return true;
                    }
                    if !crate::bodies::should_collide_in_world(world_index, body_id, u.get(a + 1)) {
                        return true;
                    }
                    let target_shape = crate::query_abi::active_shape(world_index, target).0;
                    let output = shape_time_of_impact(
                        &target_shape,
                        target_sweep(world_index, target, base),
                        &shape,
                        sw,
                        fraction,
                        sensor,
                    );
                    if sensor {
                        if output.fraction <= fraction && hit_count < 8 {
                            hits[hit_count] = (target as u32, fast as u32, output.fraction);
                            hit_count += 1;
                        }
                    } else if output.fraction > 0.0 && output.fraction < fraction {
                        fraction = output.fraction;
                        s2.atomic_or(i * body::SIM2_STRIDE + body::S2_FLAGS, HAD_TIME_OF_IMPACT);
                    }
                    true
                },
            );
        }
    }
    if fraction < 1.0 {
        let rotation = sw.q1.nlerp(sw.q2, fraction);
        let c = sw.c1.lerp(sw.c2, fraction);
        body::write_sim_rotation(sf, i, rotation);
        body::write_fin_center(ff, i, base.add(c));
        body::write_fin_transform_p(ff, i, base.add(c.sub(rotation.rotate(sw.local_center))));
    }
    let rotation = q(sf, i * body::SIM_STRIDE + body::ROTATION);
    put(f2, i * body::SIM2_STRIDE + body::S2_ROTATION0, rotation.v);
    f2.set(i * body::SIM2_STRIDE + body::S2_ROTATION0 + 3, rotation.s);
    put(
        f2,
        i * body::SIM2_STRIDE + body::S2_CENTER0,
        v(ff, i * body::SIM_STRIDE + body::CENTER),
    );
    let xf = Transform {
        p: v(ff, i * body::SIM_STRIDE + body::TRANSFORM_P),
        q: rotation,
    };
    id = head;
    while id != u32::MAX {
        let o = id as usize * shapes::SHAPE_STRIDE;
        let mut b = [
            f.get(o + 10),
            f.get(o + 11),
            f.get(o + 12),
            f.get(o + 13),
            f.get(o + 14),
            f.get(o + 15),
        ];
        if fraction < 1.0 {
            b = bounds(world_index, id as usize, xf);
            for n in 0..3 {
                b[n] -= 0.02;
                b[n + 3] += 0.02;
            }
        }
        for n in 0..6 {
            f.set(o + 10 + n, b[n]);
        }
        let fat = shapes::col_f(world_index);
        let fb = o + shapes::S_FAT_AABB;
        let cached = [
            fat.get(fb),
            fat.get(fb + 1),
            fat.get(fb + 2),
            fat.get(fb + 3),
            fat.get(fb + 4),
            fat.get(fb + 5),
        ];
        let escaped = !crate::finalize::aabb_contains(&cached, &b);
        let flags = u.get(o + shapes::S_FLAGS);
        u.set(
            o + shapes::S_FLAGS,
            (flags & !shapes::ENLARGED_FLAG) | if escaped { shapes::ENLARGED_FLAG } else { 0 },
        );
        if escaped {
            let margin = f.get(o + 9);
            for n in 0..3 {
                fat.set(fb + n, b[n] - margin);
                fat.set(fb + n + 3, b[n + 3] + margin);
            }
            s2.atomic_or(i * body::SIM2_STRIDE + body::S2_FLAGS, ENLARGE_BOUNDS);
        }
        id = u.get(o + 3);
    }
    for (sensor, visitor, t) in hits.into_iter().take(hit_count) {
        if t < fraction {
            crate::arena::push_sensor_hit(world_index, worker, sensor as usize, visitor as usize);
        }
    }
}
