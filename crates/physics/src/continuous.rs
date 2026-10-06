//! Continuous body sweeps from Box3D solver.c (Erin Catto, MIT).
use crate::{
    bodies, body, broad,
    col::Col,
    continuous_shape::{shape_time_of_impact, Visitor},
    distance::{ShapeProxy, Sweep},
    math::{Quat, Transform, Vec3},
    query::Shape,
    shapes, tree,
};
/// Reserved lane, hit count and up to eight (sensor, visitor) output pairs per body.
pub const STRIDE: usize = 18;
pub(crate) const IS_FAST: u32 = 0x40;
pub(crate) const IS_BULLET: u32 = 0x80;
const HAD_TIME_OF_IMPACT: u32 = 0x200;
pub(crate) const ENLARGE_BOUNDS: u32 = 0x800;
static mut BASE: usize = 0;
static mut COUNT: usize = 0;
static mut ROOTS: [i32; 3] = [-1; 3];
static mut ENABLE_SLEEP: bool = true;
/// # Safety
/// `base` must address `count` bodies' rows of `STRIDE` words in reserved scratch, set before the worker fork.
pub unsafe fn reserve_at(base: usize, count: usize) {
    BASE = base;
    COUNT = count;
}
#[export_name = "continuousPtr"]
pub extern "C" fn ptr() -> usize {
    unsafe { BASE }
}
#[export_name = "continuousRoots"]
pub extern "C" fn roots(s: i32, k: i32, d: i32, enable_sleep: bool) {
    unsafe {
        ROOTS = [s, k, d];
        ENABLE_SLEEP = enable_sleep;
    }
}
#[export_name = "sensorConsumeContinuous"]
pub unsafe extern "C" fn consume(world: usize, count: usize, bullets: bool) {
    let out = scratch();
    let sims = sim2();
    let mask = IS_FAST | IS_BULLET;
    let wanted = IS_FAST | if bullets { IS_BULLET } else { 0 };
    for i in 0..count {
        if sims.get(i * body::SIM2_STRIDE + body::S2_FLAGS) & mask != wanted {
            continue;
        }
        for n in 0..out.get(i * STRIDE + 1) as usize {
            crate::sensor::record_hit(
                world,
                out.get(i * STRIDE + 2 + n * 2) as usize,
                out.get(i * STRIDE + 3 + n * 2) as usize,
            );
        }
    }
}
unsafe fn scratch() -> Col<'static, u32> {
    Col::new(BASE as *mut u32, COUNT * STRIDE)
}
unsafe fn sim() -> Col<'static, f32> {
    Col::new(
        bodies::sim_base() as *mut f32,
        (bodies::body_cap() + 8) * 32,
    )
}
unsafe fn fin() -> Col<'static, f32> {
    Col::new(
        bodies::fin_base() as *mut f32,
        (bodies::body_cap() + 8) * 12,
    )
}
unsafe fn sim2() -> Col<'static, u32> {
    Col::new(
        bodies::sim2_base() as *mut u32,
        (bodies::body_cap() + 8) * 12,
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
unsafe fn sweep(i: usize, base: Vec3) -> Sweep {
    let f = fin();
    let s = sim();
    let s2 = Col::new(
        bodies::sim2_base() as *mut f32,
        (bodies::body_cap() + 8) * 12,
    );
    Sweep {
        local_center: v(f, i * 12 + 3),
        c1: v(s2, i * 12 + 4).sub(base),
        c2: v(f, i * 12).sub(base),
        q1: q(s2, i * 12),
        q2: q(s, i * 32 + 28),
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
pub(crate) fn bounds(id: usize, xf: Transform) -> [f32; 6] {
    let r = shapes::col_f();
    let o = id * shapes::SHAPE_STRIDE;
    let kind = shapes::col().get(o);
    if kind == 3 {
        return unsafe { crate::shape_geometry::bounds(id, xf) };
    }
    let geom = [
        r.get(o + 2),
        r.get(o + 3),
        r.get(o + 4),
        r.get(o + 5),
        r.get(o + 6),
        r.get(o + 7),
        r.get(o + 8),
    ];
    if crate::finalize::is_convex_refit(kind) {
        return crate::finalize::convex_bounds(kind, &geom, xf);
    }
    let shape = unsafe { crate::query_abi::active_shape(id).0 };
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
unsafe fn target_sweep(id: usize, base: Vec3) -> Sweep {
    let u = shapes::col();
    let f = shapes::col_f();
    let o = id * shapes::SHAPE_STRIDE;
    let awake = u.get(o + 32);
    if awake != 0 {
        return sweep(awake as usize - 1, base);
    }
    let c = v(f, o + 44).sub(base);
    let q = q(f, o + 21);
    Sweep {
        local_center: v(f, o + 47),
        c1: c,
        c2: c,
        q1: q,
        q2: q,
    }
}
fn filtered(a: usize, b: usize) -> bool {
    let r = shapes::col();
    let a = a * shapes::SHAPE_STRIDE;
    let b = b * shapes::SHAPE_STRIDE;
    let g = r.get(a + 31) as i32;
    if g != 0 && g == r.get(b + 31) as i32 {
        return g < 0;
    }
    ((r.get(a + 27) & r.get(b + 25)) | (r.get(a + 28) & r.get(b + 26))) == 0
        || ((r.get(b + 27) & r.get(a + 25)) | (r.get(b + 28) & r.get(a + 26))) == 0
}
/// # Safety
/// The body columns must be reserved for the active world, no other thread may write the bodies in
/// `[start, end)`, and no thread may grow memory while this runs.
pub unsafe fn finalize(start: usize, end: usize, enabled: bool) {
    let out = Col::new(
        bodies::fin_out_base() as *mut f32,
        (bodies::body_cap() + 8) * 2,
    );
    let s2 = sim2();
    let f2 = Col::new(
        bodies::sim2_base() as *mut f32,
        (bodies::body_cap() + 8) * 12,
    );
    for i in start..end {
        let c = scratch();
        c.set(i * STRIDE + 1, 0);
        let body = bodies::record(
            crate::regions::active(),
            s2.get(i * 12 + body::S2_BODY_ID) as usize,
        );
        let threshold = if ENABLE_SLEEP && body.flags & body::flags::ENABLE_SLEEP != 0 {
            body.sleep_threshold
        } else {
            -1.0
        };
        let awake = out.get(i * 2) > threshold;
        let flags = s2.atomic_get(i * 12 + 10) & !IS_FAST;
        s2.atomic_set(i * 12 + 10, flags);
        if enabled
            && awake
            && flags & body::flags::DYNAMIC != 0
            && out.get(i * 2 + 1) > 0.5 * f2.get(i * 12 + 7)
        {
            s2.atomic_set(i * 12 + 10, flags | IS_FAST);
            if flags & IS_BULLET == 0 {
                solve(i);
            }
        } else {
            put(f2, i * 12 + 4, v(fin(), i * 12));
            let rotation = q(sim(), i * 32 + 28);
            put(f2, i * 12, rotation.v);
            f2.set(i * 12 + 3, rotation.s);
        }
    }
}
/// # Safety
/// As `finalize`, and `reserve_at` must have reserved continuous rows for every body in `[start, end)`.
pub unsafe fn bullets(start: usize, end: usize) {
    for i in start..end {
        if sim2().atomic_get(i * 12 + 10) & (IS_FAST | IS_BULLET) == (IS_FAST | IS_BULLET) {
            solve(i);
        }
    }
}
unsafe fn solve(i: usize) {
    let u = shapes::col();
    let f = shapes::col_f();
    let s2 = sim2();
    let sf = sim();
    let ff = fin();
    let f2 = Col::new(
        bodies::sim2_base() as *mut f32,
        (bodies::body_cap() + 8) * 12,
    );
    let base = v(f2, i * 12 + 4);
    let sw = sweep(i, base);
    let end = end(sw);
    let bullet = s2.atomic_get(i * 12 + 10) & IS_BULLET != 0;
    let body_id = s2.get(i * 12 + 9);
    let mut fraction = 1.0;
    let mut hits = [(0u32, 0u32, 0.0f32); 8];
    let mut hit_count = 0;
    let head = s2.get(i * 12 + 11);
    let mut id = head;
    while id != u32::MAX {
        let fast = id as usize;
        let o = fast * shapes::SHAPE_STRIDE;
        id = u.get(o + 1);
        let old = [
            f.get(o + 34),
            f.get(o + 35),
            f.get(o + 36),
            f.get(o + 37),
            f.get(o + 38),
            f.get(o + 39),
        ];
        let box2 = offset(bounds(fast, end), base);
        for n in 0..6 {
            f.set(o + 34 + n, box2[n]);
        }
        if u.get(o + 41) != u32::MAX {
            continue;
        }
        let shape = crate::query_abi::active_shape(fast).0;
        let mut points = [Vec3::ZERO; 2];
        let (proxy, centroid, min_extent) = match &shape {
            Shape::Sphere(s) => {
                points[0] = s.center;
                (
                    ShapeProxy {
                        points: &points,
                        count: 1,
                        radius: s.radius,
                    },
                    s.center,
                    s.radius,
                )
            }
            Shape::Capsule(s) => {
                points = [s.center1, s.center2];
                (
                    ShapeProxy {
                        points: &points,
                        count: 2,
                        radius: s.radius,
                    },
                    s.center1.lerp(s.center2, 0.5),
                    s.radius,
                )
            }
            Shape::Hull(h) => (
                ShapeProxy {
                    points: h.points,
                    count: h.vertex_count,
                    radius: 0.0,
                },
                h.center,
                f.get(o + 43),
            ),
            _ => continue,
        };
        let swept = union(old, box2);
        for t in 0..if bullet { 3 } else { 1 } {
            if ROOTS[t] == -1 {
                continue;
            }
            let pool =
                core::slice::from_raw_parts(broad::tree_ptr(t), broad::tree_cap(t) * tree::STRIDE);
            let mut stack = [0; tree::STACK_SIZE];
            tree::query(
                pool,
                ROOTS[t],
                broad::tree_cap(t),
                [swept[0], swept[1], swept[2]],
                [swept[3], swept[4], swept[5]],
                u32::MAX,
                u32::MAX,
                false,
                &mut stack,
                |_, target| {
                    let target = target as usize;
                    let a = target * shapes::SHAPE_STRIDE;
                    if target == fast || u.get(a + 29) == body_id {
                        return true;
                    }
                    let sensor = u.get(a + 41) != u32::MAX;
                    if sensor
                        && (u.get(a + shapes::S_FLAGS) & shapes::SENSOR_FLAG == 0
                            || u.get(o + shapes::S_FLAGS) & shapes::SENSOR_FLAG == 0)
                        || filtered(fast, target)
                    {
                        return true;
                    }
                    let awake = u.get(a + 32);
                    let target_flags = if awake != 0 {
                        s2.atomic_get((awake as usize - 1) * 12 + 10)
                    } else {
                        u.get(a + 42)
                    };
                    if target_flags & IS_BULLET != 0 {
                        return true;
                    }
                    if !crate::bodies::should_collide(body_id, u.get(a + 29)) {
                        return true;
                    }
                    let b = union(
                        bounds(fast, start(sw)),
                        bounds(fast, transform(sw, fraction)),
                    );
                    let visitor = Visitor {
                        proxy,
                        centroid,
                        min_extent,
                        sweep: sw,
                        lower: lo(b),
                        upper: hi(b),
                    };
                    let target_shape = crate::query_abi::active_shape(target).0;
                    let output = shape_time_of_impact(
                        &target_shape,
                        target_sweep(target, base),
                        &visitor,
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
                        s2.atomic_or(i * 12 + 10, HAD_TIME_OF_IMPACT);
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
    let rotation = q(sf, i * 32 + 28);
    put(f2, i * 12, rotation.v);
    f2.set(i * 12 + 3, rotation.s);
    put(f2, i * 12 + 4, v(ff, i * 12));
    let xf = Transform {
        p: v(ff, i * 12 + 9),
        q: rotation,
    };
    id = head;
    while id != u32::MAX {
        let o = id as usize * shapes::SHAPE_STRIDE;
        let mut b = [
            f.get(o + 34),
            f.get(o + 35),
            f.get(o + 36),
            f.get(o + 37),
            f.get(o + 38),
            f.get(o + 39),
        ];
        if fraction < 1.0 {
            b = bounds(id as usize, xf);
            for n in 0..3 {
                b[n] -= 0.02;
                b[n + 3] += 0.02;
            }
        }
        for n in 0..6 {
            f.set(o + 34 + n, b[n]);
        }
        let fat = crate::fataabb::col();
        let fb = id as usize * 6;
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
            let margin = f.get(o + 40);
            for n in 0..3 {
                fat.set(fb + n, b[n] - margin);
                fat.set(fb + n + 3, b[n + 3] + margin);
            }
            s2.atomic_or(i * 12 + 10, ENLARGE_BOUNDS);
        }
        id = u.get(o + 1);
    }
    let c = scratch();
    let mut n = 0;
    for (sensor, visitor, t) in hits.into_iter().take(hit_count) {
        if t < fraction {
            c.set(i * STRIDE + 2 + n * 2, sensor);
            c.set(i * STRIDE + 3 + n * 2, visitor);
            n += 1;
        }
    }
    c.set(i * STRIDE + 1, n as u32);
}
