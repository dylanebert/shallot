//! Continuous convex impact, ported from Box3D distance.c (Erin Catto, MIT).
use crate::distance::{get_proxy_support, shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::math::{absf, maxf, Quat, Transform, Vec3};

#[derive(Clone, Copy)]
pub struct Sweep {
    pub local_center: Vec3,
    pub c1: Vec3,
    pub c2: Vec3,
    pub q1: Quat,
    pub q2: Quat,
}
impl Sweep {
    pub fn transform(self, time: f32) -> Transform {
        let q = self.q1.nlerp(self.q2, time);
        Transform {
            q,
            p: self.c1.lerp(self.c2, time).sub(q.rotate(self.local_center)),
        }
    }
    fn final_transform(self) -> Transform {
        Transform {
            q: self.q2,
            p: self.c2.sub(self.q2.rotate(self.local_center)),
        }
    }
}
pub struct TOIInput<'a> {
    pub proxy_a: ShapeProxy<'a>,
    pub proxy_b: ShapeProxy<'a>,
    pub sweep_a: Sweep,
    pub sweep_b: Sweep,
    pub max_fraction: f32,
}
#[derive(Clone, Copy)]
pub struct TOIOutput {
    pub state: i32,
    pub fraction: f32,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance_iterations: i32,
    pub push_back_iterations: i32,
    pub root_iterations: i32,
    pub used_fallback: bool,
}
impl TOIOutput {
    pub const ZERO: Self = Self {
        state: 0,
        fraction: 0.0,
        distance: 0.0,
        point: Vec3::ZERO,
        normal: Vec3::ZERO,
        distance_iterations: 0,
        push_back_iterations: 0,
        root_iterations: 0,
        used_fallback: false,
    };
}
#[derive(Clone, Copy, PartialEq)]
enum Separation {
    Vertices,
    Edges,
    FaceA,
    FaceB,
}
struct Function<'a> {
    a: &'a ShapeProxy<'a>,
    b: &'a ShapeProxy<'a>,
    sa: Sweep,
    sb: Sweep,
    kind: Separation,
    w1: Vec3,
    w2: Vec3,
}
fn unique(count: usize, indices: &[usize; 3]) -> usize {
    debug_assert!((1..=3).contains(&count));
    match count {
        1 => 1,
        2 => {
            if indices[0] != indices[1] {
                2
            } else {
                1
            }
        }
        3 => {
            if indices[0] != indices[1] && indices[0] != indices[2] && indices[1] != indices[2] {
                3
            } else if indices[0] == indices[1]
                && indices[0] == indices[2]
                && indices[1] == indices[2]
            {
                1
            } else {
                2
            }
        }
        _ => 0,
    }
}
impl<'a> Function<'a> {
    fn new(
        cache: SimplexCache,
        a: &'a ShapeProxy<'a>,
        sa: Sweep,
        b: &'a ShapeProxy<'a>,
        sb: Sweep,
        normal: Vec3,
        t: f32,
    ) -> Self {
        let mut f = Self {
            a,
            b,
            sa,
            sb,
            kind: Separation::Vertices,
            w1: normal,
            w2: Vec3::ZERO,
        };
        let mut ia = [cache.index_a[0], cache.index_a[1], cache.index_a[2]];
        let mut ib = [cache.index_b[0], cache.index_b[1], cache.index_b[2]];
        let ua = unique(cache.count, &ia);
        let ub = unique(cache.count, &ib);
        let xa = sa.transform(t);
        let xb = sb.transform(t);
        let dp = xb.p.sub(xa.p);
        if cache.count == 3 && (ua == 3 || ub == 3) {
            if ua == 3 {
                let v1 = a.points[ia[0]];
                let v2 = a.points[ia[1]];
                let v3 = a.points[ia[2]];
                let mut axis = v2.sub(v1).cross(v3.sub(v1)).normalize();
                let point = v1.add(v2).add(v3).scale(1.0 / 3.0);
                let delta = xb.q.rotate(b.points[ib[0]]).sub(xa.q.rotate(point)).add(dp);
                if delta.dot(xa.q.rotate(axis)) < 0.0 {
                    axis = axis.neg();
                }
                f.kind = Separation::FaceA;
                f.w1 = axis;
                f.w2 = point;
            } else {
                let v1 = b.points[ib[0]];
                let v2 = b.points[ib[1]];
                let v3 = b.points[ib[2]];
                let mut axis = v2.sub(v1).cross(v3.sub(v1)).normalize();
                let point = v1.add(v2).add(v3).scale(1.0 / 3.0);
                let delta = xa.q.rotate(a.points[ia[0]]).sub(xb.q.rotate(point)).sub(dp);
                if delta.dot(xb.q.rotate(axis)) < 0.0 {
                    axis = axis.neg();
                }
                f.kind = Separation::FaceB;
                f.w1 = axis;
                f.w2 = point;
            }
        } else if cache.count >= 2 && ua == 2 && ub == 2 {
            if ia[0] == ia[1] {
                ia[1] = ia[2];
            }
            if ib[0] == ib[1] {
                ib[1] = ib[2];
            }
            let va = a.points[ia[0]];
            let vb = b.points[ib[0]];
            let ea = a.points[ia[1]].sub(va).normalize();
            let mut eb = b.points[ib[1]].sub(vb).normalize();
            let mut axis = xa.q.rotate(ea).cross(xb.q.rotate(eb));
            let tolerance = if cache.count == 2 {
                0.05 * 0.05
            } else {
                0.005 * 0.005
            };
            if axis.length_sq() >= tolerance {
                let delta = xb.q.rotate(vb).sub(xa.q.rotate(va)).add(dp);
                if delta.dot(axis) < 0.0 {
                    axis = axis.neg();
                    eb = eb.neg();
                }
                let final_axis = sa
                    .final_transform()
                    .q
                    .rotate(ea)
                    .cross(sb.final_transform().q.rotate(eb));
                if final_axis.dot(axis) < 0.0 {
                    f.w1 = axis.normalize();
                } else {
                    f.kind = Separation::Edges;
                    f.w1 = ea;
                    f.w2 = eb;
                }
            }
        }
        f
    }
    fn axis(&self, xa: Transform, xb: Transform) -> Vec3 {
        match self.kind {
            Separation::Vertices => self.w1,
            Separation::Edges => xa.q.rotate(self.w1).cross(xb.q.rotate(self.w2)).normalize(),
            Separation::FaceA => xa.q.rotate(self.w1),
            Separation::FaceB => xb.q.rotate(self.w1),
        }
    }
    fn minimum(&self, t: f32) -> (f32, usize, usize) {
        let xa = self.sa.transform(t);
        let xb = self.sb.transform(t);
        let axis = self.axis(xa, xb);
        match self.kind {
            Separation::Vertices | Separation::Edges => {
                let ia = get_proxy_support(self.a, xa.q.inv_rotate(axis));
                let ib = get_proxy_support(self.b, xb.q.inv_rotate(axis.neg()));
                let delta =
                    xb.q.rotate(self.b.points[ib])
                        .sub(xa.q.rotate(self.a.points[ia]))
                        .add(xb.p.sub(xa.p));
                (delta.dot(axis), ia, ib)
            }
            Separation::FaceA => {
                let ib = get_proxy_support(self.b, xb.q.inv_rotate(axis).neg());
                (
                    xb.point(self.b.points[ib]).sub(xa.point(self.w2)).dot(axis),
                    0,
                    ib,
                )
            }
            Separation::FaceB => {
                let ia = get_proxy_support(self.a, xa.q.inv_rotate(axis).neg());
                (
                    xa.point(self.a.points[ia]).sub(xb.point(self.w2)).dot(axis),
                    ia,
                    0,
                )
            }
        }
    }
    fn evaluate(&self, ia: usize, ib: usize, t: f32) -> f32 {
        let xa = self.sa.transform(t);
        let xb = self.sb.transform(t);
        let axis = self.axis(xa, xb);
        match self.kind {
            Separation::Vertices | Separation::Edges => xb
                .point(self.b.points[ib])
                .sub(xa.point(self.a.points[ia]))
                .dot(axis),
            Separation::FaceA => xb.point(self.b.points[ib]).sub(xa.point(self.w2)).dot(axis),
            Separation::FaceB => xa.point(self.a.points[ia]).sub(xb.point(self.w2)).dot(axis),
        }
    }
    fn fixed_axis(&mut self, t: f32) {
        self.w1 = self.axis(self.sa.transform(t), self.sb.transform(t));
        self.w2 = Vec3::ZERO;
        self.kind = Separation::Vertices;
    }
}
pub fn time_of_impact(input: &TOIInput) -> TOIOutput {
    let mut out = TOIOutput {
        state: 0,
        fraction: -1.0,
        distance: 0.0,
        point: Vec3::ZERO,
        normal: Vec3::ZERO,
        distance_iterations: 0,
        push_back_iterations: 0,
        root_iterations: 0,
        used_fallback: false,
    };
    let mut sa = input.sweep_a;
    let mut sb = input.sweep_b;
    let origin = sa.c1;
    sa.c1 = Vec3::ZERO;
    sa.c2 = sa.c2.sub(origin);
    sb.c1 = sb.c1.sub(origin);
    sb.c2 = sb.c2.sub(origin);
    let a = &input.proxy_a;
    let b = &input.proxy_b;
    let target = maxf(0.005, a.radius + b.radius - 0.005);
    let tolerance = 0.25 * 0.005;
    let mut t1 = 0.0;
    let mut cache = SimplexCache::empty();
    loop {
        let xa = sa.transform(t1);
        let xb = sb.transform(t1);
        let distance = shape_distance(
            &DistanceInput {
                proxy_a: *a,
                proxy_b: *b,
                transform: xa.inv_mul(xb),
                use_radii: false,
            },
            &mut cache,
        );
        out.distance = distance.distance;
        let normal = xa.q.rotate(distance.normal);
        let pa = xa.point(distance.point_a);
        let pb = xa.point(distance.point_b);
        let point = pa
            .mul_add(a.radius, normal)
            .lerp(pb.mul_add(-b.radius, normal), 0.5)
            .add(origin);
        out.distance_iterations += 1;
        if distance.distance <= 0.0
            || distance.distance <= target + tolerance
            || out.distance_iterations == 25
        {
            out.state = if distance.distance <= 0.0 {
                2
            } else if distance.distance <= target + tolerance {
                3
            } else {
                1
            };
            out.fraction = if out.state == 2 { 0.0 } else { t1 };
            out.point = point;
            out.normal = normal;
            break;
        }
        let mut f = Function::new(cache, a, sa, b, sb, normal, t1);
        let mut done = false;
        let mut t2 = input.max_fraction;
        let mut pushes = 0;
        loop {
            let (mut s2, ia, ib) = f.minimum(t2);
            if s2 - target > tolerance {
                out.state = 4;
                out.fraction = input.max_fraction;
                done = true;
                break;
            }
            if s2 >= target - tolerance {
                t1 = t2;
                break;
            }
            let mut s1 = f.evaluate(ia, ib, t1);
            if s1 < target - tolerance {
                out.state = 1;
                out.fraction = t1;
                done = true;
                break;
            }
            if s1 <= target + tolerance {
                out.state = 3;
                out.fraction = t1;
                done = true;
                break;
            }
            let mut roots = 0;
            let mut a1 = t1;
            let mut a2 = t2;
            loop {
                let t = if roots & 1 != 0 {
                    a1 + (target - s1) * (a2 - a1) / (s2 - s1)
                } else {
                    0.5 * (a1 + a2)
                };
                out.root_iterations += 1;
                roots += 1;
                let s = f.evaluate(ia, ib, t);
                if absf(s - target) <= tolerance {
                    t2 = t;
                    break;
                }
                if s > target {
                    a1 = t;
                    s1 = s;
                } else {
                    a2 = t;
                    s2 = s;
                }
                if roots == 50 {
                    break;
                }
            }
            if roots == 49 && f.kind == Separation::Edges {
                t2 = input.max_fraction;
                f.fixed_axis(t1);
            }
            out.push_back_iterations += 1;
            pushes += 1;
            if pushes == a.count + b.count {
                break;
            }
        }
        if done {
            out.point = point;
            out.normal = normal;
            break;
        }
    }
    out
}
