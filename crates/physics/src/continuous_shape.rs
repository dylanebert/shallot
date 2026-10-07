//! Shape impact traversals from Box3D shape.c (Erin Catto, MIT).
use crate::distance::{time_of_impact, ShapeProxy, Sweep, TOIInput, TOIOutput};
use crate::math::{maxf, Mat3, Transform, Vec3};
use crate::mesh_query::{bounds_overlap, max, min};
use crate::query::Shape;

/// Geometry of the convex visitor, computed once for a target traversal.
struct Visitor<'a> {
    pub proxy: ShapeProxy<'a>,
    pub centroid: Vec3,
    pub min_extent: f32,
    pub sweep: Sweep,
    /// Swept bounds through the target query's initial maximum fraction, in the sweep frame.
    pub lower: Vec3,
    pub upper: Vec3,
}

fn make_visitor<'a>(shape: &'a Shape, sweep: Sweep, fraction: f32) -> Visitor<'a> {
    let (centroid, min_extent) = match shape {
        Shape::Sphere(s) => (s.center, s.radius),
        Shape::Capsule(s) => (s.center1.lerp(s.center2, 0.5), s.radius),
        Shape::Hull(h) => (h.center, h.inner_radius),
        _ => unreachable!(),
    };
    let bounds = |xf: Transform| match shape {
        Shape::Sphere(s) => {
            let center = xf.point(s.center);
            let r = Vec3::new(s.radius, s.radius, s.radius);
            (center.sub(r), center.add(r))
        }
        Shape::Capsule(s) => {
            let a = xf.point(s.center1);
            let b = xf.point(s.center2);
            let r = Vec3::new(s.radius, s.radius, s.radius);
            (min(a, b).sub(r), max(a, b).add(r))
        }
        Shape::Hull(h) => transform_bounds(xf, h.bounds[0], h.bounds[1]),
        _ => unreachable!(),
    };
    let (lo1, hi1) = bounds(start(sweep));
    let (lo2, hi2) = bounds(sweep.transform(fraction));
    Visitor {
        proxy: shape_proxy(shape),
        centroid,
        min_extent,
        sweep,
        lower: min(lo1, lo2),
        upper: max(hi1, hi2),
    }
}

fn transform_bounds(xf: Transform, lower: Vec3, upper: Vec3) -> (Vec3, Vec3) {
    let center = xf.point(lower.add(upper).scale(0.5));
    let extent = Mat3::from_quat(xf.q)
        .abs()
        .mul_v(upper.sub(lower).scale(0.5));
    (center.sub(extent), center.add(extent))
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
struct MeshImpact<'a> {
    visitor: &'a Visitor<'a>,
    input: TOIInput<'a>,
    c1: Vec3,
    c2: Vec3,
    fallback_radius: f32,
    is_sensor: bool,
    output: TOIOutput,
}
impl<'a> MeshImpact<'a> {
    fn new(
        visitor: &'a Visitor<'a>,
        target: Sweep,
        fraction: f32,
        fallback_radius: f32,
        is_sensor: bool,
    ) -> Self {
        let xf = start(target);
        Self {
            visitor,
            input: TOIInput {
                proxy_a: ShapeProxy {
                    points: &[],
                    count: 3,
                    radius: 0.0,
                },
                proxy_b: visitor.proxy,
                sweep_a: target,
                sweep_b: visitor.sweep,
                max_fraction: fraction,
            },
            fallback_radius,
            is_sensor,
            c1: xf.inv_point(start(visitor.sweep).point(visitor.centroid)),
            c2: xf.inv_point(end(visitor.sweep).point(visitor.centroid)),
            output: TOIOutput::ZERO,
        }
    }
    fn triangle(&mut self, points: [Vec3; 3]) -> bool {
        let [a, b, c] = points;
        let n = b.sub(a).cross(c.sub(a)).normalize();
        let offset1 = n.dot(self.c1.sub(a));
        let offset2 = n.dot(self.c2.sub(a));
        if offset1 < 0.0
            || (!self.is_sensor
                && offset1 - offset2 < self.fallback_radius
                && offset2 > self.fallback_radius)
        {
            return true;
        }
        // The triangle loan is used only by the synchronous TOI calls below and cleared before
        // points leaves this frame. No traversal callback can observe this temporary reference.
        self.input.proxy_a.points = unsafe { core::slice::from_raw_parts(points.as_ptr(), 3) };
        let mut output = time_of_impact(&self.input);
        if output.fraction > 0.0 && output.fraction < self.input.max_fraction {
            self.input.max_fraction = output.fraction;
            self.output = output;
        } else if output.fraction == 0.0 {
            let mut fallback_input = self.input;
            fallback_input.proxy_b = ShapeProxy {
                points: core::slice::from_ref(&self.visitor.centroid),
                count: 1,
                radius: self.fallback_radius + 0.005,
            };
            output = time_of_impact(&fallback_input);
            if output.fraction > 0.0 && output.fraction < self.input.max_fraction {
                output.used_fallback = true;
                self.input.max_fraction = output.fraction;
                self.output = output;
            }
        }
        self.input.proxy_a.points = &[];
        true
    }
}
fn mesh_impact(
    shape: &Shape,
    target: Sweep,
    visitor: &Visitor,
    fraction: f32,
    radius: f32,
    is_sensor: bool,
    lower: Vec3,
    upper: Vec3,
) -> TOIOutput {
    let mut context = MeshImpact::new(visitor, target, fraction, radius, is_sensor);
    match shape {
        Shape::Mesh(mesh) => {
            let flip = mesh.scale.x * mesh.scale.y * mesh.scale.z <= 0.0;
            crate::mesh_query::visit_triangles(*mesh, lower, upper, flip, |_, triangle| {
                context.triangle(triangle)
            });
        }
        Shape::Height(field) => field.visit_cells(lower, upper, |_, [a, b, c, d]| {
            let lo = min(min(a, b), min(c, d));
            let hi = max(max(a, b), max(c, d));
            if bounds_overlap(lower, upper, lo, hi) {
                if field.clockwise() {
                    context.triangle([a, b, c]);
                    context.triangle([d, c, b]);
                } else {
                    context.triangle([a, c, b]);
                    context.triangle([d, b, c]);
                }
            }
            true
        }),
        _ => unreachable!(),
    }
    context.output
}
fn shape_proxy<'a>(shape: &'a Shape) -> ShapeProxy<'a> {
    match shape {
        Shape::Sphere(s) => ShapeProxy {
            points: core::slice::from_ref(&s.center),
            count: 1,
            radius: s.radius,
        },
        Shape::Capsule(s) => ShapeProxy {
            points: s.points(),
            count: 2,
            radius: s.radius,
        },
        Shape::Hull(h) => ShapeProxy {
            points: h.points,
            count: h.vertex_count,
            radius: 0.0,
        },
        _ => unreachable!(),
    }
}

fn convex_impact(
    shape: &Shape,
    target: Sweep,
    proxy_b: ShapeProxy,
    sweep_b: Sweep,
    fraction: f32,
) -> TOIOutput {
    time_of_impact(&TOIInput {
        proxy_a: shape_proxy(shape),
        proxy_b,
        sweep_a: target,
        sweep_b,
        max_fraction: fraction,
    })
}

pub fn shape_time_of_impact(
    shape: &Shape,
    target: Sweep,
    shape_b: &Shape,
    sweep_b: Sweep,
    fraction: f32,
    is_sensor: bool,
) -> TOIOutput {
    match shape {
        Shape::Mesh(_) | Shape::Height(_) => {
            let visitor = make_visitor(shape_b, sweep_b, fraction);
            let (lower, upper) =
                transform_bounds(start(target).invert(), visitor.lower, visitor.upper);
            mesh_impact(
                shape,
                target,
                &visitor,
                fraction,
                maxf(0.5 * visitor.min_extent, 0.005),
                is_sensor,
                lower,
                upper,
            )
        }
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(compound) => {
            let visitor = make_visitor(shape_b, sweep_b, fraction);
            let xf = Transform {
                p: target.c1,
                q: target.q1,
            };
            let (lower, upper) = transform_bounds(xf.invert(), visitor.lower, visitor.upper);
            let mut output = TOIOutput::ZERO;
            let mut fraction = fraction;
            crate::compound_query::query(*compound, lower, upper, |_, index| {
                let (child, child_xf, _) = crate::compound_query::child(*compound, index as usize);
                let world_xf = xf.mul(child_xf);
                let sweep = Sweep {
                    local_center: Vec3::ZERO,
                    c1: world_xf.p,
                    c2: world_xf.p,
                    q1: world_xf.q,
                    q2: world_xf.q,
                };
                let result = match child {
                    Shape::Mesh(_) => {
                        let (lo, hi) = transform_bounds(child_xf.invert(), lower, upper);
                        mesh_impact(
                            &child,
                            sweep,
                            &visitor,
                            fraction,
                            maxf(0.75 * visitor.min_extent, 0.02),
                            false,
                            lo,
                            hi,
                        )
                    }
                    _ => convex_impact(&child, sweep, visitor.proxy, visitor.sweep, fraction),
                };
                if result.fraction > 0.0 && result.fraction < fraction {
                    output = result;
                    fraction = result.fraction;
                }
                true
            });
            output
        }
        _ => convex_impact(shape, target, shape_proxy(shape_b), sweep_b, fraction),
    }
}
