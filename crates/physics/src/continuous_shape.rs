//! Shape impact traversals from Box3D shape.c (Erin Catto, MIT).
use crate::distance::{time_of_impact, ShapeProxy, Sweep, TOIInput, TOIOutput};
use crate::math::{maxf, Mat3, Transform, Vec3};
use crate::mesh_query::{bounds_overlap, max, min};
use crate::query::Shape;

/// Geometry of the convex visitor, computed once for a target traversal.
pub struct Visitor<'a> {
    pub proxy: ShapeProxy<'a>,
    pub centroid: Vec3,
    pub min_extent: f32,
    pub sweep: Sweep,
    /// Swept bounds through the target query's initial maximum fraction, in the sweep frame.
    pub lower: Vec3,
    pub upper: Vec3,
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
    target: Sweep,
    c1: Vec3,
    c2: Vec3,
    fallback_radius: f32,
    is_sensor: bool,
    fraction: f32,
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
            target,
            fraction,
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
        let mut input = TOIInput {
            proxy_a: ShapeProxy {
                points: &points,
                count: 3,
                radius: 0.0,
            },
            proxy_b: self.visitor.proxy,
            sweep_a: self.target,
            sweep_b: self.visitor.sweep,
            max_fraction: self.fraction,
        };
        let mut output = time_of_impact(&input);
        if output.fraction > 0.0 && output.fraction < self.fraction {
            self.fraction = output.fraction;
            self.output = output;
        } else if output.fraction == 0.0 {
            let centroid = [self.visitor.centroid];
            input.proxy_b = ShapeProxy {
                points: &centroid,
                count: 1,
                radius: self.fallback_radius + 0.005,
            };
            output = time_of_impact(&input);
            if output.fraction > 0.0 && output.fraction < self.fraction {
                output.used_fallback = true;
                self.fraction = output.fraction;
                self.output = output;
            }
        }
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
                if field.clockwise {
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
fn convex_impact(shape: &Shape, target: Sweep, visitor: &Visitor, fraction: f32) -> TOIOutput {
    let mut points = [Vec3::ZERO; 2];
    let proxy = match shape {
        Shape::Sphere(s) => {
            points[0] = s.center;
            ShapeProxy {
                points: &points,
                count: 1,
                radius: s.radius,
            }
        }
        Shape::Capsule(s) => {
            points = [s.center1, s.center2];
            ShapeProxy {
                points: &points,
                count: 2,
                radius: s.radius,
            }
        }
        Shape::Hull(h) => ShapeProxy {
            points: h.points,
            count: h.vertex_count,
            radius: 0.0,
        },
        _ => unreachable!(),
    };
    time_of_impact(&TOIInput {
        proxy_a: proxy,
        proxy_b: visitor.proxy,
        sweep_a: target,
        sweep_b: visitor.sweep,
        max_fraction: fraction,
    })
}

pub fn shape_time_of_impact(
    shape: &Shape,
    target: Sweep,
    visitor: &Visitor,
    fraction: f32,
    is_sensor: bool,
) -> TOIOutput {
    match shape {
        Shape::Mesh(_) | Shape::Height(_) => {
            let (lower, upper) =
                transform_bounds(start(target).invert(), visitor.lower, visitor.upper);
            mesh_impact(
                shape,
                target,
                visitor,
                fraction,
                maxf(0.5 * visitor.min_extent, 0.005),
                is_sensor,
                lower,
                upper,
            )
        }
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(compound) => {
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
                            visitor,
                            fraction,
                            maxf(0.75 * visitor.min_extent, 0.02),
                            false,
                            lo,
                            hi,
                        )
                    }
                    _ => convex_impact(&child, sweep, visitor, fraction),
                };
                if result.fraction > 0.0 && result.fraction < fraction {
                    output = result;
                    fraction = result.fraction;
                }
                true
            });
            output
        }
        _ => convex_impact(shape, target, visitor, fraction),
    }
}
