//! Body queries, in attachment order, using the caller-supplied placement.
use crate::distance::{shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::manifold::Capsule;
use crate::math::Vec3;
use crate::query::{self, PlaneResult, RayCastInput, ShapeCastInput};
use crate::shapes::{NULL_SHAPE, SHAPE_STRIDE, S_NEXT};
use crate::{query_abi, world_query};

/// Operations: ray, shape cast, overlap, closest point, collide mover.
#[export_name = "bodyQuery"]
pub extern "C" fn run(world: usize, operation: u32, head: u32, capacity: usize) {
    run_in_world(world, operation, head, capacity)
}

pub extern "C" fn run_in_world(world: usize, operation: u32, head: u32, capacity: usize) {
    if operation == 4 && capacity == 0 {
        return;
    }
    unsafe {
        let header = world_query::HEADER;
        let (r, transform, proxy) = query_abi::input();
        let translation = Vec3::new(r[9], r[10], r[11]);
        let mut fraction = r[12];
        let result = &raw mut world_query::RESULT as *mut u32;
        result.write(NULL_SHAPE);
        let mut closest = f32::MAX;
        let mut point = transform.p;
        let mut id = head;
        let mut count = 0;
        while id != NULL_SHAPE {
            let shape_id = id as usize;
            id = crate::shapes::col(world).get(shape_id * SHAPE_STRIDE + S_NEXT);
            if operation != 3 && !world_query::accepts(world, shape_id, &header) {
                continue;
            }
            let (shape, materials) = query_abi::shape(world, shape_id);
            if operation == 2 {
                if query::overlap_shape(&shape, transform, proxy) {
                    result.write(shape_id as u32);
                    break;
                }
                continue;
            }
            if operation == 3 {
                let convex = match &shape {
                    query::Shape::Sphere(s) => ShapeProxy {
                        points: core::slice::from_ref(&s.center),
                        count: 1,
                        radius: s.radius,
                    },
                    query::Shape::Capsule(s) => ShapeProxy {
                        points: s.points(),
                        count: 2,
                        radius: s.radius,
                    },
                    query::Shape::Hull(h) => ShapeProxy {
                        points: h.points,
                        count: h.vertex_count,
                        radius: 0.0,
                    },
                    _ => continue,
                };
                let out = shape_distance(
                    &DistanceInput {
                        proxy_a: proxy,
                        proxy_b: convex,
                        transform,
                        use_radii: false,
                    },
                    &mut SimplexCache::empty(),
                );
                if out.distance < closest {
                    closest = out.distance;
                    point = out.point_b;
                }
                continue;
            }
            if operation == 4 {
                if !matches!(
                    shape,
                    query::Shape::Sphere(_) | query::Shape::Capsule(_) | query::Shape::Hull(_)
                ) {
                    continue;
                }
                let mover = Capsule {
                    center1: proxy.points[0],
                    center2: proxy.points[1],
                    radius: proxy.radius,
                };
                let plane = (&raw mut world_query::RESULT as *mut u32).cast::<PlaneResult>();
                if query::collide_mover(
                    core::slice::from_raw_parts_mut(plane, 1),
                    &shape,
                    transform,
                    &mover,
                    materials,
                ) != 0
                {
                    world_query::callback(2, shape_id, plane.cast::<u8>(), 1);
                    count += 1;
                    if count == capacity {
                        return;
                    }
                }
                continue;
            }
            let out = if operation == 0 {
                query::ray_cast_shape(
                    &shape,
                    transform,
                    &RayCastInput {
                        origin: Vec3::ZERO,
                        translation,
                        max_fraction: fraction,
                    },
                )
            } else {
                query::shape_cast_shape(
                    &shape,
                    transform,
                    &ShapeCastInput {
                        proxy,
                        translation,
                        max_fraction: fraction,
                        can_encroach: r[13] != 0.0,
                    },
                )
            };
            if out.hit && out.fraction <= fraction {
                fraction = out.fraction;
                result.write(shape_id as u32);
                world_query::write_cast(&out, result.add(4), out.material_index);
            }
        }
        if operation == 3 {
            result.add(3).write(closest.to_bits());
            result.add(6).write(point.x.to_bits());
            result.add(7).write(point.y.to_bits());
            result.add(8).write(point.z.to_bits());
        }
    }
}

/// Sweep the body's eligible convex shapes against a translating capsule.
/// Input lanes 20..27 hold the end transform relative to the mover origin.
#[export_name = "bodyTimeOfImpactMover"]
pub unsafe extern "C" fn time_of_impact_mover(world: usize, body: usize) {
    use crate::distance::{time_of_impact, Sweep, TOIInput};
    use crate::math::{Quat, Transform};
    let (r, start, proxy) = query_abi::input();
    let end = Transform {
        p: Vec3::new(r[20], r[21], r[22]),
        q: Quat {
            v: Vec3::new(r[23], r[24], r[25]),
            s: r[26],
        },
    };
    let sim = crate::bodies::column(world, body, 1, crate::body::SIM_STRIDE);
    let local_center = crate::body::read_fin(sim, 0).local_center;
    let mut input = TOIInput {
        proxy_a: proxy,
        proxy_b: proxy,
        sweep_a: Sweep {
            local_center,
            c1: start.point(local_center),
            c2: end.point(local_center),
            q1: start.q,
            q2: end.q,
        },
        sweep_b: Sweep {
            local_center: Vec3::ZERO,
            c1: Vec3::ZERO,
            c2: Vec3::new(r[9], r[10], r[11]),
            q1: Quat::IDENTITY,
            q2: Quat::IDENTITY,
        },
        max_fraction: 1.0,
    };
    let result = &raw mut world_query::RESULT as *mut u32;
    result.write(NULL_SHAPE);
    let header = world_query::HEADER;
    let mut id = crate::bodies::record(world, body).head_shape_id;
    while id != -1 {
        let shape_id = id as usize;
        id = crate::shapes::col(world).get(shape_id * SHAPE_STRIDE + S_NEXT) as i32;
        if !world_query::accepts(world, shape_id, &header) {
            continue;
        }
        let (shape, _) = query_abi::shape(world, shape_id);
        if !matches!(
            shape,
            query::Shape::Sphere(_) | query::Shape::Capsule(_) | query::Shape::Hull(_)
        ) {
            continue;
        }
        let out = time_of_impact(&TOIInput {
            proxy_a: crate::continuous_shape::shape_proxy(&shape),
            ..input
        });
        if 0.0 < out.fraction && out.fraction < input.max_fraction {
            input.max_fraction = out.fraction;
            result.write(shape_id as u32);
            world_query::write_cast(
                &crate::distance::CastOutput {
                    hit: true,
                    fraction: out.fraction,
                    point: out.point,
                    normal: out.normal,
                    triangle_index: 0,
                    ..crate::distance::CastOutput::MISS
                },
                result.add(4),
                0,
            );
        }
    }
}
