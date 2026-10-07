//! Box3D body queries, in attachment order, using the caller-supplied placement.
use crate::distance::{shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::manifold::Capsule;
use crate::math::Vec3;
use crate::query::{self, PlaneResult, RayCastInput, ShapeCastInput};
use crate::shapes::{NULL_SHAPE, SHAPE_STRIDE, S_NEXT};
use crate::{query_abi, world_query};

/// Operations: ray, shape cast, overlap, closest point, collide mover.
#[export_name = "bodyQuery"]
pub extern "C" fn run(world: usize, operation: u32, head: u32, capacity: usize) {
    if operation == 4 && capacity == 0 {
        return;
    }
    unsafe {
        crate::shapes::shape_set_active_world(world as u32);
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
            id = crate::shapes::col().get(shape_id * SHAPE_STRIDE + S_NEXT);
            if operation != 3 && !world_query::accepts(shape_id, &header) {
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
