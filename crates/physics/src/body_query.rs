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
    unsafe {
        crate::shapes::shape_set_active_world(world as u32);
        let header = world_query::HEADER;
        let (r, transform, proxy) = query_abi::input();
        let translation = Vec3::new(r[9], r[10], r[11]);
        let mut fraction = r[12];
        let mut result = [0u32; 16];
        result[0] = NULL_SHAPE;
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
                    result[0] = shape_id as u32;
                    break;
                }
                continue;
            }
            if operation == 3 {
                let mut points = [Vec3::ZERO; 2];
                let convex = match &shape {
                    query::Shape::Sphere(s) => {
                        points[0] = s.center;
                        ShapeProxy {
                            points: &points,
                            count: 1,
                            radius: s.radius,
                        }
                    }
                    query::Shape::Capsule(s) => {
                        points[0] = s.center1;
                        points[1] = s.center2;
                        ShapeProxy {
                            points: &points,
                            count: 2,
                            radius: s.radius,
                        }
                    }
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
                if count >= capacity {
                    break;
                }
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
                let mut planes = [PlaneResult::ZERO; 1];
                if query::collide_mover(&mut planes, &shape, transform, &mover, materials) != 0 {
                    world_query::callback(2, shape_id, planes.as_ptr() as *const u8, 1);
                    count += 1;
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
                result[0] = shape_id as u32;
                for (j, value) in world_query::cast_record(&out).into_iter().enumerate() {
                    result[4 + j] = value.to_bits();
                }
            }
        }
        if operation == 3 {
            result[3] = closest.to_bits();
            result[6] = point.x.to_bits();
            result[7] = point.y.to_bits();
            result[8] = point.z.to_bits();
        }
        world_query::RESULT[..16].copy_from_slice(&result);
    }
}
