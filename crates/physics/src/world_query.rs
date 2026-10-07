//! Box3D world queries over the resident broad-phase pools and shape records.
use crate::distance::{CastOutput, ShapeProxy};
use crate::manifold::Capsule;
use crate::math::{maxf, minf, Quat, Transform, Vec3};
use crate::mesh_query::proxy_bounds;
use crate::query::{self, PlaneResult, RayCastInput, ShapeCastInput};
use crate::shapes::*;
use crate::{broad, query_abi, tree};

// roots/counts, category hi/lo, mask hi/lo, origin xyz, box lower/upper xyz, excluded body + 1.
pub(crate) static mut HEADER: [u32; 20] = [0; 20];
// shape id (-1 for no hit), node/leaf visits, final fraction, followed by cast output.
// Callback-free collide mover: count at 0, then eight shape/normal/offset/point records at 16.
pub(crate) static mut RESULT: [u32; 80] = [0; 80];
#[export_name = "worldQueryHeaderPtr"]
pub extern "C" fn header_ptr() -> *mut u32 {
    &raw mut HEADER as *mut u32
}
#[export_name = "worldQueryResultPtr"]
pub extern "C" fn result_ptr() -> *const u32 {
    &raw const RESULT as *const u32
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn queryCallback(kind: u32, shape: u32, data: *const u8, count: usize) -> f32;
}
pub(crate) unsafe fn callback(kind: u32, id: usize, data: *const u8, count: usize) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        queryCallback(kind, id as u32, data, count)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (kind, id, data, count);
        1.0
    }
}
fn v(r: &[u32], i: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(r[i]),
        f32::from_bits(r[i + 1]),
        f32::from_bits(r[i + 2]),
    )
}
pub(crate) unsafe fn pose(world_index: usize, id: usize, origin: Vec3) -> Transform {
    let body = crate::shapes::col(world_index).get(id * SHAPE_STRIDE + S_QUERY_BODY) as usize;
    let fin = crate::bodies::column(world_index, body, 2, crate::body::FIN_STRIDE);
    let sim = crate::bodies::column(world_index, body, 1, crate::body::SIM_STRIDE);
    Transform {
        p: Vec3::new(fin.get(0), fin.get(1), fin.get(2)).sub(origin),
        q: Quat {
            v: Vec3::new(sim.get(3), sim.get(4), sim.get(5)),
            s: sim.get(6),
        },
    }
}
pub(crate) fn accepts(world_index: usize, id: usize, header: &[u32; 20]) -> bool {
    let r = crate::shapes::col_slice(world_index);
    let n = id * SHAPE_STRIDE;
    ((r[n + S_QUERY_CATEGORY] & header[8]) | (r[n + S_QUERY_CATEGORY - 1] & header[9])) != 0
        && ((r[n + S_QUERY_MASK] & header[6]) | (r[n + S_QUERY_MASK - 1] & header[7])) != 0
}
pub(crate) unsafe fn write_cast(out: &CastOutput, target: *mut u32, material: i32) {
    for (index, value) in [
        u32::from(out.hit) as f32,
        out.fraction,
        out.point.x,
        out.point.y,
        out.point.z,
        out.normal.x,
        out.normal.y,
        out.normal.z,
        out.iterations as f32,
        out.triangle_index as f32,
        out.child_index as f32,
        material as f32,
    ]
    .into_iter()
    .enumerate()
    {
        target.add(index).write(value.to_bits());
    }
}
// Traversal and narrow-phase scratch belongs to the calling sensor task.
pub(crate) unsafe fn sensor_task(
    world_index: usize,
    sensor_id: usize,
    mut emit: impl FnMut(usize),
) {
    {
        let r = crate::shapes::col(world_index);
        let n = sensor_id * SHAPE_STRIDE;
        let lo = Vec3::new(f32::from_bits(r.get(n + 10)), f32::from_bits(r.get(n + 11)), f32::from_bits(r.get(n + 12)));
        let hi = Vec3::new(f32::from_bits(r.get(n + 13)), f32::from_bits(r.get(n + 14)), f32::from_bits(r.get(n + 15)));
        let (sensor, _) = query_abi::active_shape(world_index, sensor_id);
        let sensor_transform = pose(world_index, sensor_id, Vec3::ZERO);
        let mut stack = [0; tree::STACK_SIZE];
        for i in 0..3 {
            let pool = core::slice::from_raw_parts(
                broad::tree_ptr(world_index, i),
                broad::tree_cap(world_index, i) * tree::STRIDE,
            );
            tree::query(
                pool,
                if broad::tree_cap(world_index, i) == 0 { -1 } else { *broad::tree_state(world_index, i) as i32 },
                if broad::tree_cap(world_index, i) == 0 { 0 } else { *broad::tree_state(world_index, i).add(1) as usize },
                [lo.x, lo.y, lo.z],
                [hi.x, hi.y, hi.z],
                r.get(n + S_QUERY_MASK),
                r.get(n + S_QUERY_MASK - 1),
                false,
                &mut stack,
                |_, id| {
                    let id = id as usize;
                    let o = id * SHAPE_STRIDE;
                    if id == sensor_id {
                        return true;
                    }
                    if !matches!(
                        r.get(o + S_TYPE),
                        crate::finalize::TY_SPHERE
                            | crate::finalize::TY_CAPSULE
                            | crate::finalize::TY_HULL
                    ) {
                        return true;
                    }
                    if r.get(o + S_QUERY_SENSOR) & SENSOR_FLAG == 0
                        || r.get(n + S_QUERY_BODY) == r.get(o + S_QUERY_BODY)
                    {
                        return true;
                    }
                    let group = r.get(n + S_QUERY_GROUP) as i32;
                    let same_group = group != 0 && group == r.get(o + S_QUERY_GROUP) as i32;
                    if same_group {
                        if group < 0 {
                            return true;
                        }
                    } else if ((r.get(n + S_QUERY_CATEGORY) & r.get(o + S_QUERY_MASK))
                        | (r.get(n + S_QUERY_CATEGORY - 1) & r.get(o + S_QUERY_MASK - 1)))
                        == 0
                        || ((r.get(o + S_QUERY_CATEGORY) & r.get(n + S_QUERY_MASK))
                            | (r.get(o + S_QUERY_CATEGORY - 1) & r.get(n + S_QUERY_MASK - 1)))
                            == 0
                    {
                        return true;
                    }
                    let (visitor, _) = query_abi::active_shape(world_index, id);
                    let relative = sensor_transform.inv_mul(pose(world_index, id, Vec3::ZERO));
                    let mut points = [core::mem::MaybeUninit::<Vec3>::uninit(); 128];
                    let (count, radius) = match visitor {
                        query::Shape::Sphere(s) => {
                            points[0].write(relative.point(s.center));
                            (1, s.radius)
                        }
                        query::Shape::Capsule(s) => {
                            points[0].write(relative.point(s.center1));
                            points[1].write(relative.point(s.center2));
                            (2, s.radius)
                        }
                        query::Shape::Hull(h) => {
                            let count = h.vertex_count.min(128);
                            for j in 0..count {
                                points[j].write(relative.point(h.points[j]));
                            }
                            (count, 0.0)
                        }
                        query::Shape::Mesh(_) | query::Shape::Height(_)
                            if matches!(
                                sensor,
                                query::Shape::Mesh(_) | query::Shape::Height(_)
                            ) =>
                        {
                            return true
                        }
                        _ => panic!("physics: mesh/height/compound have no shape proxy"),
                    };
                    if query::overlap_shape(
                        &sensor,
                        Transform::IDENTITY,
                        ShapeProxy {
                            // Each convex branch initializes precisely count points.
                            points: core::slice::from_raw_parts(
                                points.as_ptr().cast::<Vec3>(),
                                count,
                            ),
                            count,
                            radius,
                        },
                    ) {
                        emit(id);
                    }
                    true
                },
            );
        }
    }
}

/// Operations: AABB overlap, shape overlap, ray, closest ray, shape cast, collide mover, cast mover.
/// Without a callback, collide mover publishes the first eight planes in traversal order.
#[export_name = "worldQuery"]
pub extern "C" fn run(world: usize, operation: u32, user_callback: u32) {
    run_in_world(world, operation, user_callback)
}

pub extern "C" fn run_in_world(world: usize, operation: u32, user_callback: u32) {
    unsafe {
        let header = HEADER;
        let exclude_body = matches!(operation, 3 | 5 | 6) && header[19] != 0;
        let origin = v(&header, 10);
        let (r, _, proxy) = query_abi::input();
        // A user callback can recursively query; keep inputs independent of the shared ABI scratch.
        let mut points = [core::mem::MaybeUninit::<Vec3>::uninit(); 128];
        let proxy = if user_callback != 0 && matches!(operation, 1 | 4 | 5 | 6) {
            for (target, point) in points.iter_mut().zip(&proxy.points[..proxy.count]) {
                target.write(*point);
            }
            ShapeProxy {
                points: core::slice::from_raw_parts(points.as_ptr().cast::<Vec3>(), proxy.count),
                count: proxy.count,
                radius: proxy.radius,
            }
        } else {
            proxy
        };
        let translation = Vec3::new(r[9], r[10], r[11]);
        let mut fraction = 1.0;
        let (mut lo, mut hi) = if operation == 0 {
            (v(&header, 13), v(&header, 16))
        } else if operation == 2 || operation == 3 {
            // Ray traversal uses origin and translation, not a shape proxy's bounds.
            (Vec3::ZERO, Vec3::ZERO)
        } else if operation == 5 {
            let a = proxy.points[0];
            let b = proxy.points[1];
            let radius = Vec3::new(proxy.radius, proxy.radius, proxy.radius);
            (
                Vec3::new(minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z)).sub(radius),
                Vec3::new(maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z)).add(radius),
            )
        } else {
            proxy_bounds(proxy)
        };
        if operation != 0 && operation != 2 && operation != 3 {
            lo = lo.add(origin);
            hi = hi.add(origin);
        }
        let result = &raw mut RESULT as *mut u32;
        result.write(if operation == 5 { 0 } else { u32::MAX });
        let mut plane_count = 0;
        let mut node_visits = 0;
        let mut leaf_visits = 0;
        let mut stack = [0; tree::STACK_SIZE];
        for i in 0..3 {
            let pool = core::slice::from_raw_parts(
                broad::tree_ptr(world, i),
                broad::tree_cap(world, i) * tree::STRIDE,
            );
            let root = header[i * 2] as i32;
            let count = header[i * 2 + 1] as usize;
            let tree_fraction = fraction;
            let mut visit = |clip: f32, _: i32, shape_id: u32| -> f32 {
                let id = shape_id as usize;
                if !accepts(world, id, &header)
                    || (exclude_body
                        && crate::shapes::col(world).get(id * SHAPE_STRIDE + S_QUERY_BODY)
                            + 1
                            == header[19])
                {
                    return clip;
                }
                if operation == 0 {
                    return if user_callback != 0 {
                        callback(0, id, core::ptr::null(), 0)
                    } else {
                        1.0
                    };
                }
                if operation == 6
                    && user_callback != 0
                    && callback(3, id, core::ptr::null(), 0) == 0.0
                {
                    return fraction;
                }
                let (shape, materials) = query_abi::shape(world, id);
                let transform = pose(world, id, origin);
                if operation == 1 {
                    if !query::overlap_shape(&shape, transform, proxy) {
                        return 1.0;
                    }
                    return if user_callback != 0 {
                        callback(0, id, core::ptr::null(), 0)
                    } else {
                        1.0
                    };
                }
                if operation == 5 {
                    let mover = Capsule {
                        center1: proxy.points[0],
                        center2: proxy.points[1],
                        radius: proxy.radius,
                    };
                    let mut planes = [PlaneResult::ZERO; 64];
                    let count =
                        query::collide_mover(&mut planes, &shape, transform, &mover, materials);
                    if user_callback == 0 {
                        for plane in &planes[..count] {
                            let index = plane_count;
                            if index == 8 {
                                break;
                            }
                            let n = 16 + index * 8;
                            result.add(n).write(shape_id);
                            let values = [
                                plane.plane.normal.x,
                                plane.plane.normal.y,
                                plane.plane.normal.z,
                                plane.plane.offset,
                                plane.point.x,
                                plane.point.y,
                                plane.point.z,
                            ];
                            for j in 0..7 {
                                result.add(n + 1 + j).write(values[j].to_bits());
                            }
                            plane_count += 1;
                        }
                    }
                    return if count != 0 && user_callback != 0 {
                        callback(2, id, planes.as_ptr() as *const u8, count)
                    } else {
                        1.0
                    };
                }
                let output = if operation == 2 || operation == 3 {
                    query::ray_cast_shape(
                        &shape,
                        transform,
                        &RayCastInput {
                            origin: Vec3::ZERO,
                            translation,
                            max_fraction: clip,
                        },
                    )
                } else {
                    query::shape_cast_shape(
                        &shape,
                        transform,
                        &ShapeCastInput {
                            proxy,
                            translation,
                            max_fraction: clip,
                            can_encroach: operation == 6 && proxy.radius > 0.0,
                        },
                    )
                };
                if !output.hit || ((operation == 3 || operation == 6) && output.fraction == 0.0) {
                    return clip;
                }
                if (user_callback != 0 && operation != 6) || operation == 3 {
                    let material_count = shape_material_count(world as u32, id as u32);
                    write_cast(
                        &output,
                        result.add(4),
                        output
                            .material_index
                            .clamp(0, material_count.saturating_sub(1) as i32),
                    );
                }
                let value = if user_callback != 0 && operation != 6 {
                    callback(1, id, result.add(4).cast::<u8>(), 12)
                } else {
                    output.fraction
                };
                if (0.0..=1.0).contains(&value) {
                    fraction = value;
                }
                if operation == 3 {
                    result.write(shape_id);
                }
                value
            };
            let stats = if operation <= 1 || operation == 5 {
                tree::query(
                    pool,
                    root,
                    count,
                    [lo.x, lo.y, lo.z],
                    [hi.x, hi.y, hi.z],
                    header[8],
                    header[9],
                    false,
                    &mut stack,
                    |id, data| visit(1.0, id, data) != 0.0,
                )
            } else if operation == 2 || operation == 3 {
                tree::ray_cast(
                    pool,
                    root,
                    count,
                    origin,
                    translation,
                    tree_fraction,
                    header[8],
                    header[9],
                    false,
                    visit,
                )
            } else {
                tree::box_cast(
                    pool,
                    root,
                    count,
                    lo,
                    hi,
                    translation,
                    tree_fraction,
                    header[8],
                    header[9],
                    false,
                    visit,
                )
            };
            node_visits += stats.0;
            leaf_visits += stats.1;
            if operation >= 2 && operation != 5 && fraction == 0.0 {
                break;
            }
        }
        if operation == 5 {
            result.write(plane_count as u32);
        }
        result.add(1).write(node_visits);
        result.add(2).write(leaf_visits);
        result.add(3).write(fraction.to_bits());
    }
}
