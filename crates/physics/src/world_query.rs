//! Box3D world queries over the resident broad-phase pools and shape records.
use crate::distance::{CastOutput, ShapeProxy};
use crate::manifold::Capsule;
use crate::math::{Quat, Transform, Vec3};
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
pub(crate) unsafe fn pose(id: usize, origin: Vec3) -> Transform {
    let r = crate::shapes::col_slice();
    let n = id * SHAPE_STRIDE + S_QUERY_POSE;
    let awake = r[id * SHAPE_STRIDE + 32];
    if awake != 0 {
        let index = awake as usize - 1;
        let fin = core::slice::from_raw_parts(
            (crate::bodies::fin_base() as *const u32).add(index * 12),
            12,
        );
        let sim = core::slice::from_raw_parts(
            (crate::bodies::sim_base() as *const u32).add(index * 32),
            32,
        );
        return Transform {
            p: v(fin, 9).sub(origin),
            q: Quat {
                v: v(sim, 28),
                s: f32::from_bits(sim[31]),
            },
        };
    }
    Transform {
        p: v(r, n).sub(origin),
        q: Quat {
            v: v(r, n + 3),
            s: f32::from_bits(r[n + 6]),
        },
    }
}
pub(crate) fn accepts(id: usize, header: &[u32; 20]) -> bool {
    let r = crate::shapes::col_slice();
    let n = id * SHAPE_STRIDE;
    (header[19] == 0 || r[n + S_QUERY_BODY] + 1 != header[19])
        && ((r[n + S_QUERY_CATEGORY] & header[8]) | (r[n + S_QUERY_CATEGORY + 1] & header[9])) != 0
        && ((r[n + S_QUERY_MASK] & header[6]) | (r[n + S_QUERY_MASK + 1] & header[7])) != 0
}
pub(crate) fn cast_record(out: &CastOutput) -> [f32; 12] {
    [
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
        out.material_index as f32,
    ]
}
// Header, traversal stack and narrow-phase points belong to the calling task, not the query ABI.
pub(crate) unsafe fn sensor_task(
    sensor_id: usize,
    header: &[u32; 20],
    mut emit: impl FnMut(usize),
) {
    {
        let lo = v(header, 13);
        let hi = v(header, 16);
        let r = crate::shapes::col();
        let n = sensor_id * SHAPE_STRIDE;
        let (sensor, _) = query_abi::active_shape(sensor_id);
        let sensor_transform = pose(sensor_id, Vec3::ZERO);
        let mut stack = [0; tree::STACK_SIZE];
        for i in 0..3 {
            let pool =
                core::slice::from_raw_parts(broad::tree_ptr(i), broad::tree_cap(i) * tree::STRIDE);
            tree::query(
                pool,
                header[2 * i] as i32,
                header[2 * i + 1] as usize,
                [lo.x, lo.y, lo.z],
                [hi.x, hi.y, hi.z],
                r.get(n + S_QUERY_MASK),
                r.get(n + S_QUERY_MASK + 1),
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
                        | (r.get(n + S_QUERY_CATEGORY + 1) & r.get(o + S_QUERY_MASK + 1)))
                        == 0
                        || ((r.get(o + S_QUERY_CATEGORY) & r.get(n + S_QUERY_MASK))
                            | (r.get(o + S_QUERY_CATEGORY + 1) & r.get(n + S_QUERY_MASK + 1)))
                            == 0
                    {
                        return true;
                    }
                    let (visitor, _) = query_abi::active_shape(id);
                    let relative = sensor_transform.inv_mul(pose(id, Vec3::ZERO));
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
    unsafe {
        crate::shapes::shape_set_active_world(world as u32);
        let header = HEADER;
        let origin = v(&header, 10);
        let (r, _, proxy) = query_abi::input();
        // A user callback can recursively query; keep inputs independent of the shared ABI scratch.
        let mut points = [Vec3::ZERO; 128];
        points[..proxy.count].copy_from_slice(&proxy.points[..proxy.count]);
        let proxy = ShapeProxy {
            points: &points,
            count: proxy.count,
            radius: proxy.radius,
        };
        let translation = Vec3::new(r[9], r[10], r[11]);
        let mut fraction = 1.0;
        let (mut lo, mut hi) = if operation == 0 {
            (v(&header, 13), v(&header, 16))
        } else {
            proxy_bounds(proxy)
        };
        if operation != 0 {
            lo = lo.add(origin);
            hi = hi.add(origin);
        }
        let mut result = [0u32; 80];
        result[0] = if operation == 5 { 0 } else { u32::MAX };
        let mut stack = [0; tree::STACK_SIZE];
        for i in 0..3 {
            let pool =
                core::slice::from_raw_parts(broad::tree_ptr(i), broad::tree_cap(i) * tree::STRIDE);
            let root = header[i * 2] as i32;
            let count = header[i * 2 + 1] as usize;
            let tree_fraction = fraction;
            let mut visit = |clip: f32, _: i32, shape_id: u32| -> f32 {
                let id = shape_id as usize;
                if !accepts(id, &header) {
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
                let transform = pose(id, origin);
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
                        center1: points[0],
                        center2: points[1],
                        radius: proxy.radius,
                    };
                    let mut planes = [PlaneResult::ZERO; 64];
                    let count =
                        query::collide_mover(&mut planes, &shape, transform, &mover, materials);
                    if user_callback == 0 {
                        for plane in &planes[..count] {
                            let index = result[0] as usize;
                            if index == 8 {
                                break;
                            }
                            let n = 16 + index * 8;
                            result[n] = shape_id;
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
                                result[n + 1 + j] = values[j].to_bits();
                            }
                            result[0] += 1;
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
                let mut record = cast_record(&output);
                let material_count = shape_material_count(world as u32, id as u32);
                record[11] = output
                    .material_index
                    .clamp(0, material_count.saturating_sub(1) as i32)
                    as f32;
                let value = if user_callback != 0 && operation != 6 {
                    callback(1, id, record.as_ptr() as *const u8, 12)
                } else {
                    output.fraction
                };
                if (0.0..=1.0).contains(&value) {
                    fraction = value;
                }
                if operation == 3 {
                    result[0] = shape_id;
                    for j in 0..12 {
                        result[4 + j] = record[j].to_bits();
                    }
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
            result[1] += stats.0;
            result[2] += stats.1;
            if operation >= 2 && operation != 5 && fraction == 0.0 {
                break;
            }
        }
        result[3] = fraction.to_bits();
        RESULT = result;
    }
}
