//! Read-only debug draw, buffered before JavaScript callback dispatch.
use crate::{
    bodies, body, broad,
    joint_abi::*,
    joint_record,
    math::{Quat, Transform, Vec3},
    shapes, tree,
};
static mut BUFFER: Vec<u32> = Vec::new();
static mut VISITED: Vec<u64> = Vec::new();
static mut BODY_ORDER: Vec<usize> = Vec::new();
fn vector(out: &mut Vec<u32>, v: Vec3) {
    out.extend([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]);
}
fn transform(out: &mut Vec<u32>, t: Transform) {
    vector(out, t.p);
    vector(out, t.q.v);
    out.push(t.q.s.to_bits());
}
fn begin(out: &mut Vec<u32>, kind: u32, color: u32) -> usize {
    let n = out.len();
    out.extend([kind, 0, color]);
    n
}
fn end(out: &mut [u32], n: usize) {
    out[n + 1] = (out.len() - n) as u32;
}
fn segment(out: &mut Vec<u32>, a: Vec3, b: Vec3, color: u32) {
    let n = begin(out, 6, color);
    vector(out, a);
    vector(out, b);
    end(out, n);
}
fn point(out: &mut Vec<u32>, p: Vec3, size: f32, color: u32) {
    let n = begin(out, 7, color);
    vector(out, p);
    out.push(size.to_bits());
    end(out, n);
}
fn frame(out: &mut Vec<u32>, t: Transform) {
    let n = begin(out, 8, 0);
    transform(out, t);
    end(out, n);
}
pub(crate) unsafe fn pose(world_index: usize, id: usize) -> Transform {
    let sim = bodies::column(world_index, id, 1, body::SIM_STRIDE);
    let fin = bodies::column(world_index, id, 2, body::FIN_STRIDE);
    Transform {
        p: Vec3::new(fin.get(0), fin.get(1), fin.get(2)),
        q: Quat {
            v: Vec3::new(sim.get(3), sim.get(4), sim.get(5)),
            s: sim.get(6),
        },
    }
}
unsafe fn image(out: &mut Vec<u32>, kind: u32, p: *const u32) {
    let words = if kind == 3 { *p.add(35) } else { *p.add(4) } as usize / 4;
    out.extend_from_slice(core::slice::from_raw_parts(p, words));
}
unsafe fn solid(
    out: &mut Vec<u32>,
    kind: u32,
    geo: *const u32,
    scale: Vec3,
    t: Transform,
    color: u32,
) {
    let n = begin(out, kind, color);
    transform(out, t);
    if kind == 0 {
        out.extend_from_slice(core::slice::from_raw_parts(geo, 7));
    } else if kind == 5 {
        out.extend_from_slice(core::slice::from_raw_parts(geo, 4));
    } else {
        vector(out, scale);
        image(out, kind, geo);
    }
    end(out, n);
}
unsafe fn vf(p: *const u32, n: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(*p.add(n)),
        f32::from_bits(*p.add(n + 1)),
        f32::from_bits(*p.add(n + 2)),
    )
}
unsafe fn tf(p: *const u32, n: usize) -> Transform {
    Transform {
        p: vf(p, n),
        q: Quat {
            v: vf(p, n + 3),
            s: f32::from_bits(*p.add(n + 6)),
        },
    }
}
unsafe fn shape(world_index: usize, out: &mut Vec<u32>, id: usize, t: Transform, color: u32) {
    let r = shapes::col_slice(world_index);
    let n = id * shapes::SHAPE_STRIDE;
    let kind = r[n + shapes::S_TYPE];
    let geo = r.as_ptr().add(n + shapes::S_GEOM);
    if kind != 1 {
        let p = if kind == 0 || kind == 5 {
            geo
        } else {
            *geo as *const u32
        };
        solid(
            out,
            kind,
            p,
            if kind == 4 { vf(geo, 1) } else { Vec3::ZERO },
            t,
            color,
        );
        return;
    }
    let p = *geo as *const u32;
    for (kind, offset_lane, count_lane, stride) in [
        (0, 19, 20, 8),
        (3, 21, 22, 9),
        (4, 24, 25, 15),
        (5, 27, 28, 5),
    ] {
        let start = *p.add(offset_lane) as usize / 4;
        for i in 0..*p.add(count_lane) as usize {
            let c = p.add(start + i * stride);
            let (geometry, child, scale) = if kind == 0 || kind == 5 {
                (c, Transform::IDENTITY, Vec3::ZERO)
            } else if kind == 3 {
                (p.add(*c.add(7) as usize / 4), tf(c, 0), Vec3::ZERO)
            } else {
                (p.add(*c.add(10) as usize / 4), tf(c, 0), vf(c, 7))
            };
            solid(out, kind, geometry, scale, t.mul(child), color);
        }
    }
}
unsafe fn color(world: usize, id: usize) -> u32 {
    let r = shapes::col_slice(world as usize);
    let n = id * shapes::SHAPE_STRIDE;
    let custom = shapes::material(world as usize, id, 0)[8];
    if custom != 0 {
        return custom;
    }
    let body_id = r[n + shapes::S_QUERY_BODY] as usize;
    let b = bodies::record(world, body_id);
    let sim = bodies::column(world as usize, body_id, 5, body::SIM2_STRIDE);
    let flags = sim.get(body::S2_FLAGS).to_bits();
    if b.body_type == 2 && b.mass == 0.0 {
        0xff0000
    } else if b.set_index == 1 {
        0x708090
    } else if r[n + shapes::S_FLAGS] & shapes::SENSOR_FLAG != 0 {
        0xf5deb3
    } else if b.flags & 0x200 != 0 {
        0x00ff00
    } else if flags & 0x80 != 0 && b.set_index == 2 {
        0x40e0d0
    } else if b.flags & 0x100 != 0 {
        0xffff00
    } else if flags & 0x40 != 0 {
        0xffa500
    } else if b.body_type == 0 {
        0xa9a9a9
    } else if b.body_type == 1 {
        if b.set_index == 2 {
            0x4682b4
        } else {
            0xb0c4de
        }
    } else if b.set_index == 2 {
        0xd2b48c
    } else {
        0x778899
    }
}
#[export_name = "worldDrawShape"]
pub unsafe extern "C" fn observe_shape(world: usize, id: usize) -> usize {
    crate::regions::select(world as u32);
    unsafe { observe_shape_in_world(world, id) }
}

pub unsafe extern "C" fn observe_shape_in_world(world: usize, id: usize) -> usize {
    let out = &mut *(&raw mut BUFFER);
    let start = out.len();
    shape(world as usize, out, id, Transform::IDENTITY, 0);
    start
}
#[export_name = "worldDrawRelease"]
pub unsafe extern "C" fn release(start: usize) {
    (&mut *(&raw mut BUFFER)).truncate(start);
}
#[export_name = "worldDrawPtr"]
pub extern "C" fn pointer() -> *const u32 {
    unsafe { (&*(&raw const BUFFER)).as_ptr() }
}
#[export_name = "worldDrawLen"]
pub extern "C" fn length() -> usize {
    unsafe { (&*(&raw const BUFFER)).len() }
}
#[export_name = "worldDraw"]
pub unsafe extern "C" fn run(
    world: usize,
    flags: u32,
    mask_hi: u32,
    mask_lo: u32,
    inv_h: f32,
) -> usize {
    crate::regions::select(world as u32);
    unsafe { run_in_world(world, flags, mask_hi, mask_lo, inv_h) }
}

pub unsafe extern "C" fn run_in_world(
    world: usize,
    flags: u32,
    mask_hi: u32,
    mask_lo: u32,
    inv_h: f32,
) -> usize {
    let header = crate::world_query::HEADER;
    let out = &mut *(&raw mut BUFFER);
    let start = out.len();
    let visited = &mut *(&raw mut VISITED);
    visited.resize(bodies::body_cap_in_world(world as usize).div_ceil(64), 0);
    visited.fill(0);
    let order = &mut *(&raw mut BODY_ORDER);
    order.clear();
    let lo = [
        f32::from_bits(header[13]),
        f32::from_bits(header[14]),
        f32::from_bits(header[15]),
    ];
    let hi = [
        f32::from_bits(header[16]),
        f32::from_bits(header[17]),
        f32::from_bits(header[18]),
    ];
    let mut stack = [0; tree::STACK_SIZE];
    for i in 0..3 {
        let pool = core::slice::from_raw_parts(
            broad::tree_ptr(world as usize, i),
            broad::tree_cap(world as usize, i) * tree::STRIDE,
        );
        tree::query(
            pool,
            header[2 * i] as i32,
            header[2 * i + 1] as usize,
            lo,
            hi,
            mask_hi,
            mask_lo,
            false,
            &mut stack,
            |_, id| {
                let id = id as usize;
                let r = shapes::col_slice(world as usize);
                let n = id * shapes::SHAPE_STRIDE;
                let body_id = r[n + shapes::S_QUERY_BODY] as usize;
                let bit = 1u64 << (body_id % 64);
                if visited[body_id / 64] & bit == 0 {
                    visited[body_id / 64] |= bit;
                    order.push(body_id);
                }
                if flags & 1 != 0 {
                    shape(
                        world as usize,
                        out,
                        id,
                        pose(world as usize, body_id),
                        color(world, id),
                    );
                }
                if flags & 2 != 0 {
                    let o = begin(out, 9, 0xffd700);
                    let bounds = n + shapes::S_FAT_AABB;
                    out.extend_from_slice(&r[bounds..bounds + 6]);
                    end(out, o);
                }
                true
            },
        );
    }
    if flags & 4 != 0 {
        for &id in order.iter() {
            let b = bodies::record(world, id);
            if b.body_type != 2 {
                continue;
            }
            let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
            let t = Transform {
                p: body::read_fin(fin, 0).center,
                q: pose(world as usize, id).q,
            };
            frame(out, t);
            let n = begin(out, 10, 0xffffff);
            vector(out, t.point(Vec3::new(0.1, 0.1, 0.1)));
            out.push(b.mass.to_bits());
            end(out, n);
        }
    }
    if flags & 8 != 0 {
        for id in 0..joint_record::capacity_in_world(world as usize) {
            let j = joint_record::record(world as usize, id);
            if j.set_index < 0 {
                continue;
            }
            let a = j.edges[0].body_id as usize;
            let b = j.edges[1].body_id as usize;
            if bodies::record(world, a).set_index == 1 || bodies::record(world, b).set_index == 1 {
                continue;
            }
            let ta = pose(world as usize, a);
            let tb = pose(world as usize, b);
            let col = crate::col::Col::new(
                joint_record::sim_pointer_in_world(world as usize, id) as *mut f32,
                JOINT_STRIDE,
            );
            let pa = ta.point(get_vec3(col, 0, J_LOCAL_FRAME_A));
            let pb = tb.point(get_vec3(col, 0, J_LOCAL_FRAME_B));
            if j.joint_type == 2 {
                segment(out, pa, pb, 0xffd700);
                continue;
            }
            if j.joint_type == 3 {
                segment(out, pa, pb, 0xdda0dd);
                point(out, pa, 8.0, 0x9acd32);
                point(out, pb, 8.0, 0xdda0dd);
                continue;
            }
            segment(out, ta.p, pa, 0x8fbc8f);
            segment(out, pa, pb, 0x8fbc8f);
            segment(out, tb.p, pb, 0x8fbc8f);
            point(out, pa, 6.0, 0x8fbc8f);
            point(out, pb, 6.0, 0x8fbc8f);
            frame(out, Transform { p: pa, q: ta.q });
            frame(out, Transform { p: pb, q: tb.q });
            if flags & 16 != 0 {
                let (force, torque) = crate::joint_draw::reaction(col, ta, tb, inv_h);
                let p = pa.lerp(pb, 0.5);
                // The published draw used a JavaScript number for its display scale.
                let scaled = Vec3::new(
                    (0.001_f64 * force.x as f64) as f32,
                    (0.001_f64 * force.y as f64) as f32,
                    (0.001_f64 * force.z as f64) as f32,
                );
                segment(out, p, p.add(scaled), 0xf0ffff);
                let n = begin(out, 11, 0xf0ffff);
                vector(out, p);
                out.extend([force.length().to_bits(), torque.length().to_bits()]);
                end(out, n);
            }
        }
    }
    start
}
