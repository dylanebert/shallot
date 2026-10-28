//! Standalone convex collision queries. All records are fixed, instance-local scratch memory.
use crate::distance::{self, DistanceInput, ShapeCastPairInput, ShapeProxy, Simplex, SimplexCache};
use crate::math::{Quat, Transform, Vec3};
use crate::toi::{Sweep, TOIInput};

static mut INPUT: [f32; 850] = [0.0; 850];
static mut OUTPUT: [f32; 1600] = [0.0; 1600];

#[export_name = "collisionInputPtr"]
pub extern "C" fn input_ptr() -> *mut f32 {
    (&raw mut INPUT).cast()
}
#[export_name = "collisionOutputPtr"]
pub extern "C" fn output_ptr() -> *const f32 {
    (&raw const OUTPUT).cast()
}
fn v(r: &[f32], i: usize) -> Vec3 {
    Vec3::new(r[i], r[i + 1], r[i + 2])
}
fn q(r: &[f32], i: usize) -> Quat {
    Quat {
        v: v(r, i),
        s: r[i + 3],
    }
}
fn sweep(r: &[f32], i: usize) -> Sweep {
    Sweep {
        local_center: v(r, i),
        c1: v(r, i + 3),
        c2: v(r, i + 6),
        q1: q(r, i + 9),
        q2: q(r, i + 13),
    }
}
fn put(r: &mut [f32], i: usize, p: Vec3) {
    r[i..i + 3].copy_from_slice(&[p.x, p.y, p.z]);
}
/// operation: 0 distance, 1 shape cast, 2 time of impact, 3 sweep transform.
#[export_name = "collisionQuery"]
pub unsafe extern "C" fn query(operation: u32) {
    let r = &*(&raw const INPUT);
    let out = &mut *(&raw mut OUTPUT);
    out.fill(0.0);
    if operation == 3 {
        let xf = sweep(r, 800).transform(r[834]);
        put(out, 0, xf.p);
        put(out, 3, xf.q.v);
        out[6] = xf.q.s;
        return;
    }
    let proxy = |i: usize| ShapeProxy {
        points: core::slice::from_raw_parts(r.as_ptr().add(i + 2).cast::<Vec3>(), 128),
        count: r[i] as usize,
        radius: r[i + 1],
    };
    let transform = Transform {
        p: v(r, 800),
        q: q(r, 803),
    };
    match operation {
        0 => {
            let mut cache = SimplexCache::empty();
            cache.metric = r[780];
            cache.count = r[781] as u16;
            for i in 0..4 {
                cache.index_a[i] = r[782 + i] as u8;
                cache.index_b[i] = r[786 + i] as u8;
            }
            let mut simplexes = [Simplex::empty(); 32];
            let d = distance::shape_distance_debug(
                &DistanceInput {
                    proxy_a: proxy(0),
                    proxy_b: proxy(386),
                    transform,
                    use_radii: r[807] != 0.0,
                },
                &mut cache,
                &mut simplexes[..(r[808] as usize).min(32)],
            );
            put(out, 0, d.point_a);
            put(out, 3, d.point_b);
            put(out, 6, d.normal);
            out[9] = d.distance;
            out[10] = d.iterations as f32;
            out[11] = d.simplex_count as f32;
            out[12] = cache.metric;
            out[13] = cache.count as f32;
            for i in 0..4 {
                out[14 + i] = cache.index_a[i] as f32;
                out[18 + i] = cache.index_b[i] as f32;
            }
            for (i, s) in simplexes[..d.simplex_count].iter().enumerate() {
                let base = 22 + i * 49;
                out[base] = s.count as f32;
                for (j, vertex) in s.vertices.iter().enumerate() {
                    let b = base + 1 + j * 12;
                    put(out, b, vertex.w_a);
                    put(out, b + 3, vertex.w_b);
                    put(out, b + 6, vertex.w);
                    out[b + 9] = vertex.a;
                    out[b + 10] = vertex.index_a as f32;
                    out[b + 11] = vertex.index_b as f32;
                }
            }
        }
        1 => {
            let c = distance::shape_cast(&ShapeCastPairInput {
                proxy_a: proxy(0),
                proxy_b: proxy(386),
                transform,
                translation_b: v(r, 807),
                max_fraction: r[810],
                can_encroach: r[811] != 0.0,
            });
            put(out, 0, c.point);
            put(out, 3, c.normal);
            out[6] = c.fraction;
            out[7] = c.iterations as f32;
            out[8] = u32::from(c.hit) as f32;
        }
        2 => {
            let t = distance::time_of_impact(&TOIInput {
                proxy_a: proxy(0),
                proxy_b: proxy(386),
                sweep_a: sweep(r, 800),
                sweep_b: sweep(r, 817),
                max_fraction: r[834],
            });
            put(out, 0, t.point);
            put(out, 3, t.normal);
            out[6] = t.fraction;
            out[7] = t.state as f32;
            out[8] = t.distance;
            out[9] = t.distance_iterations as f32;
            out[10] = t.push_back_iterations as f32;
            out[11] = t.root_iterations as f32;
        }
        _ => unreachable!(),
    }
}
