//! ETC1, EAC A8 and BC7 mode 5 output from Basis Universal 073eb6a0.
//! Altered source: Rust arrays and explicit block bit fields instead of C++ bitfields.
//! Copyright 2016–2026 Binomial LLC. See ../../licenses/{LICENSE,NOTICE}-basis-universal.
use super::{Endpoint, Selector};
#[path = "tables.rs"]
mod tables;
use tables::{BC7_ALPHA, BC7_COLOR, BC7_SOLID, EAC};

fn selector_range(s: &Selector) -> (usize, usize, usize) {
    (usize::from(s.lo), usize::from(s.hi), usize::from(s.unique))
}
fn range_index(lo: usize, hi: usize) -> usize {
    [(0, 3), (1, 3), (0, 2), (1, 2), (2, 3), (0, 1)]
        .iter()
        .position(|&v| v == (lo, hi))
        .unwrap_or(0)
}
pub(super) fn etc1(out: &mut [u8], e: Endpoint, s: &Selector) {
    for (i, &c) in e.color.iter().enumerate() {
        out[i] = (c << 3) as u8;
    }
    out[3] = ((e.intensity << 5) | (e.intensity << 2) | 2) as u8;
    out[4..].copy_from_slice(&s.bytes);
}
pub(super) fn eac(out: &mut [u8], alpha: Option<(Endpoint, &Selector)>) {
    let Some((e, s)) = alpha else {
        out.copy_from_slice(&[255, 0x1d, 0x92, 0x49, 0x24, 0x92, 0x49, 0x24]);
        return;
    };
    let (lo, hi, _) = selector_range(s);
    if lo == hi {
        out.copy_from_slice(&[e.pixel(lo)[0], 0x1d, 0x92, 0x49, 0x24, 0x92, 0x49, 0x24]);
        return;
    }
    let r = range_index(lo, hi);
    let t = EAC[(e.color[0] + e.intensity * 32) * 4 + if r < 4 { r } else { 0 }];
    out[0] = t[0] as u8;
    out[1] = ((t[1] >> 4) | ((t[1] & 15) << 4)) as u8;
    let mut bits = 0u64;
    for y in 0..4 {
        for x in 0..4 {
            bits |= u64::from((t[2] >> (s.get(y * 4 + x) * 3)) & 7) << (45 - (y + x * 4) * 3);
        }
    }
    out[2..].copy_from_slice(&bits.to_be_bytes()[2..]);
}
fn set(out: &mut [u8], at: usize, count: usize, value: usize) {
    for i in 0..count {
        out[(at + i) / 8] = (out[(at + i) / 8] & !(1 << ((at + i) % 8)))
            | ((((value >> i) & 1) as u8) << ((at + i) % 8));
    }
}
fn get(out: &[u8], at: usize, count: usize) -> usize {
    (0..count).fold(0, |v, i| {
        v | (usize::from((out[(at + i) / 8] >> ((at + i) % 8)) & 1) << i)
    })
}
fn pack_color(out: &mut [u8], mut low: [usize; 3], mut high: [usize; 3], weights: &[usize; 16]) {
    out.fill(0);
    set(out, 0, 8, 32);
    let inv = if weights[0] & 2 != 0 {
        std::mem::swap(&mut low, &mut high);
        3
    } else {
        0
    };
    for c in 0..3 {
        set(out, 8 + c * 14, 7, low[c]);
        set(out, 15 + c * 14, 7, high[c]);
    }
    set(out, 50, 8, 255);
    set(out, 58, 8, 255);
    pack_weights(out, 66, weights, inv);
}
fn pack_weights(out: &mut [u8], mut at: usize, weights: &[usize; 16], inv: usize) {
    for (i, &s) in weights.iter().enumerate() {
        let n = if i == 0 { 1 } else { 2 };
        set(out, at, n, s ^ inv);
        at += n;
    }
}
pub(super) fn bc7_color(out: &mut [u8], e: Endpoint, s: &Selector) {
    let (lo, hi, unique) = selector_range(s);
    if unique == 1 {
        let p = e.pixel(lo).map(usize::from);
        pack_color(
            out,
            p.map(|v| BC7_SOLID[v][1] as usize),
            p.map(|v| BC7_SOLID[v][0] as usize),
            &[1; 16],
        );
    } else if unique == 2 {
        pack_color(
            out,
            e.pixel(lo).map(|v| usize::from(v >> 1)),
            e.pixel(hi).map(|v| usize::from(v >> 1)),
            &s.map(|v| if v == lo { 0 } else { 3 }),
        );
    } else {
        let r = range_index(lo, hi);
        let offsets = e.color.map(|c| (e.intensity * 32 + c) * 60 + r * 10);
        let best = (0..10)
            .min_by_key(|&m| {
                offsets
                    .iter()
                    .map(|&o| u32::from(BC7_COLOR[o + m][2]))
                    .sum::<u32>()
            })
            .unwrap();
        let mappings = [
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
            [0, 0, 2, 3],
            [0, 1, 1, 1],
            [0, 1, 2, 2],
            [0, 1, 2, 3],
            [0, 2, 3, 3],
            [1, 2, 2, 2],
            [1, 2, 3, 3],
        ];
        pack_color(
            out,
            offsets.map(|o| BC7_COLOR[o + best][0] as usize),
            offsets.map(|o| BC7_COLOR[o + best][1] as usize),
            &s.map(|v| mappings[best][v]),
        );
    }
}
pub(super) fn bc7_alpha(out: &mut [u8], e: Endpoint, s: &Selector) {
    let (lo, hi, unique) = selector_range(s);
    if unique == 1 {
        let v = usize::from(e.pixel(lo)[0]);
        set(out, 50, 8, v);
        set(out, 58, 8, v);
        return;
    }
    let (mut low, mut high, weights) = if unique == 2 {
        (
            usize::from(e.pixel(lo)[1]),
            usize::from(e.pixel(hi)[1]),
            s.map(|v| if v == lo { 0 } else { 3 }),
        )
    } else {
        let t = BC7_ALPHA[(e.intensity * 32 + e.color[0]) * 6 + range_index(lo, hi)];
        (
            t[0] as usize,
            t[1] as usize,
            s.map(|v| ((t[2] >> (v * 2)) & 3) as usize),
        )
    };
    let inv = if weights[0] & 2 != 0 {
        std::mem::swap(&mut low, &mut high);
        3
    } else {
        0
    };
    set(out, 50, 8, low);
    set(out, 58, 8, high);
    pack_weights(out, 97, &weights, inv);
}
fn expand(v: usize) -> i32 {
    ((v << 1) | (v >> 6)) as i32
}
fn interp(l: i32, h: i32, w: usize) -> i32 {
    (l * (64 - [0, 21, 43, 64][w]) + h * [0, 21, 43, 64][w] + 32) >> 6
}
fn cocg(e: Endpoint) -> [f32; 2] {
    let c = e.color.map(|v| ((v << 3) | (v >> 2)) as f32);
    [
        c[0] * 0.5 + c[2] * -0.5,
        c[0] * -0.25 + c[1] * 0.5 + c[2] * -0.25,
    ]
}
pub(super) fn chroma_filter(
    out: &mut [u8],
    nx: usize,
    ny: usize,
    blocks: &[(usize, usize)],
    endpoints: &[Endpoint],
) {
    let sample = |x: isize, y: isize| {
        cocg(
            endpoints[blocks
                [y.clamp(0, ny as isize - 1) as usize * nx + x.clamp(0, nx as isize - 1) as usize]
                .0],
        )
    };
    for by in 0..ny {
        for bx in 0..nx {
            let center = sample(bx as isize, by as isize);
            let filter = (-1..=1).any(|dy| {
                (-1..=1).any(|dx| {
                    let x = bx as isize + dx;
                    let y = by as isize + dy;
                    if x < 0 || y < 0 || x >= nx as isize || y >= ny as isize {
                        return false;
                    }
                    let n = sample(x, y);
                    (n[0] - center[0]).abs() > 10.0 || (n[1] - center[1]).abs() > 10.0
                })
            });
            if !filter {
                continue;
            }
            let out = &mut out[(by * nx + bx) * 16..(by * nx + bx + 1) * 16];
            let low = [0, 1, 2].map(|c| expand(get(out, 8 + c * 14, 7)));
            let high = [0, 1, 2].map(|c| expand(get(out, 15 + c * 14, 7)));
            let vals: [f32; 4] = std::array::from_fn(|w| {
                interp(low[0], high[0], w) as f32 * 0.25
                    + interp(low[1], high[1], w) as f32 * 0.5
                    + interp(low[2], high[2], w) as f32 * 0.25
            });
            let mut at = 66;
            let ys: [f32; 16] = std::array::from_fn(|i| {
                let n = if i == 0 { 1 } else { 2 };
                let v = vals[get(out, at, n)];
                at += n;
                v
            });
            let sum = ys.iter().sum::<f32>();
            let sq = ys.iter().map(|v| v * v).sum::<f32>();
            if sq * (1.0 / 16.0) - (sum * (1.0 / 16.0)) * (sum * (1.0 / 16.0)) < 3.0 {
                continue;
            }
            let pixels = std::array::from_fn(|i| {
                let x = (i % 4) as isize;
                let y = (i / 4) as isize;
                let ux = bx as isize + ((x - 2) >> 2);
                let uy = by as isize + ((y - 2) >> 2);
                let fx = (((x + 2) & 3) as f32 + 0.5) * 0.25;
                let fy = (((y + 2) & 3) as f32 + 0.5) * 0.25;
                let a = sample(ux, uy);
                let b = sample(ux + 1, uy);
                let c = sample(ux, uy + 1);
                let d = sample(ux + 1, uy + 1);
                let f: [f32; 2] = std::array::from_fn(|j| {
                    let ab = a[j] + (b[j] - a[j]) * fx;
                    let cd = c[j] + (d[j] - c[j]) * fx;
                    ab + (cd - ab) * fy
                });
                [ys[i] + f[0] - f[1], ys[i] + f[1], ys[i] - f[0] - f[1]]
                    .map(|v| (0.5 + v.clamp(0.0, 255.0)) as i32)
            });
            encode_color(out, &pixels);
        }
    }
}
fn to7(c: f32) -> usize {
    let mut v = (c * 127.0) as usize;
    if v < 127 {
        let lo = expand(v) as f32 / 255.0;
        let hi = expand(v + 1) as f32 / 255.0;
        v += usize::from(c > (lo + hi) / 2.0);
    }
    v.min(127)
}
fn eval(p: &[[i32; 3]; 16], low: [usize; 3], high: [usize; 3]) -> [usize; 16] {
    let colors: [[i32; 3]; 4] = std::array::from_fn(|w| {
        std::array::from_fn(|c| interp(expand(low[c]), expand(high[c]), w))
    });
    let axis: [i32; 3] = std::array::from_fn(|c| colors[3][c] - colors[0][c]);
    let dot = |p: [i32; 3]| (0..3).map(|c| p[c] * axis[c]).sum::<i32>();
    let ds = colors.map(dot);
    p.map(|p| {
        let d = dot(p) * 2;
        usize::from(d > ds[0] + ds[1])
            + usize::from(d >= ds[1] + ds[2])
            + usize::from(d >= ds[2] + ds[3])
    })
}
fn lerp8(a: i32, b: i32, s: i32) -> i32 {
    let t = (b - a) * s + 128;
    a + ((t + (t >> 8)) >> 8)
}
fn encode_color(out: &mut [u8], p: &[[i32; 3]; 16]) {
    let total: [i32; 3] = std::array::from_fn(|c| p.iter().map(|v| v[c]).sum());
    let min: [i32; 3] = std::array::from_fn(|c| p.iter().map(|v| v[c]).min().unwrap());
    let max: [i32; 3] = std::array::from_fn(|c| p.iter().map(|v| v[c]).max().unwrap());
    if min == max {
        pack_color(
            out,
            min.map(|v| BC7_SOLID[v as usize][1] as usize),
            min.map(|v| BC7_SOLID[v as usize][0] as usize),
            &[1; 16],
        );
        return;
    }
    let mean = total.map(|v| (v + 8) >> 4);
    let mut cov = [0i32; 6];
    for pixel in p {
        let [r, g, b] = std::array::from_fn(|c| pixel[c] - mean[c]);
        for (a, v) in cov
            .iter_mut()
            .zip([r * r, r * g, r * b, g * g, g * b, b * b])
        {
            *a += v;
        }
    }
    let var = cov[0].max(cov[3]).max(cov[5]);
    let (mut low, mut high);
    if var < 160 {
        low = std::array::from_fn(|c| to7(lerp8(min[c], max[c], 16) as f32 * (1.0 / 255.0)));
        high = std::array::from_fn(|c| to7(lerp8(min[c], max[c], 239) as f32 * (1.0 / 255.0)));
        pack_color(out, low, high, &eval(p, low, high));
        return;
    }
    let cov = cov.map(|v| v as f32);
    let sc = 1.0 / var as f32;
    let [wx, wy, wz] = [sc * cov[0], sc * cov[3], sc * cov[5]];
    let alt = [
        cov[0] * wx + cov[1] * wy + cov[2] * wz,
        cov[1] * wx + cov[3] * wy + cov[4] * wz,
        cov[2] * wx + cov[4] * wy + cov[5] * wz,
    ];
    let k = alt[0].abs().max(alt[1].abs()).max(alt[2].abs());
    let axis = if k.abs() >= 0.0000125 {
        let m = 2048.0 / k;
        alt.map(|v| (v * m) as i32)
    } else {
        [306, 601, 117]
    }
    .map(|v| v << 4);
    let dots: [i32; 16] =
        std::array::from_fn(|i| ((0..3).map(|c| p[i][c] * axis[c]).sum::<i32>() & !15) + i as i32);
    low = p[(*dots.iter().min().unwrap() & 15) as usize].map(|v| to7(v as f32 * (1.0 / 255.0)));
    high = p[(*dots.iter().max().unwrap() & 15) as usize].map(|v| to7(v as f32 * (1.0 / 255.0)));
    let mut weights = eval(p, low, high);
    let mut uq = [0; 3];
    let mut accum = 0u32;
    for (p, &s) in p.iter().zip(&weights) {
        accum += [0x000009, 0x010204, 0x040201, 0x090000][s];
        for c in 0..3 {
            uq[c] += s as i32 * p[c];
        }
    }
    let q = std::array::from_fn::<_, 3, _>(|c| total[c] * 3 - uq[c]);
    let z00 = ((accum >> 16) & 255) as f32;
    let z10 = ((accum >> 8) & 255) as f32;
    let z11 = (accum & 255) as f32;
    let det = z00 * z11 - z10 * z10;
    if det.abs() >= 1e-8 {
        let det = (3.0 / 255.0) / det;
        let [iz00, iz01, iz10, iz11] = [z11 * det, -z10 * det, -z10 * det, z00 * det];
        low =
            std::array::from_fn(
                |c| to7((iz10 * uq[c] as f32 + iz11 * q[c] as f32).clamp(0.0, 1.0)),
            );
        high =
            std::array::from_fn(
                |c| to7((iz00 * uq[c] as f32 + iz01 * q[c] as f32).clamp(0.0, 1.0)),
            );
        weights = eval(p, low, high);
    }
    pack_color(out, low, high, &weights);
}
