// The pure terrain primitives: the seeded permutation table, the layered 2D perlin/fbm WGSL, the heightmap
// knobs, and the derived solid-fraction band. No engine/GPU imports (only grid addressing constants), so
// `bun test` exercises the determinism foundation device-free — the GPU dispatch that consumes these lives
// in generate.ts.
//
// Shape: orrstead's `generation/noise.ts` — a doubled-permutation table + an FBM WGSL chunk. The terrain is
// a heightmap: a zero-mean multi-octave 2D perlin field (`fbm2`) lifts and drops the ground into rolling
// hills (`surface = GROUND_LEVEL + fbm2(x,z)·RELIEF`), the textbook layered-perlin landscape. The grid
// stays a full 3D density field — the carve brush sculpts overhangs by hand — but the initial generation is
// a clean heightmap, not isotropic 3D noise (which read as busy noise pockets, not landform).

import { DIM } from "./grid";

// heightmap knobs — design constants tuned by hot-reload, not runtime uniforms. The solid-fraction band
// derives from them, so they're the single source for both the shader and the gate.
export const HFREQ = 0.012; // heightmap lattice frequency in cells⁻¹ → ~3 broad hills across the 256 map
export const RELIEF = 56; // vertical amplitude of the hills in cells; |fbm2| ≤ 1 → surface ∈ GROUND ± RELIEF
const OCTAVES = 5; // baked into NOISE_WGSL below; broad shapes + medium hills + fine detail — the "layered"
const PERSISTENCE = 0.5; // each octave's amplitude vs the last → smaller detail rides the big shapes
const LACUNARITY = 2.0; // each octave's frequency vs the last
export const GROUND_LEVEL = 0.5 * DIM.y; // the mean surface height → terrain centred on the grid

const PERM_SIZE = 256;

function mulberry32(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** seeded length-512 doubled permutation table — Fisher-Yates over 0..255, concatenated with itself so the
 *  perlin lattice hashes (`perm[A + 1]`, up to index 511) never wrap. Deterministic in `seed`. */
export function makePermutation(seed: number): Uint32Array {
    const rng = mulberry32(seed);
    const base = new Uint32Array(PERM_SIZE);
    for (let i = 0; i < PERM_SIZE; i++) base[i] = i;
    for (let i = PERM_SIZE - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = base[i];
        base[i] = base[j];
        base[j] = tmp;
    }
    const perm = new Uint32Array(PERM_SIZE * 2);
    perm.set(base);
    perm.set(base, PERM_SIZE);
    return perm;
}

/** the solid-fraction band the heightmap guarantees regardless of the noise: each column is solid from the
 *  floor up to its surface, and the surface lies in `[GROUND_LEVEL − RELIEF, GROUND_LEVEL + RELIEF]`
 *  (since |fbm2| ≤ 1), so the mean solid fraction lies in `[(GROUND − RELIEF)/DIM.y, (GROUND + RELIEF)/DIM.y]`.
 *  A derived bound (not a tuned threshold): it brackets 0.5 (zero-mean field centred at GROUND) and rejects
 *  an all-air or all-solid generator. The relief gate (surface-height variance) rejects a flat one. */
export function solidFractionBand(): [number, number] {
    const lo = Math.max(0, GROUND_LEVEL - RELIEF) / DIM.y;
    const hi = Math.min(DIM.y, GROUND_LEVEL + RELIEF) / DIM.y;
    return [lo, hi];
}

/** declares `perlin2(x,y)` and `fbm2(p)`, reading a `perm` storage binding the caller declares at
 *  `@group(0) @binding(0)`. The fbm octave schedule bakes {@link OCTAVES}/{@link PERSISTENCE}/
 *  {@link LACUNARITY} as constants. */
export const NOISE_WGSL = /* wgsl */ `
const SQRT1_2: f32 = 0.7071067811865476;

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// the 8 unit gradient directions selected by the low 3 hash bits — Ken Perlin's 2D improved-noise gradients.
fn grad2(hash: u32, x: f32, y: f32) -> f32 {
    let h = hash & 7u;
    var gx: f32;
    var gy: f32;
    switch h {
        case 0u: { gx =  1.0; gy =  0.0; }
        case 1u: { gx = -1.0; gy =  0.0; }
        case 2u: { gx =  0.0; gy =  1.0; }
        case 3u: { gx =  0.0; gy = -1.0; }
        case 4u: { gx =  SQRT1_2; gy =  SQRT1_2; }
        case 5u: { gx = -SQRT1_2; gy =  SQRT1_2; }
        case 6u: { gx =  SQRT1_2; gy = -SQRT1_2; }
        default: { gx = -SQRT1_2; gy = -SQRT1_2; }
    }
    return gx * x + gy * y;
}

fn perlin2(x: f32, y: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    let X = u32(i32(fx) & 255);
    let Y = u32(i32(fy) & 255);
    let xf = x - fx;
    let yf = y - fy;
    let u = fade(xf);
    let v = fade(yf);
    let A = perm[X] + Y;
    let B = perm[X + 1u] + Y;
    let v00 = grad2(perm[A], xf, yf);
    let v10 = grad2(perm[B], xf - 1.0, yf);
    let v01 = grad2(perm[A + 1u], xf, yf - 1.0);
    let v11 = grad2(perm[B + 1u], xf - 1.0, yf - 1.0);
    return mix(mix(v00, v10, u), mix(v01, v11, u), v);
}

fn fbm2(p: vec2<f32>) -> f32 {
    var amp: f32 = 1.0;
    var freq: f32 = 1.0;
    var sum: f32 = 0.0;
    var norm: f32 = 0.0;
    for (var i: u32 = 0u; i < ${OCTAVES}u; i = i + 1u) {
        sum = sum + perlin2(p.x * freq, p.y * freq) * amp;
        norm = norm + amp;
        amp = amp * ${PERSISTENCE.toExponential()};
        freq = freq * ${LACUNARITY.toExponential()};
    }
    return sum / norm;
}
`;
