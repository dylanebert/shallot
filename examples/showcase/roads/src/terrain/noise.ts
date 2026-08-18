// The pure terrain-height primitives: the seeded permutation table + the layered 2D perlin/fbm WGSL —
// the same technique voxel's `noise.ts` uses for its heightmap generator, reused here directly on world
// (x, z) metres instead of voxel grid cells. No engine/GPU imports, so `bun test` exercises the
// determinism foundation device-free — the GPU dispatch that consumes this lives in generate.ts.
//
// Showcase examples don't import each other's `src/` — each is a self-contained project — so this is
// its own copy of the perlin/fbm/permutation shape, not a shared import from voxel.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

// heightmap knobs, in world units (metres), since the terrain kernel samples world position directly —
// unlike voxel, which samples grid-cell coordinates. HFREQ's period (1/HFREQ ≈ 330 m) puts about three
// broad hills across the 1024 m field (grid.ts); RELIEF is the vertical amplitude in metres (|fbm2| ≤ 1
// ⇒ height ∈ GROUND_LEVEL ± RELIEF) — gentle rolling hills, not mountains, the showcase's "readable at
// street level" brief.
export const HFREQ = 0.003;
export const RELIEF = 40;
const OCTAVES = 5; // baked into fbm2 below — broad shapes + medium hills + fine detail
const PERSISTENCE = d.f32(0.5); // each octave's amplitude vs the last
const LACUNARITY = d.f32(2.0); // each octave's frequency vs the last
export const GROUND_LEVEL = 0; // terrain centred on world Y=0 (the grid's own XZ centring, grid.ts)

const PERM_SIZE = 256;
const SQRT1_2 = d.f32(Math.SQRT1_2);
const NEG_SQRT1_2 = d.f32(-Math.SQRT1_2);
export const PermData = d.arrayOf(d.u32, PERM_SIZE * 2);

export const noiseLayout = tgpu.bindGroupLayout({
    perm: { storage: PermData, access: "readonly" },
});

/** the seeded RNG both the permutation table (below) and the procedural network generator
 *  (`overlay/network.ts`) share — one source, not two independently authored copies, since both are
 *  deterministic-in-seed data generators within this one project (the no-cross-import rule between
 *  showcase projects is about *other* projects, not modules inside this one). */
export function mulberry32(seed: number): () => number {
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

/**
 * TGSL mirror of the 2D improved-noise gradient lattice. The binding comes from {@link noiseLayout},
 * so the generator can resolve the noise helpers through the same schema-backed layout.
 */
export const grad2 = tgpu.fn(
    [d.u32, d.f32, d.f32],
    d.f32,
)((hash, x, y) => {
    "use gpu";
    const h = hash & d.u32(7);
    let gx = d.f32(0.0);
    let gy = d.f32(0.0);
    if (h === d.u32(0)) {
        gx = d.f32(1.0);
        gy = d.f32(0.0);
    }
    if (h === d.u32(1)) {
        gx = d.f32(-1.0);
        gy = d.f32(0.0);
    }
    if (h === d.u32(2)) {
        gx = d.f32(0.0);
        gy = d.f32(1.0);
    }
    if (h === d.u32(3)) {
        gx = d.f32(0.0);
        gy = d.f32(-1.0);
    }
    if (h === d.u32(4)) {
        gx = SQRT1_2;
        gy = SQRT1_2;
    }
    if (h === d.u32(5)) {
        gx = NEG_SQRT1_2;
        gy = SQRT1_2;
    }
    if (h === d.u32(6)) {
        gx = SQRT1_2;
        gy = NEG_SQRT1_2;
    }
    if (h === d.u32(7)) {
        gx = NEG_SQRT1_2;
        gy = NEG_SQRT1_2;
    }
    return gx * x + gy * y;
});

export const perlin2 = tgpu.fn(
    [d.vec2f],
    d.f32,
)((p) => {
    "use gpu";
    const x = p.x;
    const y = p.y;
    const fx = std.floor(x);
    const fy = std.floor(y);
    const X = d.u32(d.i32(fx) & d.i32(255));
    const Y = d.u32(d.i32(fy) & d.i32(255));
    const xf = x - fx;
    const yf = y - fy;
    const u = xf * xf * xf * (xf * (xf * d.f32(6.0) - d.f32(15.0)) + d.f32(10.0));
    const v = yf * yf * yf * (yf * (yf * d.f32(6.0) - d.f32(15.0)) + d.f32(10.0));
    const perm = noiseLayout.$.perm;
    const A = perm[X] + Y;
    const B = perm[X + d.u32(1)] + Y;
    const v00 = grad2(perm[A], xf, yf);
    const v10 = grad2(perm[B], xf - d.f32(1.0), yf);
    const v01 = grad2(perm[A + d.u32(1)], xf, yf - d.f32(1.0));
    const v11 = grad2(perm[B + d.u32(1)], xf - d.f32(1.0), yf - d.f32(1.0));
    return std.mix(std.mix(v00, v10, u), std.mix(v01, v11, u), v);
});

export const fbm2 = tgpu.fn(
    [d.vec2f],
    d.f32,
)((p) => {
    "use gpu";
    let amp = d.f32(1.0);
    let freq = d.f32(1.0);
    let sum = d.f32(0.0);
    let norm = d.f32(0.0);
    let i = d.u32(0);
    for (; i < d.u32(OCTAVES); i = i + d.u32(1)) {
        sum = sum + perlin2(std.mul(p, freq)) * amp;
        norm = norm + amp;
        amp = amp * PERSISTENCE;
        freq = freq * LACUNARITY;
    }
    return sum / norm;
});

/** height at world (x, z): `GROUND_LEVEL + fbm2((x, z) · HFREQ) · RELIEF` — the terrain's own surface
 *  function, sampled by the generator kernel both at a vertex and at its four neighbours (for the
 *  finite-difference normal). Bound through {@link noiseLayout} like {@link perlin2}. */
export const heightAt = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((x, z) => {
    "use gpu";
    return d.f32(GROUND_LEVEL) + fbm2(std.mul(d.vec2f(x, z), d.f32(HFREQ))) * d.f32(RELIEF);
});

/** the emitted perlin/fbm/height WGSL — the device-free structural seam the terrain tests resolve. */
export function noiseWgsl(): string {
    return tgpu.resolve([grad2, perlin2, fbm2, heightAt], { names: "strict" });
}
