// The vertex-stage displacement sampler: the GPU-side counterpart to `reconstruction.ts`'s
// `bicubicSample`, reading a cascade's published displacement texture (`ocean.ts`'s `displace0`/
// `displace1`) at an off-grid world position and displacing a clipmap mesh vertex by the sum.
//
// Bicubic Catmull-Rom (C1) is used rather than hardware bilinear filtering (C0) because a mesh
// vertex reads the field at an arbitrary sub-texel offset every frame — a C0 kernel's gradient
// discontinuity at each texel boundary shows up on a displaced mesh as a lattice of hard creases,
// not a smooth wave surface (`reconstruction.ts`'s header). `catmullRom1D`/`wrapIndex` below mirror
// `reconstruction.ts`'s plain-TS reference (same taps, same wrap, same coefficients), asserted in
// lockstep by `wrap-catmullrom-lockstep.test.ts`, which drives both from JS directly (no GPU dispatch
// — see that file's own header for why this is sound for exactly these two functions).
//
// `bicubicSample0`/`bicubicSample1` are closed over one displacement texture each (`displace0`/
// `displace1`) rather than taking a texture parameter — TGSL device functions don't take runtime
// texture handles — so each cascade gets its own copy of the same kernel body.
//
// `oceanDisplacementLayout` declares only what the vertex stage needs (instancing + both
// displacement textures). A later stage building the full surface (vs + fs together) either reuses
// this layout object directly (if its fs needs no further bindings) or declares its own consolidated
// layout carrying every binding both stages need, mirroring the same kernel bodies against that
// layout's own texture bindings — a TGSL closure is bound to the specific layout object it was
// authored against.

import { surfaceLayout, VsIn, vsPatchSchema } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { CASCADE_CONFIGS } from "./spectrum";

/** the vertex stage's own bindings: the Part-instancing convention (`eids` + `transforms`) plus
 *  both displacement cascades' published textures. A consumer combining this with a fragment stage
 *  extends this shape with its own bindings and re-declares the kernels below against that layout —
 *  see this file's header. */
export const oceanDisplacementLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    displace0: { type: "texture-2d" },
    displace1: { type: "texture-2d" },
});

/** the displaced world position rides as the surface's own varying: sear's built-in interstage
 *  `world` carries the *displaced* position, but a fragment stage that must re-resolve the field
 *  (e.g. to reconstruct a shading normal) needs the *un*-displaced grid-plane sample position, so
 *  that rides separately. */
export const oceanDisplacementVaryings = { samplePos: d.vec2f };
export const oceanDisplacementPatch = vsPatchSchema(oceanDisplacementVaryings);

/** wrap a possibly-negative texel index into `[0, n)` — WGSL's `%` truncates toward zero, so a
 *  naive `i % n` on a negative `i` (the `-1` tap) returns negative; the double-mod is exact for
 *  any `i` in a texture's practical index range. (`%` is native integer remainder, unaffected by
 *  the `u32/u32`-compiles-as-real-division hazard `idiv` guards against elsewhere in this package —
 *  this function takes signed `i32`.) Exported so `wrap-catmullrom-lockstep.test.ts` can drive it
 *  directly from JS (typegpu's CPU execution of a `tgpu.fn` free of transcendentals reproduces the
 *  WGSL runtime exactly — pure add/mul/select/mod, unlike `gpu-fft.ts`'s `twiddleAngle`, which needs
 *  a real device dispatch because `cos`/`sin` are hardware-approximated). */
export const wrapIndex = tgpu.fn(
    [d.i32, d.i32],
    d.i32,
)((i, n) => {
    "use gpu";
    return ((i % n) + n) % n;
});

/** one dimension of uniform Catmull-Rom (tau=0.5) over 4 control points `p0..p3` (samples at local
 *  offsets -1,0,1,2), interpolating between `p1` and `p2` at `t ∈ [0,1]`. C1 by construction — the
 *  standard cubic-convolution basis, matrix-form coefficients expanded and Horner-evaluated. Exact
 *  GPU mirror of `reconstruction.ts`'s `catmullRom1D`, asserted in lockstep by
 *  `wrap-catmullrom-lockstep.test.ts` — exported for the same reason `wrapIndex` is. */
export const catmullRom1D = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32],
    d.vec4f,
)((p0, p1, p2, p3, t) => {
    "use gpu";
    const a = std.add(
        std.add(std.mul(d.f32(-0.5), p0), std.mul(d.f32(1.5), p1)),
        std.add(std.mul(d.f32(-1.5), p2), std.mul(d.f32(0.5), p3)),
    );
    const b = std.add(
        std.add(p0, std.mul(d.f32(-2.5), p1)),
        std.add(std.mul(d.f32(2.0), p2), std.mul(d.f32(-0.5), p3)),
    );
    const c = std.add(std.mul(d.f32(-0.5), p0), std.mul(d.f32(0.5), p2));
    const inner1 = std.add(b, std.mul(a, t));
    const inner2 = std.add(c, std.mul(inner1, t));
    return std.add(p1, std.mul(inner2, t));
});

/**
 * bicubic Catmull-Rom over cascade 0's displacement texture — 16 wrapped `textureLoad`s (4 rows of
 * 4), a `catmullRom1D` per row, then one across the row results. `u`,`v` are UNNORMALIZED texel
 * coordinates: `(world/L + 0.5)*N - 0.5`, exactly `reconstruction.ts`'s `bicubicSample` convention,
 * so a CPU test can drive the same arithmetic against a plain-TS `Field`.
 */
const bicubicSample0 = tgpu.fn(
    [d.i32, d.f32, d.f32],
    d.vec4f,
)((N, u, v) => {
    "use gpu";
    const ix = d.i32(std.floor(u));
    const iy = d.i32(std.floor(v));
    const fx = u - std.floor(u);
    const fy = v - std.floor(v);
    const xn = wrapIndex(ix - 1, N);
    const x0 = wrapIndex(ix, N);
    const xp = wrapIndex(ix + 1, N);
    const xq = wrapIndex(ix + 2, N);
    const yn = wrapIndex(iy - 1, N);
    const y0 = wrapIndex(iy, N);
    const yp = wrapIndex(iy + 1, N);
    const yq = wrapIndex(iy + 2, N);
    const rowN = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xn, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(x0, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xp, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xq, yn), d.u32(0)),
        fx,
    );
    const row0 = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xn, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(x0, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xp, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xq, y0), d.u32(0)),
        fx,
    );
    const rowP = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xn, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(x0, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xp, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xq, yp), d.u32(0)),
        fx,
    );
    const rowQ = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xn, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(x0, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xp, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace0, d.vec2i(xq, yq), d.u32(0)),
        fx,
    );
    return catmullRom1D(rowN, row0, rowP, rowQ, fy);
});

/** cascade 1's counterpart to {@link bicubicSample0} — closed over `displace1`; same kernel, same
 *  texel-space `u`,`v` convention. */
const bicubicSample1 = tgpu.fn(
    [d.i32, d.f32, d.f32],
    d.vec4f,
)((N, u, v) => {
    "use gpu";
    const ix = d.i32(std.floor(u));
    const iy = d.i32(std.floor(v));
    const fx = u - std.floor(u);
    const fy = v - std.floor(v);
    const xn = wrapIndex(ix - 1, N);
    const x0 = wrapIndex(ix, N);
    const xp = wrapIndex(ix + 1, N);
    const xq = wrapIndex(ix + 2, N);
    const yn = wrapIndex(iy - 1, N);
    const y0 = wrapIndex(iy, N);
    const yp = wrapIndex(iy + 1, N);
    const yq = wrapIndex(iy + 2, N);
    const rowN = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xn, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(x0, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xp, yn), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xq, yn), d.u32(0)),
        fx,
    );
    const row0 = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xn, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(x0, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xp, y0), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xq, y0), d.u32(0)),
        fx,
    );
    const rowP = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xn, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(x0, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xp, yp), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xq, yp), d.u32(0)),
        fx,
    );
    const rowQ = catmullRom1D(
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xn, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(x0, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xp, yq), d.u32(0)),
        std.textureLoad(oceanDisplacementLayout.$.displace1, d.vec2i(xq, yq), d.u32(0)),
        fx,
    );
    return catmullRom1D(rowN, row0, rowP, rowQ, fy);
});

/** cascade 0/1's world-space patch length, baked at module load — `CASCADE_CONFIGS` is a fixed
 *  export, not a runtime-varying value, so the vertex stage reads it as a compile-time constant
 *  rather than through a params uniform (the spike this was ported from used a uniform; this port
 *  drops it as an unneeded indirection since the source of truth (`CASCADE_CONFIGS`) is already
 *  shared TS, not GPU state a frame can change). `L0`/`L1` cannot drift from `CASCADE_CONFIGS`: each
 *  is a direct read of that array's own `.L` field, never a second authored literal, so a change to
 *  `CASCADE_CONFIGS` moves both without a gate. A third cascade would need its own `L2`/`N2`/texture
 *  bindings and `bicubicSample2` — this file's hard-coded `[0]`/`[1]` indices assume exactly two,
 *  unenforced here. */
const L0 = d.f32(CASCADE_CONFIGS[0].L);
const L1 = d.f32(CASCADE_CONFIGS[1].L);

/**
 * displaces a clipmap mesh vertex by the sum of both cascades' bicubic-reconstructed displacement,
 * sampled at the vertex's grid-plane world position. `oceanDisplacementPatch.samplePos` carries the
 * un-displaced position through to the fragment stage, which needs it to re-resolve the field
 * (rather than the post-displacement world position `world` carries).
 */
export const oceanDisplacementVs = tgpu.fn(
    [VsIn],
    oceanDisplacementPatch,
)((input) => {
    "use gpu";
    const worldX = input.localPos.x;
    const worldZ = input.localPos.z;
    const dim0 = std.textureDimensions(oceanDisplacementLayout.$.displace0);
    const dim1 = std.textureDimensions(oceanDisplacementLayout.$.displace1);
    const N0 = d.i32(dim0.x);
    const N1 = d.i32(dim1.x);

    const u0 = std.mul(std.add(std.div(worldX, L0), d.f32(0.5)), d.f32(N0)) - d.f32(0.5);
    const v0 = std.mul(std.add(std.div(worldZ, L0), d.f32(0.5)), d.f32(N0)) - d.f32(0.5);
    const disp0 = bicubicSample0(N0, u0, v0);

    const u1 = std.mul(std.add(std.div(worldX, L1), d.f32(0.5)), d.f32(N1)) - d.f32(0.5);
    const v1 = std.mul(std.add(std.div(worldZ, L1), d.f32(0.5)), d.f32(N1)) - d.f32(0.5);
    const disp1 = bicubicSample1(N1, u1, v1);

    const disp = std.add(disp0, disp1);
    const world = d.vec4f(worldX + disp.x, disp.y, worldZ + disp.z, d.f32(1));

    return oceanDisplacementPatch({
        world,
        worldNormal: d.vec3f(0, 1, 0), // a fragment stage re-derives the shading normal per pixel
        clip: d.vec4f(0),
        samplePos: d.vec2f(worldX, worldZ),
    });
});
