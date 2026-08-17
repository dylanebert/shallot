import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { decodePos, MeshQuant, meshIdOf, Xform, xformPoint } from "../../engine/utils/core";
import { View } from "../../standard/render/core";

// The outline's pass internals: the JFA + composite kernels with their bind group layouts, plus the pure
// CPU logic the three passes build on (the JFA step ladder, the mesh-group batching for the scoped mask
// draw, and the mask shader codegen). Split off the barrel (orbit's smooth.ts shape) — index.ts keeps the
// component / system / plugin / per-camera resource management, this file the shader half.

// px cap on the band width — bounds the JFA pass count (`ceil(log2(width))`) and the attr texture's
// width channel. 64px is a generous outline; wider would want a coarser/clamped field anyway
export const MAX_WIDTH = 64;

/**
 * the jump-flood step ladder for a band of `maxWidth` pixels: the largest power of two ≥ the width,
 * halving to 1. Bounding the first jump by the width (not the screen) is what makes the pass count
 * `ceil(log2(width))`: a pixel only needs to find seeds within `width`, so seeds nearer than the start
 * step resolve correctly. Pure.
 */
export function jfaSteps(maxWidth: number): number[] {
    const w = Math.max(1, Math.min(MAX_WIDTH, Math.ceil(maxWidth)));
    let s = 1;
    while (s < w) s <<= 1;
    const steps: number[] = [];
    for (; s >= 1; s >>= 1) steps.push(s);
    return steps;
}

/**
 * group highlighted eids by their mesh id, preserving insertion order. The mask draws one instanced
 * call per mesh: each group's eids become a contiguous instance slice. Pure.
 */
export function groupByMesh(
    eids: number[],
    meshOf: (eid: number) => number,
): Map<number, number[]> {
    const groups = new Map<number, number[]>();
    for (const eid of eids) {
        const m = meshOf(eid);
        const g = groups.get(m);
        if (g) g.push(eid);
        else groups.set(m, [eid]);
    }
    return groups;
}

/** the always-on-top mask layout: view/position/indices/transforms/maskEids/maskAttrs/meshQuant.
 *  @internal */
export const maskLayoutPlain = tgpu.bindGroupLayout({
    view: { uniform: View, visibility: ["vertex"] },
    position: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["vertex"] },
    indices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    transforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
    maskEids: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    maskAttrs: { storage: d.arrayOf(d.vec4f), access: "readonly", visibility: ["fragment"] },
    meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
});

/** the occlusion-aware mask layout: {@link maskLayoutPlain}'s entries (duplicated, not spread — the
 *  visibility-array literal types don't survive a factory-returned intermediate) plus sear's `view.depth`
 *  lane, so the fs can discard a fragment behind the visible scene. A second layout, not a 1×1 dummy depth
 *  on the plain one: an out-of-bounds `textureLoad` returns 0 = far under reverse-Z, which would silently
 *  read every fragment as un-occluded. @internal */
export const maskLayoutOcclude = tgpu.bindGroupLayout({
    view: { uniform: View, visibility: ["vertex"] },
    position: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["vertex"] },
    indices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    transforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
    maskEids: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    maskAttrs: { storage: d.arrayOf(d.vec4f), access: "readonly", visibility: ["fragment"] },
    meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
    sceneDepth: { texture: d.textureDepth2d(), visibility: ["fragment"] },
});

type MaskLayout = typeof maskLayoutPlain | typeof maskLayoutOcclude;

/**
 * the mask vs, over a specific mask layout (plain or occlude — one authored
 * kernel re-emitting per layout). Pulls the highlighted instance's 8 B position stream, dequantizes against
 * its meshId's {@link MeshQuant}, and applies the `transforms` firehose ({@link xformPoint}). @internal
 */
export function maskVertex(layout: MaskLayout) {
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position, iid: d.interpolate("flat", d.u32) },
        })((input) => {
            "use gpu";
            const raw = layout.$.position[layout.$.indices[input.vidx]];
            // the array-element reads are pointers, not copies — wrap in the element schema before
            // passing them by value into decodePos / xformPoint
            const quant = MeshQuant(layout.$.meshQuant[meshIdOf(raw.y)]);
            const p = decodePos(raw.x, raw.y, quant);
            const x = Xform(layout.$.transforms[layout.$.maskEids[input.iid]]);
            const world = d.vec4f(xformPoint(x, p), 1);
            return { pos: std.mul(layout.$.view.viewProj, world), iid: input.iid };
        })
        .$name(layout === maskLayoutOcclude ? "maskVsOcclude" : "maskVs");
}

/**
 * the mask fs, over a specific mask layout: writes the pixel's own coordinate as the JFA seed
 * (`typegpu` types a fragment target as a 4-component vector — the `rg16uint` attachment consumes `xy`) plus
 * the instance's color/width into the attr texture. The occlude variant discards a fragment behind the
 * visible scene (reverse-Z: occluded ⇔ its depth is *less* than the nearest stored scene depth).
 * @internal
 */
export function maskFragment(layout: MaskLayout, occlude: boolean) {
    return tgpu
        .fragmentFn({
            in: { clip: d.builtin.position, iid: d.interpolate("flat", d.u32) },
            out: { seed: d.vec4u, attr: d.vec4f },
        })((input) => {
            "use gpu";
            const color = layout.$.maskAttrs[input.iid * 2];
            const params = layout.$.maskAttrs[input.iid * 2 + 1]; // (width, occlude, _, _)
            // `occlude` is a captured JS boolean (the factory's own argument, not a GPU value) — the
            // transpiler folds this branch at build time exactly like the pre-port `${occlude ? … : ""}`
            // template did, so the plain variant never even sees the depth read
            if (occlude) {
                if (params.y > 0.5) {
                    const scene = std.textureLoad(
                        (layout as typeof maskLayoutOcclude).$.sceneDepth,
                        d.vec2i(input.clip.xy),
                        0,
                    );
                    if (input.clip.z < scene - 1e-4) std.discard();
                }
            }
            return {
                seed: d.vec4u(d.vec2u(input.clip.xy), 0, 0),
                attr: d.vec4f(color.xyz, params.x),
            };
        })
        .$name(occlude ? "maskFsOcclude" : "maskFs");
}

/** the emitted mask WGSL for one variant — the device-free structural seam the outline test resolves.
 *  @internal */
export function maskWgsl(occlude: boolean): string {
    const layout = occlude ? maskLayoutOcclude : maskLayoutPlain;
    return tgpu.resolve([maskVertex(layout), maskFragment(layout, occlude)], { names: "strict" });
}

/** compute workgroup tile — 8×8 = 64 threads, matching glaze/fog's screen-space composite @internal */
export const WORKGROUP = 8;

/** the JFA flood pass's I/O: the seed field it reads (ping-ponged per pass) and this pass's jump step. The
 *  step is a bare `f32` uniform — the pre-port wrapped it in a one-field `Step` struct only because a raw
 *  layout entry needed a `minBindingSize`; a typed layout takes the scalar directly. `step` collides with the
 *  WGSL builtin, so it emits as `step_1` under strict naming — inert while nothing splices this pass by name,
 *  but rename the key before making it relocatable.
 *  @internal */
export const jfaLayout = tgpu.bindGroupLayout({
    seed: { texture: d.texture2d(d.u32), visibility: ["fragment"] },
    step: { uniform: d.f32, visibility: ["fragment"] },
});

/** the composite's I/O: the resolved scene (float — the `rg11b10ufloat` offscreen and the `rgba16float`
 *  scratch alike), the final JFA field, the per-seed color/width, and the scratch the band is written into.
 *  @internal */
export const compositeLayout = tgpu.bindGroupLayout({
    scene: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    seed: { texture: d.texture2d(d.u32), visibility: ["compute"] },
    attr: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    output: {
        storageTexture: d.textureStorage2d("rgba16float", "write-only"),
        visibility: ["compute"],
    },
});

/** the fullscreen triangle every screen-space pass draws: three vertices covering the viewport, no vertex
 *  buffer, no interpolators past the clip position.
 *  @internal */
export const fullscreenVs = tgpu.vertexFn({
    in: { i: d.builtin.vertexIndex },
    out: { pos: d.builtin.position },
})((input) => {
    "use gpu";
    const p = d.vec2f(d.f32((input.i << 1) & 2), d.f32(input.i & 2));
    return { pos: d.vec4f(p.x * 2 - 1, p.y * 2 - 1, 0, 1) };
});

/**
 * one jump-flood iteration: for the 9 neighbours at the current step, keep the seed coordinate nearest this
 * pixel. Bounds-guarded — an out-of-bounds `textureLoad` returns 0, which would read as a seed at the
 * corner, so skip neighbours off the texture. The sentinel seed loses naturally (its distance is huge).
 * The 3×3 offsets are `i32` (they go negative); the emitted output is a `vec4<u32>` because typegpu types a
 * fragment target as a 4-component vector — the `rg16uint` attachment consumes `xy` and drops the rest.
 * @internal
 */
export const jfaFs = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4u,
})((input) => {
    "use gpu";
    const dim = d.vec2i(std.textureDimensions(jfaLayout.$.seed));
    const p = d.vec2i(input.pos.xy);
    const here = d.vec2f(p);
    let best = std.textureLoad(jfaLayout.$.seed, p, 0).xy;
    let bestD = std.distance(here, d.vec2f(best));
    const s = d.i32(jfaLayout.$.step);
    for (let oy = d.i32(-1); oy <= 1; oy = oy + 1) {
        for (let ox = d.i32(-1); ox <= 1; ox = ox + 1) {
            const q = std.add(p, std.mul(d.vec2i(ox, oy), s));
            if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) continue;
            const cand = std.textureLoad(jfaLayout.$.seed, q, 0).xy;
            const dist = std.distance(here, d.vec2f(cand));
            if (dist < bestD) {
                bestD = dist;
                // the copy constructor is required: TGSL rejects a bare vector-to-vector assignment
                best = d.vec2u(cand);
            }
        }
    }
    return d.vec4u(best.x, best.y, 0, 0);
});

/**
 * the composite, a compute dispatch through the sceneTransform seam: each pixel's distance to its nearest
 * seed gives the band. `smoothstep(0,1,d)` fades the outline in 1px outside the silhouette (interior d≈0 →
 * alpha 0, the object keeps its color), the outer `1 - smoothstep(width-1,width,d)` fades it out at the band
 * edge — uniform width, antialiased. The seed's color/width come from the attr texture at the resolved seed
 * coordinate. `mix(scene, band, alpha)` is straight (non-premultiplied) alpha `over`, in linear scene space
 * (the offscreen + scratch are linear HDR) — the analytic twin of the old hardware ALPHA_BLEND. It writes
 * *every* pixel (scene-through where there's no band) because the ping-pong target is a separate texture, so
 * there's no in-place `discard`; glaze tonemaps the result.
 * @internal
 */
export const compositeKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const dim = std.textureDimensions(compositeLayout.$.output);
    if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
    const p = d.vec2i(input.gid.xy);
    const base = std.textureLoad(compositeLayout.$.scene, p, 0).xyz;
    const s = std.textureLoad(compositeLayout.$.seed, p, 0).xy;
    const dist = std.distance(d.vec2f(p), d.vec2f(s));
    // out-of-bounds (sentinel seed) reads 0 → width 0 → no band
    const a = std.textureLoad(compositeLayout.$.attr, d.vec2i(s), 0);
    const width = a.w;
    const alpha = std.smoothstep(0, 1, dist) * (1 - std.smoothstep(width - 1, width, dist));
    const band = std.select(0, alpha, width > 0);
    std.textureStore(compositeLayout.$.output, p, d.vec4f(std.mix(base, a.xyz, band), 1));
});

/** the emitted JFA + composite WGSL — the device-free structural seam the outline tests resolve.
 *  @internal */
export function outlineWgsl(): { jfa: string; composite: string } {
    return {
        jfa: tgpu.resolve([fullscreenVs, jfaFs], { names: "strict" }),
        composite: tgpu.resolve([compositeKernel], { names: "strict" }),
    };
}
