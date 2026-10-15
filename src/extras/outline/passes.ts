import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { decodePos, MeshQuant, meshIdOf } from "../../core/mesh";
import { ViewUniforms, Xform, xformPoint } from "../../core/rendering";

// Cap band width to bound the jump-flood pass count.
export const MAX_WIDTH = 64;

/** the JFA ladder's slot count: one per power of two from the first ≥ MAX_WIDTH down to 1. */
export const MAX_JFA_PASSES = Math.ceil(Math.log2(MAX_WIDTH)) + 1;

/** the jump step of ladder slot `k`: MAX_JFA_PASSES slots halving from the first power of two ≥ MAX_WIDTH to 1. */
export function jfaStep(k: number): number {
    return 2 ** (MAX_JFA_PASSES - 1 - k);
}

/**
 * the jump-flood pass count for a band of `maxWidth` pixels: the ladder's last `ceil(log2(width)) + 1`
 * slots, from the first power of two ≥ the width down to 1. Bounding the first jump by the width (not the
 * screen) bounds the pass count: a pixel only needs to find seeds within `width`, so seeds nearer than the
 * start step resolve correctly.
 */
export function jfaPassCount(maxWidth: number): number {
    const w = Math.max(1, Math.min(MAX_WIDTH, Math.ceil(maxWidth)));
    let passes = 1;
    for (let s = 1; s < w; s <<= 1) passes++;
    return passes;
}

/** the always-on-top mask layout: view/position/indices/globalTransforms/maskEids/maskAttrs/meshQuant.
 *  @internal */
export const maskLayoutPlain = tgpu.bindGroupLayout({
    globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    view: { uniform: ViewUniforms, visibility: ["vertex"] },
    position: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["vertex"] },
    indices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    globalTransforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
    maskEids: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    maskAttrs: { storage: d.arrayOf(d.vec4f), access: "readonly", visibility: ["fragment"] },
    meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
});

/** Occlusion-aware mask using the shared depth lane.
 * Visibility-array literal types don't survive a factory-returned intermediate, so entries are duplicated.
 * @internal */
export const maskLayoutOcclude = tgpu.bindGroupLayout({
    globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    view: { uniform: ViewUniforms, visibility: ["vertex"] },
    position: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["vertex"] },
    indices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    globalTransforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
    maskEids: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    maskAttrs: { storage: d.arrayOf(d.vec4f), access: "readonly", visibility: ["fragment"] },
    meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
    sceneDepth: { texture: d.textureDepth2d(), visibility: ["fragment"] },
});

/** MSAA occlusion variant, loading sample zero from the stored depth lane. */
export const maskLayoutOccludeMultisampled = tgpu.bindGroupLayout({
    globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    view: { uniform: ViewUniforms, visibility: ["vertex"] },
    position: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["vertex"] },
    indices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    globalTransforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
    maskEids: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["vertex"] },
    maskAttrs: { storage: d.arrayOf(d.vec4f), access: "readonly", visibility: ["fragment"] },
    meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
    sceneDepth: { texture: d.textureDepthMultisampled2d(), visibility: ["fragment"] },
});

type MaskLayout =
    | typeof maskLayoutPlain
    | typeof maskLayoutOcclude
    | typeof maskLayoutOccludeMultisampled;

/**
 * the mask vs, over a specific mask layout (plain or occlude — one authored
 * kernel re-emitting per layout). Pulls the highlighted instance's 8 B position stream, dequantizes against
 * its meshId's {@link MeshQuant}, and applies the `globalTransforms` rows ({@link xformPoint}). @internal
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
            const encodedRow = layout.$.globalTransformRows[layout.$.maskEids[input.iid]];
            if (encodedRow === 0) return { pos: d.vec4f(2, 2, 2, 1), iid: input.iid };
            const x = Xform(layout.$.globalTransforms[encodedRow - 1]);
            const world = d.vec4f(xformPoint(x, p), 1);
            return { pos: std.mul(layout.$.view.viewProj, world), iid: input.iid };
        })
        .$name(layout === maskLayoutPlain ? "maskVs" : "maskVsOcclude");
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
            // The captured host boolean folds at build time, keeping depth reads out of the plain variant.
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

/** Compute workgroup tile side length. @internal */
export const WORKGROUP = 8;

/** Ping-ponged seed field and jump step.
 * `step` collides with the WGSL builtin, so it emits as `step_1` under strict naming;
 * rename the key before making this pass relocatable. @internal */
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
 * (the offscreen + scratch are linear HDR). It writes
 * every pixel (scene-through where there's no band) because the ping-pong target is a separate texture, so
 * there's no in-place `discard`; tonemapping reads the result.
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
