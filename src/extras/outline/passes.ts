import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { decodePos, MeshQuant, meshIdOf } from "../../core/mesh";
import { ViewUniforms, Xform, xformPoint } from "../../core/rendering";

// Cap band width to bound the jump-flood pass count.
export const MAX_WIDTH = 64;

/** Golus's width-bounded ladder: ceil(log2(width + 1)) levels at the maximum width. */
export const MAX_JFA_PASSES = Math.ceil(Math.log2(MAX_WIDTH + 1));

/** The jump step of ladder slot `k`, halving to 1 in the last slot. */
export function jfaStep(k: number): number {
    return 2 ** (MAX_JFA_PASSES - 1 - k);
}

/**
 * Golus's linked JumpFloodOutlineRenderer uses ceil(log2(width + 1)) levels.
 * Their steps sum to 2^levels - 1 >= width, covering every integer offset in band range.
 * This bounds reach, not accuracy: competing seeds can still kill the nearest seed's paths.
 */
export function jfaPassCount(maxWidth: number): number {
    const width = Math.max(1, Math.min(MAX_WIDTH, maxWidth));
    return Math.ceil(Math.log2(width + 1));
}

/** the always-on-top mask layout. @internal */
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
 * the mask vs, over a specific mask layout (one authored kernel re-emitting per layout). Pulls the
 * highlighted instance's 8 B position stream, dequantizes against its meshId's {@link MeshQuant}, and
 * applies the `globalTransforms` rows ({@link xformPoint}). @internal
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
            return { pos: std.mul(layout.$.view.clipFromWorld, world), iid: input.iid };
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

/** the composite's I/O: the resolved scene, the final JFA field, the per-seed color/width,
 *  and the rgba16float scratch the band is written into.
 *  @internal */
export const compositeLayout = tgpu.bindGroupLayout({
    distance: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    scene: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    seed: { texture: d.texture2d(d.u32), visibility: ["compute"] },
    attr: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    output: {
        storageTexture: d.textureStorage2d("rgba16float", "write-only"),
        visibility: ["compute"],
    },
});

/**
 * Skip neighbors off the texture: an out-of-bounds `textureLoad` returns 0, which would read as a seed
 * at the corner. Offsets are `i32` because they go negative; the output is a `vec4<u32>` because typegpu types a
 * fragment target as a 4-component vector — the `rg16uint` attachment consumes `xy` and drops the rest.
 * @internal
 */
function jfaFragment(scale: number) {
    return tgpu.fragmentFn({
        in: { pos: d.builtin.position },
        out: d.vec4u,
    })((input) => {
        "use gpu";
        const dim = d.vec2i(std.textureDimensions(jfaLayout.$.seed));
        const p = d.vec2i(input.pos.xy);
        const here = std.add(std.mul(d.vec2f(p), scale), (scale - 1) * 0.5);
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
}
export const jfaFs = jfaFragment(1);
export const jfaHalfFs = jfaFragment(2);

export const reduceLayout = tgpu.bindGroupLayout({
    mask: { texture: d.texture2d(d.u32), visibility: ["fragment"] },
});

/** Preserve a full-resolution seed from each 2×2 footprint, nearest its center. */
export const reduceFs = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4u,
})((input) => {
    "use gpu";
    const base = std.mul(d.vec2i(input.pos.xy), 2);
    const dim = d.vec2i(std.textureDimensions(reduceLayout.$.mask));
    const center = std.add(d.vec2f(base), 0.5);
    let best = d.vec2u(30000);
    let bestD = d.f32(1e10);
    for (let y = d.i32(0); y < 2; y++) {
        for (let x = d.i32(0); x < 2; x++) {
            const p = std.add(base, d.vec2i(x, y));
            if (p.x >= dim.x || p.y >= dim.y) continue;
            const seed = std.textureLoad(reduceLayout.$.mask, p, 0).xy;
            const dist = std.distance(center, d.vec2f(seed));
            if (dist < bestD) {
                best = d.vec2u(seed);
                bestD = dist;
            }
        }
    }
    return d.vec4u(best, 0, 0);
});

export const distanceLayout = tgpu.bindGroupLayout({
    seed: { texture: d.texture2d(d.u32), visibility: ["fragment"] },
});
export const distanceFs = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
})((input) => {
    "use gpu";
    const p = d.vec2i(input.pos.xy);
    const seed = std.textureLoad(distanceLayout.$.seed, p, 0).xy;
    const distance = std.distance(std.add(std.mul(d.vec2f(p), 2), 0.5), d.vec2f(seed));
    // Distances are normal f16 values; round their 11 significant bits to nearest, ties to even.
    // The attachment conversion is then exact on every backend.
    const parts = std.frexp(distance);
    const rounded = std.ldexp(std.round(parts.fract * 2048), parts.exp - 11);
    return d.vec4f(rounded, 0, 0, 0);
});

/**
 * Blends the band over the premultiplied linear HDR scene by its coverage, keeping the scene's alpha
 * elsewhere, so a camera stacked over others covers the cameras below only with what it drew and its band.
 * Write every pixel, passing the scene through where there is no band: the ping-pong target is a separate
 * texture, so discarding would leave pixels unwritten for tonemapping.
 * @internal
 */
function composite(half: boolean) {
    return tgpu.computeFn({
        workgroupSize: [WORKGROUP, WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const dim = std.textureDimensions(compositeLayout.$.output);
        if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
        const p = d.vec2i(input.gid.xy);
        const scene = std.textureLoad(compositeLayout.$.scene, p, 0);
        const q = half ? d.vec2i(std.div(d.vec2f(p), 2)) : d.vec2i(p);
        const s = std.textureLoad(compositeLayout.$.seed, q, 0).xy;
        let dist = std.distance(d.vec2f(p), d.vec2f(s));
        if (half) {
            const fieldSize = d.vec2i(std.textureDimensions(compositeLayout.$.distance));
            const uv = std.sub(std.div(std.add(d.vec2f(p), 0.5), 2), 0.5);
            const base = d.vec2i(std.floor(uv));
            const f = std.fract(uv);
            const a = std.clamp(base, d.vec2i(0), std.sub(fieldSize, 1));
            const b = std.clamp(std.add(base, 1), d.vec2i(0), std.sub(fieldSize, 1));
            // Explicit f32 interpolation avoids hardware half-precision filtering changing the band coverage.
            dist = std.mix(
                std.mix(
                    std.textureLoad(compositeLayout.$.distance, a, 0).x,
                    std.textureLoad(compositeLayout.$.distance, d.vec2i(b.x, a.y), 0).x,
                    f.x,
                ),
                std.mix(
                    std.textureLoad(compositeLayout.$.distance, d.vec2i(a.x, b.y), 0).x,
                    std.textureLoad(compositeLayout.$.distance, b, 0).x,
                    f.x,
                ),
                f.y,
            );
            if (std.textureLoad(compositeLayout.$.attr, p, 0).w > 0) dist = 0;
        }
        // out-of-bounds (sentinel seed) reads 0 → width 0 → no band
        const a = std.textureLoad(compositeLayout.$.attr, d.vec2i(s), 0);
        const width = a.w;
        const alpha = std.smoothstep(0, 1, dist) * (1 - std.smoothstep(width - 1, width, dist));
        const band = std.select(0, alpha, width > 0);
        std.textureStore(
            compositeLayout.$.output,
            p,
            d.vec4f(std.mix(scene.xyz, a.xyz, band), scene.w + band * (1 - scene.w)),
        );
    });
}
export const compositeKernel = composite(false);
export const compositeHalfKernel = composite(true);
