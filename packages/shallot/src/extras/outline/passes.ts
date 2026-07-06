import { POS_QUANT_WGSL, XFORM_WGSL } from "../../engine/utils/core";
import { VIEW_STRUCT_WGSL } from "../../standard/render/core";

// The pure CPU logic behind the outline's three passes: the JFA step ladder, the mesh-group batching for
// the scoped mask draw, and the mask shader codegen. Split off the barrel (orbit's smooth.ts shape) — used
// by index.ts and pinned by the unit test, never author surface.

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

/**
 * the JFA seed-mask shader for the outline pass. The vs pulls the highlighted instances' position stream +
 * applies the `transforms` firehose; the fs writes the pixel's own coordinate as the JFA seed + the
 * instance's color/width into the attr texture. `occlude` adds the depth gate: sample sear's `view.depth`
 * and discard fragments behind the visible scene (reverse-Z, an occluded fragment's depth is *less* than
 * the nearest scene depth), so an occluded object contributes no silhouette. Two pipeline variants, not a
 * 1×1 dummy depth: an out-of-bounds textureLoad returns 0 = far under reverse-Z, which would silently make
 * every fragment read as un-occluded. Pure codegen.
 */
export function maskCode(occlude: boolean): string {
    return /* wgsl */ `
${VIEW_STRUCT_WGSL}
${POS_QUANT_WGSL}
${XFORM_WGSL}
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> position: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> transforms: array<Xform>;
@group(0) @binding(4) var<storage, read> maskEids: array<u32>;
@group(0) @binding(5) var<storage, read> maskAttrs: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> meshQuant: array<MeshQuant>;
${occlude ? "@group(0) @binding(6) var sceneDepth: texture_depth_2d;" : ""}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) iid: u32,
}

// the mask only needs position — pull the 8 B position stream + dequantize against the meshId's MeshQuant
@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iid: u32) -> VOut {
    let raw = position[indices[vidx]];
    let p = decodePos(raw.x, raw.y, meshQuant[meshIdOf(raw.y)]);
    let eid = maskEids[iid];
    let world = vec4<f32>(xformPoint(transforms[eid], p), 1.0);
    var out: VOut;
    out.clip = view.viewProj * world;
    out.iid = iid;
    return out;
}

struct MaskOut {
    @location(0) seed: vec2<u32>,
    @location(1) attr: vec4<f32>,
}

@fragment
fn fs(in: VOut) -> MaskOut {
    let color = maskAttrs[in.iid * 2u];
    let params = maskAttrs[in.iid * 2u + 1u]; // (width, occlude, _, _)
${
    occlude
        ? `    if (params.y > 0.5) {
        let scene = textureLoad(sceneDepth, vec2<i32>(in.clip.xy), 0);
        // reverse-Z (near→1/far→0): the object is occluded where its depth is behind (less than) the
        // nearest scene depth, so discard the mask there
        if (in.clip.z < scene - 1e-4) { discard; }
    }`
        : ""
}
    var out: MaskOut;
    out.seed = vec2<u32>(in.clip.xy);
    out.attr = vec4<f32>(color.rgb, params.x);
    return out;
}
`;
}
