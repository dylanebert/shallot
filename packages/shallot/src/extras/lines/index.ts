// Lines — the kitchen debug-line producer. One shared segment buffer, two feeders: an immediate API
// (`segment` / `box` / `arrow`, appended and cleared each frame — the scale path) and the retained
// `Line` / `Arrow` components (declarative scene annotations, expanded into segments each frame).
// Everything draws as one instanced 6-vertex quad per segment, rendered as a sear `"alpha"` surface
// inside the color pass — translucent, depth-tested, depth-write off, no overlay pass. Screen-space
// constant-pixel width: the surface projects each segment's endpoints itself (sear's `screen` mode)
// and writes `clipPos`, expanding the quad by a pixel half-width read from `view.resolution`. Bevy's
// gizmo model; arrows are folded in (a shaft segment + segment-fletched head), no separate primitive.
// The segment staging + upload + immediate API live in `segments.ts`.

import type { Plugin, State, System } from "../../engine";
import { Compute, f32, formatHex, sparse, vec4 } from "../../engine";
import { packColor } from "../../engine/utils/core";
import { mesh, RenderPlugin } from "../../standard/render";
import { BeginFrameSystem, Draws, Meshes, Surfaces } from "../../standard/render/core";
import { PrepassSystem } from "../../standard/sear/core";
import { composeTransform, Transform, TransformsPlugin } from "../../standard/transforms";
import {
    disposeSegments,
    flushSegments,
    head,
    Lines,
    push,
    ready,
    resetCount,
    warmSegments,
} from "./segments";

export { arrow, box, segment } from "./segments";

/**
 * a debug line anchored to an entity, drawn from its {@link Transform} position along a world-rotated
 * offset. A retained scene annotation, expanded into one screen-space segment each frame
 *
 * @example
 * ```
 * <a line="offset: 0 1 0; thickness: 3; color: 0x44ff88" transform />
 * ```
 */
export const Line = {
    /** line vector from the entity in its local frame, rotated by the transform (`0 1 0` = one unit up) */
    offset: sparse(vec4),
    /** constant screen width in pixels */
    thickness: sparse(f32),
    /** hex sRGB color */
    color: sparse(f32),
    /** 0..1 opacity multiplier */
    opacity: sparse(f32),
    /** drawn when nonzero; set to 0 to hide without removing (edit-mode safe) */
    visible: sparse(f32),
};

/**
 * an arrowhead on a {@link Line}: four world-space fins (Bevy's fletched shape) at the line's endpoints.
 * Requires a {@link Line} on the same entity
 *
 * @example
 * ```
 * <a arrow="size: 1.5" line="offset: 2 0 0; color: 0xffcc00" transform />
 * ```
 */
export const Arrow = {
    /** a head at the start endpoint when nonzero */
    start: sparse(f32),
    /** a head at the end endpoint when nonzero */
    end: sparse(f32),
    /** head size relative to the shaft length */
    size: sparse(f32),
};

const SEGMENT_WGSL = /* wgsl */ `
struct Segment {
    a: vec3<f32>,
    width: f32,
    b: vec3<f32>,
    color: u32,
}

fn lineSrgbToLinear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}
`;

// localPos.xy carries the quad corner: x = t (0 start, 1 end), y = edge (-1, +1). The chunk projects
// both endpoints, near-plane-clips, then offsets the chosen endpoint perpendicular by a pixel half-width
// (constant-pixel: pixels → NDC is × 2/resolution). 1px of AA pad each side; a sub-pixel width clamps the
// geometry to 1px and fades the alpha to keep its energy. world = the chosen endpoint so the (unused,
// DCE'd) shadow sample reads a valid position
const LINE_VS = /* wgsl */ `
let seg = lineSegments[iid];
let t = localPos.x;
let edge = localPos.y;
let widthPx = seg.width;

var sClip = view.viewProj * vec4<f32>(seg.a, 1.0);
var eClip = view.viewProj * vec4<f32>(seg.b, 1.0);
let nearW = 1e-5;
if (sClip.w < nearW && eClip.w < nearW) {
    // both endpoints behind the camera — collapse offscreen (z < 0 clips)
    clipPos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    lineRgba = vec4<f32>(0.0);
    edgeDist = 0.0;
    halfPx = 0.0;
    world = vec4<f32>(seg.a, 1.0);
} else {
    if (sClip.w < nearW) {
        let k = (nearW - sClip.w) / (eClip.w - sClip.w);
        sClip = mix(sClip, eClip, k);
    } else if (eClip.w < nearW) {
        let k = (nearW - eClip.w) / (sClip.w - eClip.w);
        eClip = mix(eClip, sClip, k);
    }
    let sNdc = sClip.xy / sClip.w;
    let eNdc = eClip.xy / eClip.w;
    let res = view.resolution;
    let dirPx = (eNdc - sNdc) * res;
    let lenPx = length(dirPx);
    let dir = select(vec2<f32>(1.0, 0.0), dirPx / lenPx, lenPx > 1e-4);
    let perp = vec2<f32>(-dir.y, dir.x);
    let halfW = max(widthPx, 1.0) * 0.5;
    let total = halfW + 1.0;
    let useEnd = t > 0.5;
    let baseNdc = select(sNdc, eNdc, useEnd);
    let baseClip = select(sClip, eClip, useEnd);
    let ndc = baseNdc + perp * (edge * total) * 2.0 / res;
    clipPos = vec4<f32>(ndc, baseClip.z / baseClip.w, 1.0);
    let unp = unpack4x8unorm(seg.color);
    lineRgba = vec4<f32>(lineSrgbToLinear(unp.rgb), unp.a * min(widthPx, 1.0));
    edgeDist = edge * total;
    halfPx = halfW;
    world = vec4<f32>(select(seg.a, seg.b, useEnd), 1.0);
}
`;

// signed-distance edge AA: fade over one screen-space derivative either side of the half-width
const LINE_FS = /* wgsl */ `
let aa = 1.0 - smoothstep(halfPx - fwidth(edgeDist), halfPx + fwidth(edgeDist), abs(edgeDist));
col = vec4<f32>(lineRgba.rgb, lineRgba.a * aa);
`;

// the canonical quad: posU.xyz = (t, edge, 0); normalV unused. sear pulls these as localPos, the
// chunk expands. 4 corners, 6 indices (two triangles)
// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, -1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 0, 0, 0, 1,
    0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

const _m = new Float32Array(16);
let _quadBase = 0;

// each retained Line is one segment from the entity's world pos along its rotated offset; an Arrow on it
// adds fletched heads at the endpoints. Appended on top of this frame's immediate segments. Small counts
// (scene annotations) — the immediate API is the scale path
function expandRetained(state: State): void {
    for (const eid of state.query([Line, Transform])) {
        if (!Line.visible.get(eid)) continue;
        composeTransform(eid, _m);
        const ox = Line.offset.x.get(eid);
        const oy = Line.offset.y.get(eid);
        const oz = Line.offset.z.get(eid);
        const sx = _m[12];
        const sy = _m[13];
        const sz = _m[14];
        const ex = sx + _m[0] * ox + _m[4] * oy + _m[8] * oz;
        const ey = sy + _m[1] * ox + _m[5] * oy + _m[9] * oz;
        const ez = sz + _m[2] * ox + _m[6] * oy + _m[10] * oz;
        const w = Line.thickness.get(eid);
        const c = packColor(Line.color.get(eid), Line.opacity.get(eid));
        push(sx, sy, sz, ex, ey, ez, w, c);
        if (state.has(eid, Arrow)) {
            const size = Arrow.size.get(eid);
            if (Arrow.end.get(eid)) head(ex, ey, ez, sx, sy, sz, size, w, c);
            if (Arrow.start.get(eid)) head(sx, sy, sz, ex, ey, ez, size, w, c);
        }
    }
}

// runs after the immediate appends (simulation systems) and before sear reads the segment buffer
// (PrepassSystem resolves the draw's bind group): expands retained components, then uploads + clears
const LinesSystem: System = {
    name: "lines",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    setup() {
        _quadBase = Meshes.get("lineQuad")?.indexBase ?? 0;
        Draws.register({
            name: "lines",
            surface: "lines",
            mesh: "lineQuad",
            args: { indirect: Lines.args! },
        });
    },
    update(state) {
        if (!Compute.device || !ready()) return;
        expandRetained(state);
        flushSegments(Compute.device, _quadBase);
    },
};

/**
 * the kitchen debug-line producer: an immediate {@link segment} / {@link box} / {@link arrow} API plus
 * the retained {@link Line} / {@link Arrow} components, both feeding one instanced-quad draw rendered
 * as a sear `"alpha"` surface (screen-space constant-pixel width, no overlay pass). Depends on
 * {@link RenderPlugin}; a Sear camera renders it
 */
export const LinesPlugin: Plugin = {
    name: "Lines",
    components: { Line, Arrow },
    systems: [LinesSystem],
    dependencies: [RenderPlugin, TransformsPlugin],
    traits: {
        Line: {
            requires: [Transform],
            defaults: () => ({
                offset: [1, 0, 0, 0],
                thickness: 2,
                color: 0xffffff,
                opacity: 1,
                visible: 1,
            }),
            format: { color: formatHex },
        },
        Arrow: {
            requires: [Line],
            defaults: () => ({ start: 0, end: 1, size: 1 }),
        },
    },

    initialize() {
        resetCount();
        mesh({ name: "lineQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });
        Surfaces.register({
            name: "lines",
            blend: "alpha",
            screen: true,
            bindings: { lineSegments: { type: "storage", element: "Segment" } },
            interpolators: { lineRgba: "vec4<f32>", edgeDist: "f32", halfPx: "f32" },
            preamble: SEGMENT_WGSL,
            vs: LINE_VS,
            fs: LINE_FS,
        });
    },

    warm() {
        if (!Compute.device) return;
        warmSegments(Compute.device);
    },

    dispose() {
        disposeSegments();
    },
};
