// Lines — the kitchen debug-line producer. One shared segment buffer, two feeders: an immediate API
// (`segment` / `box` / `arrow`, appended and cleared each frame — the scale path) and the retained
// `Line` / `Arrow` components (declarative scene annotations, expanded into segments each frame).
// Everything draws as one instanced 6-vertex quad per segment, rendered as a sear `"alpha"` surface
// inside the color pass — translucent, depth-tested, depth-write off, no overlay pass. Screen-space
// constant-pixel width: the surface projects each segment's endpoints itself (sear's `screen` mode)
// and writes its own clip position, expanding the quad by a pixel half-width read from `view.resolution`.
// Bevy's gizmo model; arrows are folded in (a shaft segment + segment-fletched head), no separate
// primitive. The segment staging + upload + immediate API live in `segments.ts`, the surface in
// `surface.ts`.

import type { Plugin, State, System } from "../../engine";
import { Compute, f32, formatHex, sparse, vec4 } from "../../engine";
import { packColor } from "../../engine/utils/core";
import { mesh, RenderPlugin } from "../../standard/render";
import { BeginFrameSystem, Draws, Meshes } from "../../standard/render/core";
import { PrepassSystem, registerSurface } from "../../standard/sear/core";
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
import { lineFs, lineLayout, lineVaryings, lineVs } from "./surface";

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

    initialize(state) {
        resetCount();
        mesh({ name: "lineQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });
        registerSurface(state, {
            name: "lines",
            layout: lineLayout,
            blend: "alpha",
            screen: true,
            varyings: lineVaryings,
            vs: lineVs,
            fs: lineFs,
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
