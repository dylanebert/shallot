import { registration } from "../../engine";
// Lines — the shallot debug-line producer. One shared segment buffer, two feeders: an immediate API
// (`drawLine` / `drawWireBox` / `drawArrow`, appended and cleared each frame — the scale path) and the retained
// `Line` / `Arrow` components (declarative scene annotations, expanded into segments each frame).
// Everything draws as one instanced 6-vertex quad per segment, rendered as a sear `"alpha"` surface
// inside the color pass — translucent, depth-tested, depth-write off, no overlay pass. Screen-space
// constant-pixel width: the surface projects each segment's endpoints itself (sear's `screen` mode)
// and writes its own clip position, expanding the quad by a pixel half-width read from `view.resolution`.
// Bevy's gizmo model; arrows are folded in (a shaft segment + segment-fletched head), no separate
// primitive. The segment staging + upload + immediate API live in `segments.ts`, the surface in
// `surface.ts`.

import { Meshes, MeshPlugin, registerMesh } from "../../core/mesh";
import { BeginFrameSystem, Draws, RenderingPlugin, registerSurface } from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { composeGlobalTransform, f32, GlobalTransform, vec4 } from "../../engine";
import { packColor } from "../../engine/utils";
import { RenderPrepassesSystem } from "../../standard/rendering";
import {
    disposeSegments,
    flushSegments,
    head,
    initializeSegmentState,
    Lines,
    push,
    ready,
    resetCount,
    warmSegments,
} from "./segments";
import { lineFs, lineLayout, lineVaryings, lineVs } from "./surface";

export { drawArrow, drawLine, drawWireBox } from "./segments";

/**
 * a debug line anchored to an entity, drawn from its {@link Transform} position along a world-rotated
 * offset. A retained scene annotation, expanded into one screen-space segment each frame
 *
 * @example
 * ```
 * const eid = world.create();
 * world.add(eid, Line, { offset: [0, 1, 0, 0], thickness: 3, color: 0x44ff88 });
 * world.add(eid, Transform);
 * ```
 */
export const Line = {
    /** line vector from the entity in its local frame, rotated by the transform (`0 1 0` = one unit up) */
    offset: vec4,
    /** constant screen width in pixels */
    thickness: f32,
    /** hex sRGB color */
    color: f32,
    /** 0..1 opacity multiplier */
    opacity: f32,
    /** drawn when nonzero; set to 0 to hide without removing */
    visible: f32,
};

/**
 * an arrowhead on a {@link Line}: four world-space fins (Bevy's fletched shape) at the line's endpoints.
 * Requires a {@link Line} on the same entity
 *
 * @example
 * ```
 * const eid = world.create();
 * world.add(eid, Line, { offset: [2, 0, 0, 0], color: 0xffcc00 });
 * world.add(eid, Arrow, { size: 1.5 });
 * world.add(eid, Transform);
 * ```
 */
export const Arrow = {
    /** a head at the start endpoint when nonzero */
    start: f32,
    /** a head at the end endpoint when nonzero */
    end: f32,
    /** head size relative to the shaft length */
    size: f32,
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

// each retained Line is one segment from the entity's world pos along its rotated offset; an Arrow on it
// adds fletched heads at the endpoints. Appended on top of this frame's immediate segments. Small counts
// (scene annotations) — the immediate API is the scale path
function expandRetained(world: World): void {
    for (const eid of world.query([Line, GlobalTransform])) {
        if (!world.storage(Line).visible.get(eid)) continue;
        composeGlobalTransform(world, eid, _m);
        const ox = world.storage(Line).offset.x.get(eid);
        const oy = world.storage(Line).offset.y.get(eid);
        const oz = world.storage(Line).offset.z.get(eid);
        const sx = _m[12];
        const sy = _m[13];
        const sz = _m[14];
        const ex = sx + _m[0] * ox + _m[4] * oy + _m[8] * oz;
        const ey = sy + _m[1] * ox + _m[5] * oy + _m[9] * oz;
        const ez = sz + _m[2] * ox + _m[6] * oy + _m[10] * oz;
        const w = world.storage(Line).thickness.get(eid);
        const c = packColor(
            world.storage(Line).color.get(eid),
            world.storage(Line).opacity.get(eid),
        );
        push(world, sx, sy, sz, ex, ey, ez, w, c);
        if (world.has(eid, Arrow)) {
            const size = world.storage(Arrow).size.get(eid);
            if (world.storage(Arrow).end.get(eid)) head(world, ex, ey, ez, sx, sy, sz, size, w, c);
            if (world.storage(Arrow).start.get(eid))
                head(world, sx, sy, sz, ex, ey, ez, size, w, c);
        }
    }
}

// runs after the immediate appends (simulation systems) and before sear reads the segment buffer
// (RenderPrepassesSystem resolves the draw's bind group): expands retained components, then uploads + clears
const LinesSystem: System = {
    name: "lines",
    group: "draw",
    after: [BeginFrameSystem],
    before: [RenderPrepassesSystem],
    setup(world: World) {
        world.resource(Draws).register({
            name: "lines",
            surface: "lines",
            mesh: "lineQuad",
            args: { indirect: world.resource(Lines).args! },
        });
    },
    update(world) {
        if (!world.gpu.device || !ready(world)) return;
        expandRetained(world);
        flushSegments(
            world,
            world.gpu.device,
            world.resource(Meshes).get("lineQuad")?.indexBase ?? 0,
        );
    },
};

/**
 * the shallot debug-line producer: an immediate {@link drawLine} / {@link drawWireBox} / {@link drawArrow} API plus
 * the retained {@link Line} / {@link Arrow} components, both feeding one instanced-quad draw rendered
 * as a sear `"alpha"` surface (screen-space constant-pixel width, no overlay pass). Depends on
 * {@link RenderingPlugin}; a StandardRenderer camera renders it
 */
export const LinesPlugin: Plugin = {
    name: "Lines",
    components: [
        registration("Line", Line, {
            defaults: () => ({
                offset: [1, 0, 0, 0],
                thickness: 2,
                color: 0xffffff,
                opacity: 1,
                visible: 1,
            }),
        }),
        registration("Arrow", Arrow, {
            defaults: () => ({ start: 0, end: 1, size: 1 }),
        }),
    ],
    systems: [LinesSystem],
    dependencies: [MeshPlugin, RenderingPlugin],

    initialize(world) {
        initializeSegmentState(world);
        resetCount(world);
        registerMesh(world, { name: "lineQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });
        registerSurface(world, {
            name: "lines",
            layout: lineLayout,
            blend: "alpha",
            screen: true,
            varyings: lineVaryings,
            vs: lineVs,
            fs: lineFs,
        });
    },

    warm(world: World) {
        if (!world.gpu.device) return;
        warmSegments(world, world.gpu.device);
    },

    dispose(world: World) {
        disposeSegments(world);
    },
};
