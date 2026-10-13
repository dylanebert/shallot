import { component } from "../../engine";
// Lines — the shallot debug-line producer. One shared segment buffer, two feeders: an immediate API
// (`drawLine` / `drawWireBox` / `drawArrow`, appended and cleared each frame — the scale path) and the retained
// `Line` / `Arrow` components (declarative scene annotations, expanded into segments each frame).
// Everything draws as one non-indexed six-vertex quad per segment through core's transparent phase:
// translucent, depth-tested, depth-write off, no overlay pass. Screen-space constant-pixel width; the shader
// projects endpoints and expands the quad. Lines previously used an alpha surface, so they never entered
// the opaque depth prepass or either shadow path; moving them drops no shadow-casting behavior. Bevy's
// gizmo model; arrows are folded in (a shaft segment + segment-fletched head). Staging/upload live in
// `segments.ts`; phase shaders live in `pipeline.ts`.

import {
    BeginFrameSystem,
    Camera,
    CorePipelinePlugin,
    PrepassSystem,
    RenderContext,
    RenderPhases,
    type TransparentPhase,
    type View,
} from "../../core/rendering";
import { composeGlobalTransform, GlobalTransform } from "../../core/transform";
import type { Plugin, System, World } from "../../engine";
import { f32, vec4 } from "../../engine";
import { packColor } from "../../engine/utils";
import { createLinePipeline, lineLayout, viewLayout } from "./pipeline";
import {
    disposeSegments,
    flushSegments,
    head,
    initializeSegmentState,
    Lines,
    lineCount,
    lineStableIds,
    lineStableParts,
    lineStaging,
    push,
    ready,
    resetCount,
    warmSegments,
} from "./segments";

export { drawArrow, drawLine, drawWireBox } from "./segments";

/**
 * a debug line anchored to an entity, drawn from its {@link Transform} position along a world-rotated
 * offset. A retained scene annotation, expanded into one screen-space segment each frame
 */
export const Line = component(
    "Line",
    {
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
    },
    {
        defaults: () => ({
            offset: [1, 0, 0, 0],
            thickness: 2,
            color: 0xffffff,
            opacity: 1,
            visible: 1,
        }),
    },
);

/**
 * an arrowhead on a {@link Line}: four world-space fins (Bevy's fletched shape) at the line's endpoints.
 * Requires a {@link Line} on the same entity
 */
export const Arrow = component(
    "Arrow",
    {
        /** a head at the start endpoint when nonzero */
        start: f32,
        /** a head at the end endpoint when nonzero */
        end: f32,
        /** head size relative to the shaft length */
        size: f32,
    },
    {
        defaults: () => ({ start: 0, end: 1, size: 1 }),
    },
);

const _m = new Float32Array(16);
const RETAINED_LINES_QUERY = [Line, GlobalTransform];

// each retained Line is one segment from the entity's world pos along its rotated offset; an Arrow on it
// adds fletched heads at the endpoints. Appended on top of this frame's immediate segments. Small counts
// (scene annotations) — the immediate API is the scale path
function expandRetained(world: World): void {
    for (const eid of world.query(RETAINED_LINES_QUERY)) {
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
        let segment = 0;
        push(world, sx, sy, sz, ex, ey, ez, w, c, eid, segment++);
        if (world.has(eid, Arrow)) {
            const size = world.storage(Arrow).size.get(eid);
            if (world.storage(Arrow).end.get(eid))
                segment += head(world, ex, ey, ez, sx, sy, sz, size, w, c, eid, segment);
            if (world.storage(Arrow).start.get(eid))
                head(world, sx, sy, sz, ex, ey, ez, size, w, c, eid, segment);
        }
    }
}

// runs after immediate appends and before the main color pass: expands retained components, uploads, then clears
interface LineRendererState {
    single: import("typegpu").TgpuRenderPipeline | null;
    multisample: import("typegpu").TgpuRenderPipeline | null;
    viewGroups: Map<number, { buffer: GPUBuffer; group: GPUBindGroup }>;
    segmentBuffer: object | null;
    segmentGroup: GPUBindGroup | null;
    lastPipeline: import("typegpu").TgpuRenderPipeline | null;
    lastViewGroup: GPUBindGroup | null;
    lastSegmentGroup: GPUBindGroup | null;
}
const lineRendererKey = {
    create: (): LineRendererState => ({
        single: null,
        multisample: null,
        viewGroups: new Map(),
        segmentBuffer: null,
        segmentGroup: null,
        lastPipeline: null,
        lastViewGroup: null,
        lastSegmentGroup: null,
    }),
};

function queueLines(world: World, _eid: number, _view: View, phase: TransparentPhase): void {
    const count = lineCount(world);
    if (!count || !world.resource(Lines).buffer) return;
    const values = lineStaging(world);
    const stableIds = lineStableIds(world);
    const stableParts = lineStableParts(world);
    const position = phase.positionScratch;
    for (let i = 0; i < count; i++) {
        const at = i * 8;
        position[0] = (values[at]! + values[at + 4]!) * 0.5;
        position[1] = (values[at + 1]! + values[at + 5]!) * 0.5;
        position[2] = (values[at + 2]! + values[at + 6]!) * 0.5;
        phase.addFromScratch(i, 0, stableIds[i]!, stableParts[i]!);
    }
}

function prepareLines(world: World, _eid: number, view: View): void {
    const lines = world.resource(Lines);
    if (!lineCount(world) || !lines.buffer) return;
    const state = world.resource(lineRendererKey);
    const viewBuffer = world.resource(RenderContext).viewBuffers[view.slot];
    if (!viewBuffer) return;
    let viewGroup = state.viewGroups.get(view.slot);
    if (!viewGroup || viewGroup.buffer !== viewBuffer) {
        const group = world.gpu.root.createBindGroup(viewLayout, { view: viewBuffer });
        viewGroup = { buffer: viewBuffer, group: world.gpu.root.unwrap(group) };
        state.viewGroups.set(view.slot, viewGroup);
    }
    if (state.segmentBuffer !== lines.buffer || !state.segmentGroup) {
        const group = world.gpu.root.createBindGroup(lineLayout, { segments: lines.buffer });
        state.segmentBuffer = lines.buffer;
        state.segmentGroup = world.gpu.root.unwrap(group);
    }
    state.lastPipeline = null;
    state.lastViewGroup = null;
    state.lastSegmentGroup = null;
}

function drawLines(
    world: World,
    eid: number,
    view: View,
    phase: TransparentPhase,
    start: number,
    end: number,
    pass: GPURenderPassEncoder,
    renderer: number,
): void {
    const state = world.resource(lineRendererKey);
    const viewGroup = state.viewGroups.get(view.slot)?.group;
    const segmentGroup = state.segmentGroup;
    if (!viewGroup || !segmentGroup) return;
    const pipeline =
        world.storage(Camera).antialias.get(eid) !== 0 ? state.multisample : state.single;
    if (!pipeline) return;
    const previous = start > 0 ? phase.item(start - 1) : undefined;
    if (previous?.renderer !== renderer) {
        state.lastPipeline = null;
        state.lastViewGroup = null;
        state.lastSegmentGroup = null;
    }
    pass.setPipeline(world.gpu.root.unwrap(pipeline));
    pass.setBindGroup(0, viewGroup);
    pass.setBindGroup(1, segmentGroup);
    if (state.lastPipeline !== pipeline) phase.metrics.pipelineChanges++;
    if (state.lastViewGroup !== viewGroup) phase.metrics.bindGroupChanges++;
    if (state.lastSegmentGroup !== segmentGroup) phase.metrics.bindGroupChanges++;
    state.lastPipeline = pipeline;
    state.lastViewGroup = viewGroup;
    state.lastSegmentGroup = segmentGroup;
    for (let i = start; i < end; i++) {
        pass.draw(6, 1, 0, phase.item(i).draw);
        phase.metrics.draws++;
    }
}

const LinesSystem: System = {
    name: "lines",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(world) {
        if (!world.gpu.device || !ready(world)) return;
        expandRetained(world);
        flushSegments(world, world.gpu.device);
    },
};

/**
 * the shallot debug-line producer: an immediate {@link drawLine} / {@link drawWireBox} / {@link drawArrow} API plus
 * the retained {@link Line} / {@link Arrow} components, both feeding one instanced-quad draw rendered
 * in core's transparent phase (screen-space constant-pixel width, no overlay pass). Depends on
 * {@link CorePipelinePlugin}; a core camera renders it
 */
export const LinesPlugin: Plugin = {
    gpu: {},
    name: "Lines",
    components: [Line, Arrow],
    systems: [LinesSystem],
    dependencies: [CorePipelinePlugin],

    initialize(world) {
        initializeSegmentState(world);
        resetCount(world);
        world.resource(RenderPhases).push({
            transparent: queueLines,
            prepareTransparent: prepareLines,
            drawTransparent: drawLines,
        });
    },

    warm(world: World) {
        if (!world.gpu.device) return;
        warmSegments(world);
        const format = world.resource(RenderContext).format;
        const state = world.resource(lineRendererKey);
        state.single = createLinePipeline(world.gpu.root, format, 1);
        state.multisample = createLinePipeline(world.gpu.root, format, 4);
    },

    dispose(world: World) {
        disposeSegments(world);
        const state = world.resource(lineRendererKey);
        state.viewGroups.clear();
        state.segmentBuffer = null;
        state.segmentGroup = null;
        state.lastPipeline = null;
        state.lastViewGroup = null;
        state.lastSegmentGroup = null;
        state.single = null;
        state.multisample = null;
    },
};
