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
import { component, f32 } from "../../engine";
import { packColor } from "../../engine/utils";
import { createLinePipeline, lineLayout, SEGMENT_FLOATS, viewLayout } from "./pipeline";
import {
    disposeSegments,
    flushSegments,
    type GizmoAsset,
    grow,
    initializeSegmentState,
    Lines,
    lineCount,
    lineStableIds,
    lineStableParts,
    lineStaging,
    ready,
    resetCount,
    segmentState,
    warmSegments,
} from "./segments";

export { drawArrow, drawLine, drawWireBox, GizmoAsset, type LineOptions } from "./segments";

/** Draws a registered asset at the entity's GlobalTransform. Remove Gizmo to hide it. */
export const Gizmo = component(
    "Gizmo",
    {
        /** World-local handle returned by registerGizmo. */
        handle: f32,
        /** Constant screen width in pixels for every segment; defaults to 2. */
        width: f32,
        /** 0..1 opacity multiplier. */
        opacity: f32,
        /** Clip-depth bias, clamped to [-1, 1]: 0 (default) depth-tests normally, -1 draws on top, 1 moves to the far plane. Negative intermediate values bias exponentially toward the camera. */
        depthBias: f32,
    },
    { defaults: () => ({ handle: 0, width: 2, opacity: 1, depthBias: 0 }) },
);

const gizmoAssets = { create: () => [] as GizmoAsset[] };

/** Registers shared local-space geometry in this World; returns its handle. */
export function registerGizmo(world: World, asset: GizmoAsset): number {
    return world.resource(gizmoAssets).push(asset);
}

const RETAINED_LINES_QUERY = [Gizmo, GlobalTransform];
const matrixKey = { create: () => new Float32Array(16) };

function expandRetained(world: World): void {
    const matrix = world.resource(matrixKey);
    const gizmos = world.storage(Gizmo);
    const assets = world.resource(gizmoAssets);
    const target = segmentState(world);
    for (const eid of world.query(RETAINED_LINES_QUERY)) {
        const asset = assets[gizmos.handle.get(eid) - 1];
        if (!asset) continue;
        composeGlobalTransform(world, eid, matrix);
        const source = asset.segments;
        const width = gizmos.width.get(eid);
        const alpha = packColor(0, gizmos.opacity.get(eid)) & 0xff000000;
        if (target.count + source.count > target.capacity)
            grow(target, target.count + source.count);
        for (let i = 0; i < source.count; i++) {
            const at = i * SEGMENT_FLOATS;
            const index = target.count++;
            const out = index * SEGMENT_FLOATS;
            const triangle = source.u32[at + 11];
            for (let point = 0; point < (triangle ? 12 : 8); point += 4) {
                const x = source.f32[at + point];
                const y = source.f32[at + point + 1];
                const z = source.f32[at + point + 2];
                target.f32[out + point] =
                    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
                target.f32[out + point + 1] =
                    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
                target.f32[out + point + 2] =
                    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
            }
            target.f32[out + 3] = width;
            target.u32[out + 7] = (source.u32[at + 7] & 0xffffff) | alpha;
            target.u32[out + 11] = triangle;
            target.f32[out + 12] = gizmos.depthBias.get(eid);
            target.stableIds[index] = eid;
            target.stableParts[index] = i;
        }
    }
}

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
        const at = i * SEGMENT_FLOATS;
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
        if (!ready(world)) return;
        expandRetained(world);
        flushSegments(world, world.gpu.device);
    },
};

/**
 * the shallot debug-line producer: an immediate {@link drawLine} / {@link drawWireBox} / {@link drawArrow} API plus
 * retained {@link Gizmo} assets, both feeding one instanced-quad draw rendered
 * in core's transparent phase (screen-space constant-pixel width, no overlay pass). Depends on
 * {@link CorePipelinePlugin}; a core camera renders it
 */
export const LinesPlugin: Plugin = {
    gpu: {},
    name: "Lines",
    components: [Gizmo],
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
