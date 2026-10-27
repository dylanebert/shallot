import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import {
    BeginFrameSystem,
    Camera,
    CorePipelinePlugin,
    initialViewSampleCounts,
    PrepassSystem,
    RenderContext,
    RenderPhases,
    renderLayers,
    type TransparentPhase,
    type View,
} from "../../core/rendering";
import { composeGlobalTransform, GlobalTransform } from "../../core/transform";
import type { Plugin, System, World } from "../../engine";
import { component, f32, precompile } from "../../engine";
import { drawInstances, setRenderBindGroup, setRenderPipeline } from "../../engine/runtime";
import {
    createLinePipeline,
    instanceLayout,
    LineInstance,
    lineLayout,
    SEGMENT_BYTES,
    SEGMENT_FLOATS,
    Segment,
    viewLayout,
} from "./pipeline";
import {
    disposeSegments,
    flushSegments,
    type GizmoAsset,
    initializeSegmentState,
    Lines,
    lineCount,
    ready,
    resetCount,
    type SegmentState,
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

interface AssetUpload {
    asset: GizmoAsset;
    revision: number;
    buffer: (TgpuBuffer<d.WgslArray<typeof Segment>> & StorageFlag) | null;
    group: GPUBindGroup | null;
    center: Float32Array;
}
const gizmoAssets = { create: () => [] as AssetUpload[] };

/** Registers shared local-space geometry in this World; returns its handle. */
export function registerGizmo(world: World, asset: GizmoAsset): number {
    return world
        .resource(gizmoAssets)
        .push({ asset, revision: -1, buffer: null, group: null, center: new Float32Array(3) });
}

const RETAINED_LINES_QUERY = [Gizmo, GlobalTransform];
const matrixKey = { create: () => new Float32Array(16) };

function boundsCenter(
    source: SegmentState,
    count: number,
    center: Float32Array | Float64Array,
    mask = 0xffffffff,
): boolean {
    let visible = false;
    for (let axis = 0; axis < 3; axis++) {
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < count; i++) {
            if (!(source.layers[i] & mask)) continue;
            visible = true;
            const at = i * SEGMENT_FLOATS;
            for (let point = 0; point < (source.u32[at + 11] ? 12 : 8); point += 4) {
                min = Math.min(min, source.f32[at + point + axis]);
                max = Math.max(max, source.f32[at + point + axis]);
            }
        }
        center[axis] = (min + max) * 0.5;
    }
    return visible;
}

function uploadAssets(world: World): void {
    for (const upload of world.resource(gizmoAssets)) {
        const source = upload.asset.segments;
        if (upload.revision === source.revision) continue;
        upload.revision = source.revision;
        if (!source.count) continue;
        if (
            !upload.buffer ||
            world.gpu.root.unwrap(upload.buffer).size < source.count * SEGMENT_BYTES
        ) {
            const stale = upload.buffer;
            if (stale) world.gpu.device.queue.onSubmittedWorkDone().then(() => stale.destroy());
            upload.buffer = world.gpu.root
                .createBuffer(d.arrayOf(Segment, source.capacity))
                .$usage("storage");
            upload.group = world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(lineLayout, { segments: upload.buffer }),
            );
        }
        world.gpu.device.queue.writeBuffer(
            world.gpu.root.unwrap(upload.buffer),
            0,
            source.staging,
            0,
            source.count * SEGMENT_BYTES,
        );
        boundsCenter(source, source.count, upload.center);
    }
}

const INSTANCE_BYTES = d.sizeOf(LineInstance);
const INSTANCE_WORDS = INSTANCE_BYTES / 4;
function retainedRecords(capacity = 64) {
    const staging = new ArrayBuffer(capacity * INSTANCE_BYTES);
    return {
        capacity,
        count: 0,
        floats: new Float32Array(staging),
        ints: new Uint32Array(staging),
        eids: new Uint32Array(capacity),
        handles: new Uint32Array(capacity),
        centers: new Float64Array(capacity * 3),
    };
}
const retainedKey = { create: () => retainedRecords() };

function extractGizmos(world: World): void {
    const records = world.resource(retainedKey);
    records.count = 0;
    const gizmos = world.storage(Gizmo);
    const assets = world.resource(gizmoAssets);
    const matrix = world.resource(matrixKey);
    for (const eid of world.query(RETAINED_LINES_QUERY)) {
        const handle = gizmos.handle.get(eid);
        const asset = assets[handle - 1];
        if (!asset?.asset.segments.count) continue;
        const row = ++records.count;
        if (row >= records.capacity) {
            const next = retainedRecords(records.capacity * 2);
            next.count = records.count;
            next.ints.set(records.ints);
            next.eids.set(records.eids);
            next.handles.set(records.handles);
            next.centers.set(records.centers);
            Object.assign(records, next);
        }
        composeGlobalTransform(world, eid, matrix);
        const at = row * INSTANCE_WORDS;
        records.floats.set(matrix, at);
        records.floats[at + 16] = gizmos.width.get(eid);
        records.floats[at + 17] = gizmos.opacity.get(eid);
        records.floats[at + 18] = gizmos.depthBias.get(eid);
        records.ints[at + 19] = renderLayers(world, eid);
        records.eids[row] = eid;
        records.handles[row] = handle;
        const c = asset.center;
        for (let axis = 0; axis < 3; axis++)
            records.centers[row * 3 + axis] =
                matrix[axis] * c[0] +
                matrix[4 + axis] * c[1] +
                matrix[8 + axis] * c[2] +
                matrix[12 + axis];
    }
}

interface InstanceUpload {
    buffer: TgpuBuffer<d.WgslArray<typeof LineInstance>> & StorageFlag;
    group: GPUBindGroup;
    staging: ArrayBuffer;
    floats: Float32Array;
    ints: Uint32Array;
    capacity: number;
}
const lineRendererKey = {
    create: () => ({
        single: null as ReturnType<typeof createLinePipeline> | null,
        multisample: null as ReturnType<typeof createLinePipeline> | null,
        viewGroups: new Map<number, { buffer: GPUBuffer; group: GPUBindGroup }>(),
        instances: new Map<number, InstanceUpload>(),
        segmentBuffer: null as object | null,
        segmentGroup: null as GPUBindGroup | null,
    }),
};

function queueLines(world: World, camera: number, _view: View, phase: TransparentPhase): void {
    const mask = renderLayers(world, camera);
    if (boundsCenter(segmentState(world), lineCount(world), phase.positionScratch, mask))
        phase.addFromScratch(0, 0, 0);
    const records = world.resource(retainedKey);
    const position = phase.positionScratch;
    for (let row = 1; row <= records.count; row++) {
        if (!(records.ints[row * INSTANCE_WORDS + 19] & mask)) continue;
        for (let axis = 0; axis < 3; axis++) position[axis] = records.centers[row * 3 + axis];
        phase.addFromScratch(row, records.handles[row], records.eids[row]);
    }
}

function prepareLines(
    world: World,
    camera: number,
    view: View,
    phase: TransparentPhase,
    renderer: number,
): void {
    const state = world.resource(lineRendererKey);
    const viewBuffer = world.resource(RenderContext).viewBuffers[view.slot];
    if (!viewBuffer) return;
    if (state.viewGroups.get(view.slot)?.buffer !== viewBuffer) {
        const group = world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(viewLayout, { view: viewBuffer }),
        );
        state.viewGroups.set(view.slot, { buffer: viewBuffer, group });
    }
    const lines = world.resource(Lines);
    if (lines.buffer && state.segmentBuffer !== lines.buffer) {
        state.segmentBuffer = lines.buffer;
        state.segmentGroup = world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(lineLayout, { segments: lines.buffer }),
        );
    }
    if (!phase.length) return;
    let instances = state.instances.get(view.slot);
    if (!instances || instances.capacity < phase.length) {
        const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(phase.length)));
        const stale = instances?.buffer;
        if (stale) world.gpu.device.queue.onSubmittedWorkDone().then(() => stale.destroy());
        const buffer = world.gpu.root
            .createBuffer(d.arrayOf(LineInstance, capacity))
            .$usage("storage");
        const staging = new ArrayBuffer(capacity * d.sizeOf(LineInstance));
        instances = {
            buffer,
            group: world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(instanceLayout, { instances: buffer }),
            ),
            staging,
            floats: new Float32Array(staging),
            ints: new Uint32Array(staging),
            capacity,
        };
        state.instances.set(view.slot, instances);
    }
    const { floats, ints } = instances;
    const records = world.resource(retainedKey);
    const cameraLayers = renderLayers(world, camera);
    for (let i = 0; i < phase.length; i++) {
        const item = phase.item(i);
        if (item.renderer !== renderer) continue;
        const at = i * INSTANCE_WORDS;
        if (item.batch) {
            const source = item.draw * INSTANCE_WORDS;
            for (let word = 0; word < INSTANCE_WORDS; word++)
                ints[at + word] = records.ints[source + word];
        } else {
            floats.fill(0, at, at + INSTANCE_WORDS);
            floats[at] = floats[at + 5] = floats[at + 10] = floats[at + 15] = 1;
            floats[at + 16] = -1;
            floats[at + 17] = 1;
            ints[at + 19] = cameraLayers;
        }
    }
    world.gpu.device.queue.writeBuffer(
        world.gpu.root.unwrap(instances.buffer),
        0,
        instances.staging,
        0,
        phase.length * d.sizeOf(LineInstance),
    );
}

function drawLines(
    world: World,
    eid: number,
    view: View,
    phase: TransparentPhase,
    start: number,
    end: number,
    pass: GPURenderPassEncoder,
    _renderer: number,
): void {
    const state = world.resource(lineRendererKey);
    const viewGroup = state.viewGroups.get(view.slot)?.group;
    const instances = state.instances.get(view.slot);
    const batch = phase.item(start).batch;
    const asset = batch ? world.resource(gizmoAssets)[batch - 1] : undefined;
    const group = batch ? asset?.group : state.segmentGroup;
    const count = batch ? asset?.asset.segments.count : lineCount(world);
    if (!viewGroup || !instances || !group || !count) return;
    const key = world.storage(Camera).antialias.get(eid) ? "multisample" : "single";
    const pipeline = (state[key] ??= createLinePipeline(
        world,
        world.resource(RenderContext).format,
        key === "single" ? 1 : 4,
    ));
    setRenderPipeline(pass, world.gpu.root.unwrap(pipeline));
    setRenderBindGroup(pass, 0, viewGroup);
    setRenderBindGroup(pass, 1, group);
    setRenderBindGroup(pass, 2, instances.group);
    drawInstances(pass, 6 * count, end - start, start);
    phase.metrics.draws++;
    phase.metrics.pipelineChanges++;
    phase.metrics.bindGroupChanges += 3;
}

const LinesSystem: System = {
    name: "lines",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(world) {
        if (!ready(world)) return;
        uploadAssets(world);
        extractGizmos(world);
        flushSegments(world, world.gpu.device);
    },
};

/** Screen-space pixel-width immediate lines and retained local-space gizmos in core's sorted transparent phase. */
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
    warm(world) {
        warmSegments(world);
        const format = world.resource(RenderContext).format;
        const state = world.resource(lineRendererKey);
        precompile(world, "lines-prepare", () => {
            for (const samples of initialViewSampleCounts(world))
                state[samples === 1 ? "single" : "multisample"] = createLinePipeline(
                    world,
                    format,
                    samples,
                );
            return [];
        });
    },
    dispose(world) {
        disposeSegments(world);
        for (const asset of world.resource(gizmoAssets)) asset.buffer?.destroy();
        const state = world.resource(lineRendererKey);
        for (const instances of state.instances.values()) instances.buffer.destroy();
        state.instances.clear();
        state.viewGroups.clear();
        state.segmentBuffer = null;
        state.segmentGroup = null;
        state.single = state.multisample = null;
    },
};
