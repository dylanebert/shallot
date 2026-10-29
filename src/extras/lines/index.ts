import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import {
    BeginFrameSystem,
    Camera,
    CorePipelinePlugin,
    globalTransformTable,
    initialViewSampleCounts,
    PrepassSystem,
    RenderContext,
    RenderPhases,
    renderLayers,
    type TransparentPhase,
    type View,
    type Xform,
} from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
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

/** Draws a registered asset at the entity's interpolated GlobalTransform, as meshes draw. Remove Gizmo to hide it. */
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
    count: number;
    start: number;
    cursor: number;
}
const gizmoAssets = { create: () => [] as AssetUpload[] };

/** Registers shared local-space geometry in this World; returns its handle. */
export function registerGizmo(world: World, asset: GizmoAsset): number {
    return world
        .resource(gizmoAssets)
        .push({ asset, revision: -1, buffer: null, group: null, count: 0, start: 0, cursor: 0 });
}

const RETAINED_LINES_QUERY = [Gizmo, GlobalTransform];

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
        handles: new Uint32Array(capacity),
    };
}
const retainedKey = { create: () => retainedRecords() };

function extractGizmos(world: World): void {
    const records = world.resource(retainedKey);
    records.count = 0;
    const gizmos = world.storage(Gizmo);
    const assets = world.resource(gizmoAssets);
    for (const eid of world.query(RETAINED_LINES_QUERY)) {
        const handle = gizmos.handle.get(eid);
        const asset = assets[handle - 1];
        if (!asset?.asset.segments.count) continue;
        const row = ++records.count;
        if (row >= records.capacity) {
            const next = retainedRecords(records.capacity * 2);
            next.count = records.count;
            next.ints.set(records.ints);
            next.handles.set(records.handles);
            Object.assign(records, next);
        }
        const at = row * INSTANCE_WORDS;
        records.ints[at] = eid;
        records.floats[at + 1] = gizmos.width.get(eid);
        records.floats[at + 2] = gizmos.opacity.get(eid);
        records.floats[at + 3] = gizmos.depthBias.get(eid);
        records.ints[at + 4] = renderLayers(world, eid);
        records.handles[row] = handle;
    }
}

interface InstanceUpload {
    buffer: TgpuBuffer<d.WgslArray<typeof LineInstance>> & StorageFlag;
    group: GPUBindGroup | null;
    placement: object | null;
    rows: object | null;
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
    const assets = world.resource(gizmoAssets);
    for (const asset of assets) asset.count = 0;
    for (let row = 1; row <= records.count; row++) {
        if (records.ints[row * INSTANCE_WORDS + 4] & mask) assets[records.handles[row] - 1].count++;
    }
    let start = 1;
    for (let i = 0; i < assets.length; i++) {
        const asset = assets[i];
        asset.start = asset.cursor = start;
        start += asset.count;
        // Bevy's AlwaysOnTop gizmos follow sorted transparency; depth testing remains enabled.
        if (asset.count) phase.addUnsorted(i + 1, i + 1, i + 1);
    }
}

function prepareLines(
    world: World,
    camera: number,
    view: View,
    phase: TransparentPhase,
    _renderer: number,
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
    const records = world.resource(retainedKey);
    const required = records.count + 1;
    let instances = state.instances.get(view.slot);
    if (!instances || instances.capacity < required) {
        const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(required)));
        const stale = instances?.buffer;
        if (stale) world.gpu.device.queue.onSubmittedWorkDone().then(() => stale.destroy());
        const buffer = world.gpu.root
            .createBuffer(d.arrayOf(LineInstance, capacity))
            .$usage("storage");
        const staging = new ArrayBuffer(capacity * d.sizeOf(LineInstance));
        instances = {
            buffer,
            group: null,
            placement: null,
            rows: null,
            staging,
            floats: new Float32Array(staging),
            ints: new Uint32Array(staging),
            capacity,
        };
        state.instances.set(view.slot, instances);
    }
    const table = globalTransformTable(world);
    const rows = table.eidToRowTyped;
    if (!rows) return;
    if (instances.placement !== table.typed || instances.rows !== rows) {
        instances.placement = table.typed;
        instances.rows = rows;
        instances.group = world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(instanceLayout, {
                instances: instances.buffer,
                globalTransforms: table.typed as TgpuBuffer<d.WgslArray<typeof Xform>> &
                    StorageFlag,
                globalTransformRows: rows as TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag,
            }),
        );
    }
    const { floats, ints } = instances;
    const cameraLayers = renderLayers(world, camera);
    floats.fill(0, 0, INSTANCE_WORDS);
    floats[1] = -1;
    floats[2] = 1;
    ints[4] = cameraLayers;
    const assets = world.resource(gizmoAssets);
    let count = 1;
    for (let row = 1; row <= records.count; row++) {
        const source = row * INSTANCE_WORDS;
        if (!(records.ints[source + 4] & cameraLayers)) continue;
        const at = assets[records.handles[row] - 1].cursor++ * INSTANCE_WORDS;
        for (let word = 0; word < INSTANCE_WORDS; word++)
            ints[at + word] = records.ints[source + word];
        count++;
    }
    world.gpu.device.queue.writeBuffer(
        world.gpu.root.unwrap(instances.buffer),
        0,
        instances.staging,
        0,
        count * INSTANCE_BYTES,
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
    if (!viewGroup || !instances?.group || !group || !count) return;
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
    drawInstances(pass, 6 * count, asset ? asset.count : end - start, asset ? asset.start : 0);
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

/** Screen-space pixel-width lines: immediate geometry sorts as one item; retained gizmos draw unsorted per asset after sorted transparency. */
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
