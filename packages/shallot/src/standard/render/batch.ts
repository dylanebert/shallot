import { capacity, write, type Buf } from "../../engine";
import { Shape } from "../../engine/utils";
import { beginComputePass, type ComputeNode, type ExecutionContext } from "../compute";
import { bindView, binding, type Binding, type BufferView, type GBuf, gbuf } from "../compute";
import {
    MAX_SURFACES,
    MAX_SHAPES,
    MAX_BATCH_SLOTS,
    getMesh,
    getMeshVersion,
    meshCount,
    computeShapeAABB,
    isUnboundedShape,
    type ShapeAtlas,
} from "./mesh";
import { surfaceRegistry } from "./surface";
export const INDIRECT_STRIDE = 5;
export const CULL_ENTITY_STRIDE = 2;
export const TRANSPARENT_COUNTER = MAX_BATCH_SLOTS * 2 + 1;
const SCATTER_COUNTERS_SIZE = MAX_BATCH_SLOTS * 2 + 2;
function partMaskSize() {
    return capacity() >>> 5;
}

export interface Batching {
    entityIds: GBuf;
    indirect: GPUBuffer;
    slotCounts: GPUBuffer;
    entityBatchInfo: GBuf;
    scatterCounters: GPUBuffer;
    transparentEntities: GBuf;
    cullEntities: GBuf;
    activeSlotsGPU: GPUBuffer;
    prefixParams: GPUBuffer;
    resolveInputBuffer: GBuf;
    resolveParamsBuffer: GPUBuffer;
    cullEntityCount: number;
    shapeAABBs: Float32Array;
    activeSlots: Uint32Array;
    activeSlotCount: number;
    partMask: Uint32Array;
}

function resolveShapesOffset() {
    return 0;
}
function resolveMeshGeomOffset() {
    return capacity() >>> 2;
}
function resolveSurfacesOffset() {
    return resolveMeshGeomOffset() + capacity();
}
function resolveVolumesOffset() {
    return resolveSurfacesOffset() + (capacity() >>> 1);
}
function resolveMaskOffset() {
    return resolveVolumesOffset() + (capacity() >>> 2);
}
function resolveCountOffset() {
    return resolveMaskOffset() + partMaskSize();
}
function resolveTotalU32s() {
    return resolveCountOffset() + 1;
}
export const singleU32 = new Uint32Array(1);

export function createBatching(device: GPUDevice): Batching {
    const StorageDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const resolveParamsBuffer = device.createBuffer({
        label: "resolve-params",
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(resolveParamsBuffer, 0, new Uint32Array([capacity()]));

    const b: Batching = {
        entityIds: gbuf(device, "batch-entity-ids", StorageDst, (c) => c * 4),
        indirect: device.createBuffer({
            label: "batch-indirect",
            size: MAX_BATCH_SLOTS * 2 * INDIRECT_STRIDE * 4,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        }),
        slotCounts: device.createBuffer({
            label: "batch-slot-counts",
            size: MAX_BATCH_SLOTS * 2 * 4,
            usage: StorageDst,
        }),
        entityBatchInfo: gbuf(device, "batch-entity-info", StorageDst, (c) => c * 4),
        scatterCounters: device.createBuffer({
            label: "batch-scatter-counters",
            size: SCATTER_COUNTERS_SIZE * 4,
            usage: StorageDst,
        }),
        transparentEntities: gbuf(device, "batch-transparent-entities", StorageDst, (c) => c * 4),
        cullEntities: gbuf(
            device,
            "batch-cull-entities",
            StorageDst,
            (c) => c * CULL_ENTITY_STRIDE * 4,
        ),
        activeSlotsGPU: device.createBuffer({
            label: "batch-active-slots",
            size: MAX_BATCH_SLOTS * 4,
            usage: StorageDst,
        }),
        prefixParams: device.createBuffer({
            label: "batch-prefix-params",
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        resolveInputBuffer: gbuf(device, "resolve-input", StorageDst, () => resolveTotalU32s() * 4),
        resolveParamsBuffer,
        cullEntityCount: 0,
        shapeAABBs: new Float32Array(MAX_SHAPES * 6),
        activeSlots: new Uint32Array(MAX_BATCH_SLOTS),
        activeSlotCount: 0,
        partMask: new Uint32Array(partMaskSize()),
    };

    return b;
}

let cachedAABBVersion = -1;

export function ensureShapeAABBs(batching: Batching): void {
    const version = getMeshVersion();
    if (version === cachedAABBVersion) return;
    cachedAABBVersion = version;
    batching.shapeAABBs.fill(0);

    for (let shapeId = 0; shapeId < MAX_SHAPES; shapeId++) {
        const mesh = getMesh(shapeId);
        if (!mesh) continue;
        const o = shapeId * 6;
        if (isUnboundedShape(shapeId)) {
            batching.shapeAABBs[o] = -1e6;
            batching.shapeAABBs[o + 1] = -1e6;
            batching.shapeAABBs[o + 2] = -1e6;
            batching.shapeAABBs[o + 3] = 1e6;
            batching.shapeAABBs[o + 4] = 1e6;
            batching.shapeAABBs[o + 5] = 1e6;
            continue;
        }
        const aabb = computeShapeAABB(mesh);
        batching.shapeAABBs[o] = aabb.minX;
        batching.shapeAABBs[o + 1] = aabb.minY;
        batching.shapeAABBs[o + 2] = aabb.minZ;
        batching.shapeAABBs[o + 3] = aabb.maxX;
        batching.shapeAABBs[o + 4] = aabb.maxY;
        batching.shapeAABBs[o + 5] = aabb.maxZ;
    }
}

export function drawBatches(
    pass: GPURenderPassEncoder,
    indirectBuffer: GPUBuffer,
    indirectOffset: number,
    activeSlots: Uint32Array,
    activeSlotCount: number,
): void {
    for (let j = 0; j < activeSlotCount; j++) {
        const i = activeSlots[j];
        pass.drawIndexedIndirect(indirectBuffer, (indirectOffset + i) * INDIRECT_STRIDE * 4);
    }
}

let cachedSlotMeshVersion = -1;
let cachedSlotSurfaceCount = -1;

export function computeActiveSlots(batching: Batching): void {
    const meshVer = getMeshVersion();
    const surfs = surfaceRegistry.count();
    if (meshVer === cachedSlotMeshVersion && surfs === cachedSlotSurfaceCount) return;
    cachedSlotMeshVersion = meshVer;
    cachedSlotSurfaceCount = surfs;

    const shapes = meshCount();
    let count = 0;
    for (let s = 0; s < shapes; s++) {
        for (let f = 0; f < surfs; f++) {
            batching.activeSlots[count++] = s * MAX_SURFACES + f;
        }
    }
    batching.activeSlotCount = count;
}

export function uploadResolveInputs(
    device: GPUDevice,
    batching: Batching,
    partShapes: Buf,
    meshGeometry: Buf,
    partSurfaces: Buf,
    partVolumes: Buf,
    entityCount: number,
): void {
    const buf = batching.resolveInputBuffer.buffer;
    write(device.queue, buf, resolveShapesOffset() * 4, partShapes, entityCount);
    write(device.queue, buf, resolveMeshGeomOffset() * 4, meshGeometry, entityCount);
    write(device.queue, buf, resolveSurfacesOffset() * 4, partSurfaces, entityCount);
    write(device.queue, buf, resolveVolumesOffset() * 4, partVolumes, entityCount);
    device.queue.writeBuffer(
        buf,
        resolveMaskOffset() * 4,
        batching.partMask as Uint32Array<ArrayBuffer>,
    );
    singleU32[0] = entityCount;
    device.queue.writeBuffer(buf, resolveCountOffset() * 4, singleU32 as Uint32Array<ArrayBuffer>);
}

const RESOLVE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> resolveInput: array<u32>;
@group(0) @binding(1) var<storage, read_write> outShapes: array<u32>;
@group(0) @binding(2) var<storage, read_write> outSurfaces: array<u32>;
@group(0) @binding(3) var<storage, read> sizes: array<vec4<f32>>;

struct ResolveParams { capacity: u32 }
@group(0) @binding(4) var<uniform> resolveParams: ResolveParams;

const INVALID_SHAPE: u32 = 0xFFFFFFFFu;

const SHAPE_BOX: u32 = ${Shape.Box}u;
const SHAPE_SPHERE: u32 = ${Shape.Sphere}u;
const SHAPE_CAPSULE: u32 = ${Shape.Capsule}u;
const SHAPE_PLANE: u32 = ${Shape.Plane}u;
const SHAPE_MESH: u32 = ${Shape.Mesh}u;

fn unpackU8(offset: u32, index: u32) -> u32 {
    let word = resolveInput[offset + (index >> 2u)];
    return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

fn unpackU16(offset: u32, index: u32) -> u32 {
    let word = resolveInput[offset + (index >> 1u)];
    return (word >> ((index & 1u) * 16u)) & 0xFFFFu;
}

fn shapeToPrimitive(shape: u32) -> u32 {
    if (shape == SHAPE_MESH) { return 7u; }
    return shape & 7u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cap = resolveParams.capacity;
    let SHAPES_OFFSET = 0u;
    let MESH_GEOM_OFFSET = cap >> 2u;
    let SURFACES_OFFSET = MESH_GEOM_OFFSET + cap;
    let VOLUMES_OFFSET = SURFACES_OFFSET + (cap >> 1u);
    let MASK_OFFSET = VOLUMES_OFFSET + (cap >> 2u);
    let COUNT_OFFSET = MASK_OFFSET + (cap >> 5u);

    let eid = gid.x;
    if (eid >= resolveInput[COUNT_OFFSET]) { return; }

    let maskWord = resolveInput[MASK_OFFSET + (eid >> 5u)];
    if ((maskWord & (1u << (eid & 31u))) == 0u) {
        outShapes[eid] = INVALID_SHAPE;
        return;
    }

    let shape = unpackU8(SHAPES_OFFSET, eid);
    var shapeId: u32;
    switch (shape) {
        case SHAPE_BOX: { shapeId = 0u; }
        case SHAPE_SPHERE: { shapeId = 1u; }
        case SHAPE_CAPSULE: { shapeId = 2u; }
        case SHAPE_PLANE: { shapeId = 3u; }
        case SHAPE_MESH: { shapeId = resolveInput[MESH_GEOM_OFFSET + eid]; }
        default: { shapeId = 0u; }
    }
    outShapes[eid] = shapeId;

    let surf = unpackU16(SURFACES_OFFSET, eid);
    let vol = unpackU8(VOLUMES_OFFSET, eid);
    let hasShadows = select(0u, 1u, sizes[eid].w != 0.0);
    let prim = shapeToPrimitive(shape);

    outSurfaces[eid] = (surf & 0xFFu)
        | ((vol & 0xFu) << 8u)
        | (hasShadows << 12u)
        | (prim << 13u)
        | ((shapeId & 0xFFFFu) << 16u);
}
`;

const COUNT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> shapes: array<u32>;
@group(0) @binding(1) var<storage, read> surfaces: array<u32>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> slotCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> entityBatchInfo: array<u32>;
@group(0) @binding(5) var<storage, read> entityCount: array<u32>;

const INVALID_SHAPE: u32 = 0xFFFFFFFFu;
const MAX_SURFACES: u32 = ${MAX_SURFACES}u;
const MAX_BATCH_SLOTS: u32 = ${MAX_BATCH_SLOTS}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= entityCount[0]) { return; }

    let shapeId = shapes[eid];
    if (shapeId == INVALID_SHAPE) {
        entityBatchInfo[eid] = INVALID_SHAPE;
        return;
    }

    let surfaceId = surfaces[eid] & 0xFFu;
    let batchIndex = shapeId * MAX_SURFACES + surfaceId;
    if (batchIndex >= MAX_BATCH_SLOTS) {
        entityBatchInfo[eid] = INVALID_SHAPE;
        return;
    }

    let alpha = colors[eid].w;
    let isTransparent = select(0u, 1u, alpha < 1.0);
    let slotIndex = batchIndex + isTransparent * MAX_BATCH_SLOTS;

    atomicAdd(&slotCounts[slotIndex], 1u);
    entityBatchInfo[eid] = batchIndex | (isTransparent << 31u);
}
`;

const PREFIX_SUM_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> slotCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(2) var<storage, read> meshMeta: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> activeSlots: array<u32>;
@group(0) @binding(4) var<uniform> params: vec2<u32>;

const MAX_SURFACES: u32 = ${MAX_SURFACES}u;
const MAX_BATCH_SLOTS: u32 = ${MAX_BATCH_SLOTS}u;
const INDIRECT_STRIDE: u32 = 5u;

fn writeSlot(slotIndex: u32, batchIndex: u32, offset: ptr<function, u32>) {
    let count = slotCounts[slotIndex];
    let iBase = slotIndex * INDIRECT_STRIDE;

    if (count > 0u) {
        let shapeId = batchIndex / MAX_SURFACES;
        let sm = meshMeta[shapeId];
        indirect[iBase] = sm.z * 3u;
        indirect[iBase + 1u] = count;
        indirect[iBase + 2u] = sm.y;
        indirect[iBase + 3u] = 0u;
        indirect[iBase + 4u] = *offset;
    } else {
        indirect[iBase + 1u] = 0u;
    }

    *offset += count;
}

@compute @workgroup_size(1)
fn main() {
    var offset: u32 = 0u;
    let slotCount = params.x;

    for (var i: u32 = 0u; i < slotCount; i++) {
        writeSlot(activeSlots[i], activeSlots[i], &offset);
    }
    for (var i: u32 = 0u; i < slotCount; i++) {
        writeSlot(activeSlots[i] + MAX_BATCH_SLOTS, activeSlots[i], &offset);
    }
}
`;

const SCATTER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> entityBatchInfo: array<u32>;
@group(0) @binding(1) var<storage, read> indirect: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> entityIds: array<u32>;
@group(0) @binding(4) var<storage, read_write> cullEntities: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> entityCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> transparentEntities: array<u32>;

const INVALID_SHAPE: u32 = 0xFFFFFFFFu;
const MAX_BATCH_SLOTS: u32 = ${MAX_BATCH_SLOTS}u;
const INDIRECT_STRIDE: u32 = 5u;
const CULL_COUNTER: u32 = ${MAX_BATCH_SLOTS * 2}u;
const TRANSPARENT_COUNTER: u32 = ${TRANSPARENT_COUNTER}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= entityCount[0]) { return; }

    let info = entityBatchInfo[eid];
    if (info == INVALID_SHAPE) { return; }

    let batchIndex = info & 0x7FFFFFFFu;
    let isTransparent = info >> 31u;
    let slotIndex = batchIndex + isTransparent * MAX_BATCH_SLOTS;

    let firstInstance = indirect[slotIndex * INDIRECT_STRIDE + 4u];
    let localIdx = atomicAdd(&scatterCounters[slotIndex], 1u);
    entityIds[firstInstance + localIdx] = eid;

    if (isTransparent == 1u) {
        let transIdx = atomicAdd(&scatterCounters[TRANSPARENT_COUNTER], 1u);
        transparentEntities[transIdx] = eid;
    }

    let cullIdx = atomicAdd(&scatterCounters[CULL_COUNTER], 1u);
    cullEntities[cullIdx] = vec2(eid, slotIndex);
}
`;

export function createBatchComputeNode(render: {
    shapes: BufferView;
    surfaces: BufferView;
    colors: BufferView;
    sizes: BufferView;
    meshAtlas: ShapeAtlas;
    meshVersion: number;
    entityCountBuffer: BufferView;
    batching: Batching;
    entityCount: number;
}): ComputeNode {
    let resolvePipeline: GPUComputePipeline | null = null;
    let countPipeline: GPUComputePipeline | null = null;
    let prefixPipeline: GPUComputePipeline | null = null;
    let scatterPipeline: GPUComputePipeline | null = null;

    let resolveBG: Binding | null = null;
    let countBG: Binding | null = null;
    let prefixBG: GPUBindGroup | null = null;
    let scatterBG: Binding | null = null;
    const prefixParamsData = new Uint32Array(2);
    const resolveParamsScratch = new Uint32Array(1);
    let cachedCapacity = capacity();
    let cachedMeshVer = render.meshVersion;

    return {
        name: "batch-compute",
        scope: "frame",
        inputs: ["data"],
        outputs: ["batched"],

        async prepare(device: GPUDevice) {
            const b = render.batching;

            const resolveModule = device.createShaderModule({ code: RESOLVE_SHADER });
            const countModule = device.createShaderModule({ code: COUNT_SHADER });
            const prefixModule = device.createShaderModule({ code: PREFIX_SUM_SHADER });
            const scatterModule = device.createShaderModule({ code: SCATTER_SHADER });

            [resolvePipeline, countPipeline, prefixPipeline, scatterPipeline] = await Promise.all([
                device.createComputePipelineAsync({
                    label: "batch-resolve",
                    layout: "auto",
                    compute: { module: resolveModule, entryPoint: "main" },
                }),
                device.createComputePipelineAsync({
                    label: "batch-count",
                    layout: "auto",
                    compute: { module: countModule, entryPoint: "main" },
                }),
                device.createComputePipelineAsync({
                    label: "batch-prefix",
                    layout: "auto",
                    compute: { module: prefixModule, entryPoint: "main" },
                }),
                device.createComputePipelineAsync({
                    label: "batch-scatter",
                    layout: "auto",
                    compute: { module: scatterModule, entryPoint: "main" },
                }),
            ]);

            resolveBG = binding(device, resolvePipeline.getBindGroupLayout(0), () => [
                { binding: 0, resource: { buffer: b.resolveInputBuffer.buffer } },
                bindView(1, render.shapes),
                bindView(2, render.surfaces),
                bindView(3, render.sizes),
                { binding: 4, resource: { buffer: b.resolveParamsBuffer } },
            ]);

            countBG = binding(device, countPipeline.getBindGroupLayout(0), () => [
                bindView(0, render.shapes),
                bindView(1, render.surfaces),
                bindView(2, render.colors),
                { binding: 3, resource: { buffer: b.slotCounts } },
                { binding: 4, resource: { buffer: b.entityBatchInfo.buffer } },
                bindView(5, render.entityCountBuffer),
            ]);

            prefixBG = device.createBindGroup({
                layout: prefixPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: b.slotCounts } },
                    { binding: 1, resource: { buffer: b.indirect } },
                    { binding: 2, resource: { buffer: render.meshAtlas.meta } },
                    { binding: 3, resource: { buffer: b.activeSlotsGPU } },
                    { binding: 4, resource: { buffer: b.prefixParams } },
                ],
            });

            scatterBG = binding(device, scatterPipeline.getBindGroupLayout(0), () => [
                { binding: 0, resource: { buffer: b.entityBatchInfo.buffer } },
                { binding: 1, resource: { buffer: b.indirect } },
                { binding: 2, resource: { buffer: b.scatterCounters } },
                { binding: 3, resource: { buffer: b.entityIds.buffer } },
                { binding: 4, resource: { buffer: b.cullEntities.buffer } },
                bindView(5, render.entityCountBuffer),
                { binding: 6, resource: { buffer: b.transparentEntities.buffer } },
            ]);
        },

        execute(ctx: ExecutionContext) {
            if (!resolvePipeline || !countPipeline || !prefixPipeline || !scatterPipeline) return;
            if (!resolveBG || !countBG || !prefixBG || !scatterBG) return;

            const entityCount = render.entityCount;
            if (entityCount === 0) return;

            const b = render.batching;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                resolveParamsScratch[0] = cachedCapacity;
                ctx.device.queue.writeBuffer(b.resolveParamsBuffer, 0, resolveParamsScratch);
            }

            if (render.meshVersion !== cachedMeshVer && prefixPipeline) {
                cachedMeshVer = render.meshVersion;
                prefixBG = ctx.device.createBindGroup({
                    layout: prefixPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: b.slotCounts } },
                        { binding: 1, resource: { buffer: b.indirect } },
                        { binding: 2, resource: { buffer: render.meshAtlas.meta } },
                        { binding: 3, resource: { buffer: b.activeSlotsGPU } },
                        { binding: 4, resource: { buffer: b.prefixParams } },
                    ],
                });
            }

            ctx.encoder.clearBuffer(b.slotCounts);
            ctx.encoder.clearBuffer(b.scatterCounters);

            computeActiveSlots(b);
            prefixParamsData[0] = b.activeSlotCount;
            ctx.device.queue.writeBuffer(b.prefixParams, 0, prefixParamsData);
            ctx.device.queue.writeBuffer(
                b.activeSlotsGPU,
                0,
                b.activeSlots.buffer as ArrayBuffer,
                0,
                b.activeSlotCount * 4,
            );

            const resolvePass = beginComputePass(
                ctx.encoder,
                ctx.timestampWrites?.("batch-resolve"),
            );
            resolvePass.setPipeline(resolvePipeline);
            resolvePass.setBindGroup(0, resolveBG.group);
            resolvePass.dispatchWorkgroups(Math.ceil(entityCount / 64));
            resolvePass.end();

            const countPass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("batch-count"));
            countPass.setPipeline(countPipeline);
            countPass.setBindGroup(0, countBG.group);
            countPass.dispatchWorkgroups(Math.ceil(entityCount / 64));
            countPass.end();

            const prefixPass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("batch-prefix"));
            prefixPass.setPipeline(prefixPipeline);
            prefixPass.setBindGroup(0, prefixBG);
            prefixPass.dispatchWorkgroups(1);
            prefixPass.end();

            const scatterPass = beginComputePass(
                ctx.encoder,
                ctx.timestampWrites?.("batch-scatter"),
            );
            scatterPass.setPipeline(scatterPipeline);
            scatterPass.setBindGroup(0, scatterBG.group);
            scatterPass.dispatchWorkgroups(Math.ceil(entityCount / 64));
            scatterPass.end();

            ensureShapeAABBs(b);
        },
    };
}
