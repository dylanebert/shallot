import {
    beginComputePass,
    type ComputeNode,
    type ExecutionContext,
} from "../../../standard/compute";
import { bindView, type GBuf, type BufferView } from "../../../standard/compute";
import { capacity } from "../../../engine";
import type { BLASAtlas } from "./blas";
import {
    isDynamic,
    dynamicInfo,
    type ShapeAtlas,
    type SurfaceData,
    hasProperties,
    instanceStructWGSL,
    instanceBindingWGSL,
    compileVertexBody,
    SCENE_STRUCT_WGSL,
    SKY_STRUCT_WGSL,
} from "../../../standard/render/core";
import {
    BLAS_TRIANGLE_STRUCT_WGSL,
    TREE_NODE_STRUCT_WGSL,
    LEAF_FLAG_WGSL,
    OCT_DECODE_WGSL,
    MAX_PROPAGATION_ITERS,
} from "./structs";

const WORKGROUP_SIZE = 64;
const MAX_DYNAMIC_DISPATCHES = 256;

const DISPLACE_PARAM_SIZE = 16;
const ATLAS_COPY_PARAM_SIZE = 16;
const PROPAGATE_PARAM_SIZE = 32;
const AABB_PARAM_SIZE = 16;

const OCT_ENCODE_WGSL = /* wgsl */ `
fn octEncode(n: vec3<f32>) -> u32 {
    let absSum = abs(n.x) + abs(n.y) + abs(n.z);
    var vx = n.x / absSum;
    var vy = n.y / absSum;
    let vz = n.z / absSum;
    if (vz < 0.0) {
        let signX = select(-1.0, 1.0, vx >= 0.0);
        let signY = select(-1.0, 1.0, vy >= 0.0);
        let newVx = (1.0 - abs(vy)) * signX;
        let newVy = (1.0 - abs(vx)) * signY;
        vx = newVx;
        vy = newVy;
    }
    let x = u32(clamp((vx * 0.5 + 0.5) * 65535.0, 0.0, 65535.0));
    let y = u32(clamp((vy * 0.5 + 0.5) * 65535.0, 0.0, 65535.0));
    return (y << 16u) | x;
}`;

const ATLAS_COPY_SHADER = /* wgsl */ `
${BLAS_TRIANGLE_STRUCT_WGSL}
${OCT_ENCODE_WGSL}

struct CopyParams {
    atlasFloatOffset: u32,
    atlasIndexOffset: u32,
    triCount: u32,
    outTriOffset: u32,
}

@group(0) @binding(0) var<storage, read> atlasVerts: array<f32>;
@group(0) @binding(1) var<storage, read> atlasIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> triangles: array<BLASTriangle>;
@group(0) @binding(3) var<uniform> params: CopyParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.triCount) { return; }

    let idxBase = params.atlasIndexOffset + gid.x * 3u;
    let i0 = atlasIndices[idxBase];
    let i1 = atlasIndices[idxBase + 1u];
    let i2 = atlasIndices[idxBase + 2u];

    let stride = 8u;
    let b0 = params.atlasFloatOffset + i0 * stride;
    let b1 = params.atlasFloatOffset + i1 * stride;
    let b2 = params.atlasFloatOffset + i2 * stride;

    let v0 = vec3(atlasVerts[b0], atlasVerts[b0 + 1u], atlasVerts[b0 + 2u]);
    let v1 = vec3(atlasVerts[b1], atlasVerts[b1 + 1u], atlasVerts[b1 + 2u]);
    let v2 = vec3(atlasVerts[b2], atlasVerts[b2 + 1u], atlasVerts[b2 + 2u]);

    let n0 = vec3(atlasVerts[b0 + 3u], atlasVerts[b0 + 4u], atlasVerts[b0 + 5u]);
    let n1 = vec3(atlasVerts[b1 + 3u], atlasVerts[b1 + 4u], atlasVerts[b1 + 5u]);
    let n2 = vec3(atlasVerts[b2 + 3u], atlasVerts[b2 + 4u], atlasVerts[b2 + 5u]);

    var tri: BLASTriangle;
    tri.v0 = v0;
    tri.e1 = v1 - v0;
    tri.e2 = v2 - v0;
    tri._pad0 = 0u;
    tri._pad1 = 0u;
    tri._pad2 = 0u;
    tri.n0_enc = octEncode(n0);
    tri.n1_enc = octEncode(n1);
    tri.n2_enc = octEncode(n2);
    tri._pad3 = 0u;

    triangles[params.outTriOffset + gid.x] = tri;
}
`;

function compileDisplacementShader(surfaceData: SurfaceData): string {
    const vertexBody = compileVertexBody(surfaceData.vertex);
    const needsInstance =
        surfaceData.properties &&
        surfaceData.properties.length > 0 &&
        surfaceData.vertex?.includes("inst.");

    const instanceStructDecl = needsInstance ? instanceStructWGSL() : "";
    const instanceBindingDecl = needsInstance ? instanceBindingWGSL(5) : "";
    const instancePreamble = needsInstance ? "let inst = instanceData[params.eid];" : "";

    return /* wgsl */ `
${SCENE_STRUCT_WGSL}
${SKY_STRUCT_WGSL}
${BLAS_TRIANGLE_STRUCT_WGSL}
${OCT_DECODE_WGSL}

struct Params {
    triOffset: u32,
    triCount: u32,
    outTriOffset: u32,
    eid: u32,
}

${instanceStructDecl}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> sky: Sky;
@group(0) @binding(2) var<storage, read> baseTriangles: array<BLASTriangle>;
@group(0) @binding(3) var<storage, read_write> triangles: array<BLASTriangle>;
@group(0) @binding(4) var<uniform> params: Params;
${instanceBindingDecl}
@group(0) @binding(6) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(7) var<storage, read> sizes: array<vec4<f32>>;

struct VertexTransformResult {
    position: vec3<f32>,
    uv: vec2<f32>,
}

fn displaceVertex(localPos: vec3<f32>, normal: vec3<f32>, meshUv: vec2<f32>, eid: u32) -> VertexTransformResult {
    ${instancePreamble}
    ${vertexBody}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.triCount) { return; }
    let base = baseTriangles[params.triOffset + gid.x];

    let v0 = base.v0;
    let v1 = v0 + base.e1;
    let v2 = v0 + base.e2;
    let n0 = octDecode(base.n0_enc);
    let n1 = octDecode(base.n1_enc);
    let n2 = octDecode(base.n2_enc);

    let dv0 = displaceVertex(v0, n0, vec2(0.0), params.eid).position;
    let dv1 = displaceVertex(v1, n1, vec2(0.0), params.eid).position;
    let dv2 = displaceVertex(v2, n2, vec2(0.0), params.eid).position;

    var tri: BLASTriangle;
    tri.v0 = dv0;
    tri.e1 = dv1 - dv0;
    tri.e2 = dv2 - dv0;
    tri._pad0 = 0u;
    tri._pad1 = 0u;
    tri._pad2 = 0u;
    tri.n0_enc = base.n0_enc;
    tri.n1_enc = base.n1_enc;
    tri.n2_enc = base.n2_enc;
    tri._pad3 = 0u;
    triangles[params.outTriOffset + gid.x] = tri;
}
`;
}

const propagateShader = /* wgsl */ `
${BLAS_TRIANGLE_STRUCT_WGSL}
${LEAF_FLAG_WGSL}

struct PropagateParams {
    triOffset: u32,
    triIdOffset: u32,
    treeNodeOffset: u32,
    parentOffset: u32,
    triCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> triangles: array<BLASTriangle>;
@group(0) @binding(1) var<storage, read> triIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> treeNodesRaw: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> boundsFlags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> parentIndices: array<u32>;
@group(0) @binding(5) var<uniform> params: PropagateParams;

fn isLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn leafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}

fn getTriangleBounds(leafIdx: u32) -> array<vec3<f32>, 2> {
    let triId = triIds[params.triIdOffset + leafIdx];
    let tri = triangles[params.triOffset + triId];
    let v0 = tri.v0;
    let v1 = v0 + tri.e1;
    let v2 = v0 + tri.e2;
    return array<vec3<f32>, 2>(min(min(v0, v1), v2), max(max(v0, v1), v2));
}

fn nodeBase(idx: u32) -> u32 {
    return (params.treeNodeOffset + idx) * 8u;
}

fn readChildBounds(childIdx: u32) -> array<vec3<f32>, 2> {
    let base = nodeBase(childIdx);
    let minX = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 0u]));
    let minY = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 1u]));
    let minZ = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 2u]));
    let maxX = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 4u]));
    let maxY = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 5u]));
    let maxZ = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 6u]));
    return array<vec3<f32>, 2>(vec3(minX, minY, minZ), vec3(maxX, maxY, maxZ));
}

fn writeBounds(nodeIdx: u32, minB: vec3<f32>, maxB: vec3<f32>) {
    let base = nodeBase(nodeIdx);
    atomicStore(&treeNodesRaw[base + 0u], bitcast<u32>(minB.x));
    atomicStore(&treeNodesRaw[base + 1u], bitcast<u32>(minB.y));
    atomicStore(&treeNodesRaw[base + 2u], bitcast<u32>(minB.z));
    atomicStore(&treeNodesRaw[base + 4u], bitcast<u32>(maxB.x));
    atomicStore(&treeNodesRaw[base + 5u], bitcast<u32>(maxB.y));
    atomicStore(&treeNodesRaw[base + 6u], bitcast<u32>(maxB.z));
}

fn readLeftChild(nodeIdx: u32) -> u32 {
    return atomicLoad(&treeNodesRaw[nodeBase(nodeIdx) + 3u]);
}

fn readRightChild(nodeIdx: u32) -> u32 {
    return atomicLoad(&treeNodesRaw[nodeBase(nodeIdx) + 7u]);
}

fn writeLeafBounds(leafIdx: u32, minB: vec3<f32>, maxB: vec3<f32>) {
    let n = params.triCount;
    let leafNodeIdx = n - 1u + leafIdx;
    writeBounds(leafNodeIdx, minB, maxB);
}

fn getParent(nodeIdx: u32, isLeafNode: bool) -> u32 {
    let n = params.triCount;
    if (isLeafNode) {
        return parentIndices[params.parentOffset + nodeIdx];
    } else {
        return parentIndices[params.parentOffset + n + nodeIdx];
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.triCount;
    let leafIdx = gid.x;

    if (leafIdx >= n) { return; }

    let bounds = getTriangleBounds(leafIdx);
    writeLeafBounds(leafIdx, bounds[0], bounds[1]);

    if (n <= 1u) { return; }

    var current = leafIdx;
    var isLeafNode = true;

    for (var iter = 0u; iter < ${MAX_PROPAGATION_ITERS}u; iter++) {
        let parent = getParent(current, isLeafNode);

        let oldFlag = atomicAdd(&boundsFlags[params.treeNodeOffset + parent], 1u);
        if (oldFlag == 0u) { return; }

        let left = readLeftChild(parent);
        let right = readRightChild(parent);

        var leftMin: vec3<f32>;
        var leftMax: vec3<f32>;
        var rightMin: vec3<f32>;
        var rightMax: vec3<f32>;

        if (isLeaf(left)) {
            let leftBounds = getTriangleBounds(leafIndex(left));
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        } else {
            let leftBounds = readChildBounds(left);
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        }

        if (isLeaf(right)) {
            let rightBounds = getTriangleBounds(leafIndex(right));
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        } else {
            let rightBounds = readChildBounds(right);
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        }

        writeBounds(parent, min(leftMin, rightMin), max(leftMax, rightMax));

        current = parent;
        isLeafNode = false;

        if (parent == 0u) { break; }
    }
}
`;

const aabbShader = /* wgsl */ `
struct ShapeAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}

${TREE_NODE_STRUCT_WGSL}

struct AABBParams {
    treeNodeOffset: u32,
    entityId: u32,
    instanceSlot: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> treeNodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read_write> perEntityShapeAABBs: array<ShapeAABB>;
@group(0) @binding(2) var<uniform> params: AABBParams;

@compute @workgroup_size(1)
fn main() {
    let root = treeNodes[params.treeNodeOffset];
    var aabb: ShapeAABB;
    aabb.minX = root.minX;
    aabb.minY = root.minY;
    aabb.minZ = root.minZ;
    aabb._pad0 = params.instanceSlot;
    aabb.maxX = root.maxX;
    aabb.maxY = root.maxY;
    aabb.maxZ = root.maxZ;
    aabb._pad1 = 0u;
    perEntityShapeAABBs[params.entityId] = aabb;
}
`;

interface DisplacementPipeline {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
}

export interface DynamicShapeInfo {
    surface: SurfaceData;
    entities: number[];
}

interface RefitGPU {
    propagatePipeline: GPUComputePipeline;
    propagateLayout: GPUBindGroupLayout;
    aabbPipeline: GPUComputePipeline;
    aabbLayout: GPUBindGroupLayout;
    displacementPipelines: Map<string, DisplacementPipeline>;
    displacementBindGroupLayout: GPUBindGroupLayout;
    displacementPipelineLayout: GPUPipelineLayout;
    atlasCopyPipeline: GPUComputePipeline;
    atlasCopyBindGroup: GPUBindGroup | null;
    atlasCopyParams: GPUBuffer;
    atlasCopyStaging: GPUBuffer;
    atlasCopyData: Uint32Array<ArrayBuffer>;
    cachedMeshAtlasVerts: GPUBuffer | null;
    displaceParams: GPUBuffer;
    propagateParams: GPUBuffer;
    aabbParams: GPUBuffer;
    displaceStaging: GPUBuffer;
    propagateStaging: GPUBuffer;
    aabbStaging: GPUBuffer;
    displaceData: Uint32Array<ArrayBuffer>;
    propagateData: Uint32Array<ArrayBuffer>;
    aabbData: Uint32Array<ArrayBuffer>;
    propagateBindGroup: GPUBindGroup | null;
    aabbBindGroup: GPUBindGroup | null;
    cachedAtlas: BLASAtlas | null;
    cachedCapacity: number;
}

export function createBLASRefitNode(
    bvh: { blasAtlas: BLASAtlas },
    render: {
        scene: GPUBuffer;
        sky: GPUBuffer;
        instanceDataBuffer: GBuf | null;
        matrices: GBuf;
        sizes: BufferView;
        meshAtlas: ShapeAtlas;
    },
    getDynamicShapes: () => Map<number, DynamicShapeInfo>,
    isActive: () => boolean,
): ComputeNode {
    let gpu: RefitGPU | null = null;

    return {
        name: "blas-refit",
        scope: "frame",
        inputs: ["dynamic-vertices"],
        outputs: ["blas-nodes"],

        async prepare(device: GPUDevice) {
            const createParams = (label: string, size: number) =>
                device.createBuffer({
                    label,
                    size,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });

            const displaceParams = createParams("blas-refit-displace-params", DISPLACE_PARAM_SIZE);
            const propagateParams = createParams(
                "blas-refit-propagate-params",
                PROPAGATE_PARAM_SIZE,
            );
            const aabbParams = createParams("blas-refit-aabb-params", AABB_PARAM_SIZE);

            const createStaging = (label: string, size: number) =>
                device.createBuffer({
                    label,
                    size,
                    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });

            const displaceStaging = createStaging(
                "blas-refit-displace-staging",
                MAX_DYNAMIC_DISPATCHES * DISPLACE_PARAM_SIZE,
            );
            const propagateStaging = createStaging(
                "blas-refit-propagate-staging",
                MAX_DYNAMIC_DISPATCHES * PROPAGATE_PARAM_SIZE,
            );
            const aabbStaging = createStaging(
                "blas-refit-aabb-staging",
                MAX_DYNAMIC_DISPATCHES * AABB_PARAM_SIZE,
            );

            const entries: GPUBindGroupLayoutEntry[] = [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ];
            if (hasProperties()) {
                entries.push({
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                });
            }
            entries.push(
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
            );
            const displacementBindGroupLayout = device.createBindGroupLayout({ entries });
            const displacementPipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [displacementBindGroupLayout],
            });

            const atlasCopyModule = device.createShaderModule({ code: ATLAS_COPY_SHADER });
            const atlasCopyParams = createParams(
                "blas-refit-atlas-copy-params",
                ATLAS_COPY_PARAM_SIZE,
            );
            const atlasCopyStaging = createStaging(
                "blas-refit-atlas-copy-staging",
                MAX_DYNAMIC_DISPATCHES * ATLAS_COPY_PARAM_SIZE,
            );

            const [propagatePipeline, aabbPipeline, atlasCopyPipeline] = await Promise.all([
                device.createComputePipelineAsync({
                    label: "rt-propagate",
                    layout: "auto",
                    compute: {
                        module: device.createShaderModule({ code: propagateShader }),
                        entryPoint: "main",
                    },
                }),
                device.createComputePipelineAsync({
                    label: "rt-aabb",
                    layout: "auto",
                    compute: {
                        module: device.createShaderModule({ code: aabbShader }),
                        entryPoint: "main",
                    },
                }),
                device.createComputePipelineAsync({
                    label: "rt-atlas-copy",
                    layout: "auto",
                    compute: { module: atlasCopyModule, entryPoint: "main" },
                }),
            ]);

            gpu = {
                propagatePipeline,
                propagateLayout: propagatePipeline.getBindGroupLayout(0),
                aabbPipeline,
                aabbLayout: aabbPipeline.getBindGroupLayout(0),
                displacementPipelines: new Map(),
                displacementBindGroupLayout,
                displacementPipelineLayout,
                atlasCopyPipeline,
                atlasCopyBindGroup: null,
                atlasCopyParams,
                atlasCopyStaging,
                atlasCopyData: new Uint32Array(MAX_DYNAMIC_DISPATCHES * 4),
                cachedMeshAtlasVerts: null,
                displaceParams,
                propagateParams,
                aabbParams,
                displaceStaging,
                propagateStaging,
                aabbStaging,
                displaceData: new Uint32Array(MAX_DYNAMIC_DISPATCHES * 4),
                propagateData: new Uint32Array(MAX_DYNAMIC_DISPATCHES * 8),
                aabbData: new Uint32Array(MAX_DYNAMIC_DISPATCHES * 4),
                propagateBindGroup: null,
                aabbBindGroup: null,
                cachedAtlas: null,
                cachedCapacity: capacity(),
            };
        },

        execute(ctx: ExecutionContext) {
            if (!gpu) return;
            if (!isActive()) return;

            const dynamicShapes = getDynamicShapes();
            if (dynamicShapes.size === 0) return;

            const { device, encoder } = ctx;
            const atlas = bvh.blasAtlas;

            const cap = capacity();
            if (cap !== gpu.cachedCapacity) {
                gpu.displacementPipelines.clear();
                gpu.cachedCapacity = cap;
            }

            if (atlas !== gpu.cachedAtlas) {
                gpu.displacementPipelines.clear();
                gpu.atlasCopyBindGroup = null;

                gpu.propagateBindGroup = device.createBindGroup({
                    layout: gpu.propagateLayout,
                    entries: [
                        { binding: 0, resource: { buffer: atlas.trianglesBuffer } },
                        { binding: 1, resource: { buffer: atlas.triIdsBuffer } },
                        { binding: 2, resource: { buffer: atlas.nodesBuffer } },
                        { binding: 3, resource: { buffer: atlas.boundsFlagsBuffer } },
                        { binding: 4, resource: { buffer: atlas.parentIndicesBuffer } },
                        { binding: 5, resource: { buffer: gpu.propagateParams } },
                    ],
                });

                gpu.aabbBindGroup = device.createBindGroup({
                    layout: gpu.aabbLayout,
                    entries: [
                        { binding: 0, resource: { buffer: atlas.nodesBuffer } },
                        { binding: 1, resource: { buffer: atlas.perEntityShapeAABBs } },
                        { binding: 2, resource: { buffer: gpu.aabbParams } },
                    ],
                });

                gpu.cachedAtlas = atlas;
            }

            const { displaceData, propagateData, aabbData, atlasCopyData } = gpu;
            let dispatchCount = 0;
            let atlasCopyCount = 0;

            for (const [shapeId, { surface: surfaceData, entities }] of dynamicShapes) {
                const meta = atlas.metas[shapeId];
                if (!meta || meta.triCount === 0) continue;
                const n = meta.triCount;
                const nodeCount = n <= 1 ? 1 : 2 * n - 1;
                const useAtlasCopy = entities.some(isDynamic) && !surfaceData.vertex;

                for (let instanceIdx = 0; instanceIdx < entities.length; instanceIdx++) {
                    if (dispatchCount >= MAX_DYNAMIC_DISPATCHES) break;
                    const eid = entities[instanceIdx];
                    const slot = instanceIdx + 1;
                    const outTriOffset = meta.triOffset + slot * n;
                    const outNodeOffset = meta.treeNodeOffset + slot * nodeCount;
                    const d = dispatchCount;

                    if (useAtlasCopy) {
                        const info = dynamicInfo(eid);
                        if (info) {
                            const ac = atlasCopyCount;
                            atlasCopyData[ac * 4 + 0] = info.atlasFloatOffset;
                            atlasCopyData[ac * 4 + 1] = info.atlasIndexOffset;
                            atlasCopyData[ac * 4 + 2] = n;
                            atlasCopyData[ac * 4 + 3] = outTriOffset;
                            atlasCopyCount++;
                        }
                    } else {
                        displaceData[d * 4 + 0] = meta.baseTriOffset;
                        displaceData[d * 4 + 1] = n;
                        displaceData[d * 4 + 2] = outTriOffset;
                        displaceData[d * 4 + 3] = eid;
                    }

                    propagateData[d * 8 + 0] = outTriOffset;
                    propagateData[d * 8 + 1] = meta.triIdOffset;
                    propagateData[d * 8 + 2] = outNodeOffset;
                    propagateData[d * 8 + 3] = meta.parentOffset;
                    propagateData[d * 8 + 4] = n;
                    propagateData[d * 8 + 5] = 0;
                    propagateData[d * 8 + 6] = 0;
                    propagateData[d * 8 + 7] = 0;

                    aabbData[d * 4 + 0] = outNodeOffset;
                    aabbData[d * 4 + 1] = eid;
                    aabbData[d * 4 + 2] = instanceIdx + 1;
                    aabbData[d * 4 + 3] = 0;

                    dispatchCount++;
                }
            }

            if (dispatchCount === 0) return;

            device.queue.writeBuffer(gpu.displaceStaging, 0, displaceData, 0, dispatchCount * 4);
            device.queue.writeBuffer(gpu.propagateStaging, 0, propagateData, 0, dispatchCount * 8);
            device.queue.writeBuffer(gpu.aabbStaging, 0, aabbData, 0, dispatchCount * 4);
            if (atlasCopyCount > 0) {
                device.queue.writeBuffer(
                    gpu.atlasCopyStaging,
                    0,
                    atlasCopyData,
                    0,
                    atlasCopyCount * 4,
                );
            }

            if (
                atlasCopyCount > 0 &&
                (gpu.atlasCopyBindGroup === null ||
                    render.meshAtlas.vertices !== gpu.cachedMeshAtlasVerts)
            ) {
                gpu.cachedMeshAtlasVerts = render.meshAtlas.vertices;
                gpu.atlasCopyBindGroup = device.createBindGroup({
                    layout: gpu.atlasCopyPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: render.meshAtlas.vertices } },
                        { binding: 1, resource: { buffer: render.meshAtlas.indices } },
                        { binding: 2, resource: { buffer: atlas.trianglesBuffer } },
                        { binding: 3, resource: { buffer: gpu.atlasCopyParams } },
                    ],
                });
            }

            let dispatchIdx = 0;
            let atlasCopyIdx = 0;
            for (const [shapeId, { surface: surfaceData, entities }] of dynamicShapes) {
                const meta = atlas.metas[shapeId];
                if (!meta || meta.triCount === 0) continue;
                const useAtlasCopy = entities.some(isDynamic) && !surfaceData.vertex;

                if (!useAtlasCopy && surfaceData.vertex) {
                    const vertexKey = surfaceData.vertex;
                    if (!gpu.displacementPipelines.has(vertexKey)) {
                        const code = compileDisplacementShader(surfaceData);
                        const module = device.createShaderModule({ code });
                        const pipeline = device.createComputePipeline({
                            layout: gpu.displacementPipelineLayout,
                            compute: { module, entryPoint: "main" },
                        });

                        const bgEntries: GPUBindGroupEntry[] = [
                            { binding: 0, resource: { buffer: render.scene } },
                            { binding: 1, resource: { buffer: render.sky } },
                            { binding: 2, resource: { buffer: atlas.baseTrianglesBuffer } },
                            { binding: 3, resource: { buffer: atlas.trianglesBuffer } },
                            { binding: 4, resource: { buffer: gpu.displaceParams } },
                        ];
                        const instBuf = render.instanceDataBuffer;
                        if (hasProperties() && instBuf) {
                            bgEntries.push({ binding: 5, resource: { buffer: instBuf.buffer } });
                        }
                        bgEntries.push(
                            { binding: 6, resource: { buffer: render.matrices.buffer } },
                            bindView(7, render.sizes),
                        );

                        const bindGroup = device.createBindGroup({
                            layout: gpu.displacementBindGroupLayout,
                            entries: bgEntries,
                        });

                        gpu.displacementPipelines.set(vertexKey, { pipeline, bindGroup });
                    }
                }

                const displacement = useAtlasCopy
                    ? null
                    : gpu.displacementPipelines.get(surfaceData.vertex!)!;
                const n = meta.triCount;
                const nodeCount = n <= 1 ? 1 : 2 * n - 1;

                for (let instanceIdx = 0; instanceIdx < entities.length; instanceIdx++) {
                    if (dispatchIdx >= MAX_DYNAMIC_DISPATCHES) break;
                    const slot = instanceIdx + 1;
                    const outNodeOffset = meta.treeNodeOffset + slot * nodeCount;

                    if (useAtlasCopy && gpu.atlasCopyBindGroup) {
                        encoder.copyBufferToBuffer(
                            gpu.atlasCopyStaging,
                            atlasCopyIdx * ATLAS_COPY_PARAM_SIZE,
                            gpu.atlasCopyParams,
                            0,
                            ATLAS_COPY_PARAM_SIZE,
                        );

                        const copyPass = beginComputePass(
                            encoder,
                            ctx.timestampWrites?.("blas-atlas-copy"),
                        );
                        copyPass.setPipeline(gpu.atlasCopyPipeline);
                        copyPass.setBindGroup(0, gpu.atlasCopyBindGroup);
                        copyPass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
                        copyPass.end();
                        atlasCopyIdx++;
                    } else if (displacement) {
                        encoder.copyBufferToBuffer(
                            gpu.displaceStaging,
                            dispatchIdx * DISPLACE_PARAM_SIZE,
                            gpu.displaceParams,
                            0,
                            DISPLACE_PARAM_SIZE,
                        );

                        const displacePass = beginComputePass(
                            encoder,
                            ctx.timestampWrites?.("blas-displace"),
                        );
                        displacePass.setPipeline(displacement.pipeline);
                        displacePass.setBindGroup(0, displacement.bindGroup);
                        displacePass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
                        displacePass.end();
                    }

                    if (n <= 1) {
                        encoder.copyBufferToBuffer(
                            gpu.aabbStaging,
                            dispatchIdx * AABB_PARAM_SIZE,
                            gpu.aabbParams,
                            0,
                            AABB_PARAM_SIZE,
                        );

                        const aabbPass = beginComputePass(
                            encoder,
                            ctx.timestampWrites?.("blas-refit"),
                        );
                        aabbPass.setPipeline(gpu.aabbPipeline);
                        aabbPass.setBindGroup(0, gpu.aabbBindGroup!);
                        aabbPass.dispatchWorkgroups(1);
                        aabbPass.end();
                        dispatchIdx++;
                        continue;
                    }

                    const internalNodes = n - 1;
                    const flagsByteOffset = outNodeOffset * 4;
                    const flagsByteSize = internalNodes * 4;
                    encoder.clearBuffer(atlas.boundsFlagsBuffer, flagsByteOffset, flagsByteSize);

                    encoder.copyBufferToBuffer(
                        gpu.propagateStaging,
                        dispatchIdx * PROPAGATE_PARAM_SIZE,
                        gpu.propagateParams,
                        0,
                        PROPAGATE_PARAM_SIZE,
                    );

                    const propagatePass = beginComputePass(
                        encoder,
                        ctx.timestampWrites?.("blas-refit"),
                    );
                    propagatePass.setPipeline(gpu.propagatePipeline);
                    propagatePass.setBindGroup(0, gpu.propagateBindGroup!);
                    propagatePass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
                    propagatePass.end();

                    encoder.copyBufferToBuffer(
                        gpu.aabbStaging,
                        dispatchIdx * AABB_PARAM_SIZE,
                        gpu.aabbParams,
                        0,
                        AABB_PARAM_SIZE,
                    );

                    const aabbPass = beginComputePass(encoder, ctx.timestampWrites?.("blas-refit"));
                    aabbPass.setPipeline(gpu.aabbPipeline);
                    aabbPass.setBindGroup(0, gpu.aabbBindGroup!);
                    aabbPass.dispatchWorkgroups(1);
                    aabbPass.end();

                    dispatchIdx++;
                }
            }
        },
    };
}
