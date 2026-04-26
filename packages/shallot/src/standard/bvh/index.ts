import {
    createRadixSort,
    rebuildRadixSort,
    dispatchRadixSort,
    disposeRadixSort,
    type RadixSortState,
} from "../radix";
import { beginComputePass } from "../compute";

const WORKGROUP_SIZE = 256;

export const TREE_NODE_SIZE = 32;
export const MORTON_QUANTIZATION = 1023;
export const MAX_PROPAGATION_ITERS = 64;
export const LEAF_FLAG = 0x80000000;
export const AABB_SENTINEL = 1e30;

export const INSTANCE_AABB_STRUCT_WGSL = /* wgsl */ `
struct InstanceAABB {
    minX: f32,
    minY: f32,
    minZ: f32,
    _pad0: u32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
    _pad1: u32,
}`;

export const SCENE_BOUNDS_STRUCT_WGSL = /* wgsl */ `
struct SceneBounds {
    minX: atomic<i32>,
    minY: atomic<i32>,
    minZ: atomic<i32>,
    _pad0: u32,
    maxX: atomic<i32>,
    maxY: atomic<i32>,
    maxZ: atomic<i32>,
    _pad1: u32,
}`;

export const SCENE_BOUNDS_READ_STRUCT_WGSL = /* wgsl */ `
struct SceneBounds {
    minX: i32,
    minY: i32,
    minZ: i32,
    _pad0: u32,
    maxX: i32,
    maxY: i32,
    maxZ: i32,
    _pad1: u32,
}`;

export const FLOAT_INT_CONVERSION_WGSL = /* wgsl */ `
fn floatToSortableInt(f: f32) -> i32 {
    let bits = bitcast<i32>(f);
    let mask = (bits >> 31) & 0x7FFFFFFF;
    return bits ^ mask;
}

fn sortableIntToFloat(i: i32) -> f32 {
    let mask = (i >> 31) & 0x7FFFFFFF;
    return bitcast<f32>(i ^ mask);
}`;

export const MORTON_CODE_WGSL = /* wgsl */ `
fn expandBits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn mortonCode(x: u32, y: u32, z: u32) -> u32 {
    return (expandBits(x) << 2u) | (expandBits(y) << 1u) | expandBits(z);
}`;

export const CLZ_WGSL = /* wgsl */ `
fn clz(x: u32) -> u32 {
    return countLeadingZeros(x);
}`;

export const TREE_NODE_STRUCT_WGSL = /* wgsl */ `
struct TreeNode {
    minX: f32,
    minY: f32,
    minZ: f32,
    leftChild: u32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
    rightChild: u32,
}`;

export const LEAF_FLAG_WGSL = /* wgsl */ `const LEAF_FLAG: u32 = 0x80000000u;`;
export const AABB_SENTINEL_WGSL = /* wgsl */ `const AABB_SENTINEL: f32 = 1e30;`;

export const LEAF_FUNCTIONS_WGSL = /* wgsl */ `
fn isLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn leafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}`;

export const initBounds = new Int32Array([
    0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0, 0x80800000, 0x80800000, 0x80800000, 0,
]);

export interface LBVHConfig {
    leafAABBs: GPUBuffer;
    countBuffer: GPUBuffer;
    maxLeaves: number;
    label: string;
}

export interface LBVH {
    treeNodes: GPUBuffer;
    sortedIds: GPUBuffer;
    mortonCodes: GPUBuffer;
    sceneBounds: GPUBuffer;
    parentIndices: GPUBuffer;
    boundsFlags: GPUBuffer;
    mortonParamsBuffer: GPUBuffer;
    radixSort: RadixSortState;
    config: LBVHConfig;
    pipelines: {
        bounds: GPUComputePipeline;
        morton: GPUComputePipeline;
        tree: GPUComputePipeline;
        propagate: GPUComputePipeline;
    };
    bindGroups: {
        bounds: GPUBindGroup;
        morton: GPUBindGroup;
        tree: GPUBindGroup;
        propagate: GPUBindGroup;
    };
}

const boundsShader = /* wgsl */ `
${INSTANCE_AABB_STRUCT_WGSL}
${SCENE_BOUNDS_STRUCT_WGSL}
${AABB_SENTINEL_WGSL}

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read_write> sceneBounds: SceneBounds;
@group(0) @binding(2) var<storage, read> leafCount: array<u32>;

var<workgroup> sharedMin: array<vec3<f32>, ${WORKGROUP_SIZE}>;
var<workgroup> sharedMax: array<vec3<f32>, ${WORKGROUP_SIZE}>;

${FLOAT_INT_CONVERSION_WGSL}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let count = leafCount[0];
    let tid = gid.x;
    let localId = lid.x;

    var localMin = vec3<f32>(AABB_SENTINEL, AABB_SENTINEL, AABB_SENTINEL);
    var localMax = vec3<f32>(-AABB_SENTINEL, -AABB_SENTINEL, -AABB_SENTINEL);

    if (tid < count) {
        let aabb = leafAABBs[tid];
        localMin = vec3<f32>(aabb.minX, aabb.minY, aabb.minZ);
        localMax = vec3<f32>(aabb.maxX, aabb.maxY, aabb.maxZ);
    }

    sharedMin[localId] = localMin;
    sharedMax[localId] = localMax;
    workgroupBarrier();

    for (var stride = ${WORKGROUP_SIZE}u / 2u; stride > 0u; stride >>= 1u) {
        if (localId < stride) {
            sharedMin[localId] = min(sharedMin[localId], sharedMin[localId + stride]);
            sharedMax[localId] = max(sharedMax[localId], sharedMax[localId + stride]);
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        let wgMin = sharedMin[0];
        let wgMax = sharedMax[0];

        atomicMin(&sceneBounds.minX, floatToSortableInt(wgMin.x));
        atomicMin(&sceneBounds.minY, floatToSortableInt(wgMin.y));
        atomicMin(&sceneBounds.minZ, floatToSortableInt(wgMin.z));
        atomicMax(&sceneBounds.maxX, floatToSortableInt(wgMax.x));
        atomicMax(&sceneBounds.maxY, floatToSortableInt(wgMax.y));
        atomicMax(&sceneBounds.maxZ, floatToSortableInt(wgMax.z));
    }
}
`;

const mortonShader = /* wgsl */ `
${INSTANCE_AABB_STRUCT_WGSL}
${SCENE_BOUNDS_READ_STRUCT_WGSL}

struct MortonParams { capacity: u32 }

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read> sceneBounds: SceneBounds;
@group(0) @binding(2) var<storage, read_write> mortonCodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortedIds: array<u32>;
@group(0) @binding(4) var<storage, read> leafCount: array<u32>;
@group(0) @binding(5) var<uniform> mortonParams: MortonParams;

${FLOAT_INT_CONVERSION_WGSL}
${MORTON_CODE_WGSL}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= mortonParams.capacity) { return; }

    let count = leafCount[0];
    if (tid >= count) {
        mortonCodes[tid] = 0xFFFFFFFFu;
        sortedIds[tid] = 0u;
        return;
    }

    let aabb = leafAABBs[tid];
    let centroid = vec3<f32>(
        (aabb.minX + aabb.maxX) * 0.5,
        (aabb.minY + aabb.maxY) * 0.5,
        (aabb.minZ + aabb.maxZ) * 0.5
    );

    let boundsMin = vec3<f32>(
        sortableIntToFloat(sceneBounds.minX),
        sortableIntToFloat(sceneBounds.minY),
        sortableIntToFloat(sceneBounds.minZ)
    );
    let boundsMax = vec3<f32>(
        sortableIntToFloat(sceneBounds.maxX),
        sortableIntToFloat(sceneBounds.maxY),
        sortableIntToFloat(sceneBounds.maxZ)
    );

    let size = boundsMax - boundsMin;
    let safeSize = max(size, vec3<f32>(1e-6, 1e-6, 1e-6));

    let normalized = (centroid - boundsMin) / safeSize;
    let clamped = clamp(normalized, vec3<f32>(0.0), vec3<f32>(1.0));

    let quantized = vec3<u32>(clamped * ${MORTON_QUANTIZATION}.0);

    mortonCodes[tid] = mortonCode(quantized.x, quantized.y, quantized.z);
    sortedIds[tid] = tid;
}
`;

const treeShader = /* wgsl */ `
${TREE_NODE_STRUCT_WGSL}
${LEAF_FLAG_WGSL}
${AABB_SENTINEL_WGSL}

@group(0) @binding(0) var<storage, read> mortonCodes: array<u32>;
@group(0) @binding(1) var<storage, read_write> treeNodes: array<TreeNode>;
@group(0) @binding(2) var<storage, read_write> parentIndices: array<u32>;
@group(0) @binding(3) var<storage, read> leafCount: array<u32>;

${CLZ_WGSL}

fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) {
        return -1;
    }
    let codeI = mortonCodes[i];
    let codeJ = mortonCodes[j];
    if (codeI == codeJ) {
        return i32(clz(u32(i) ^ u32(j))) + 32;
    }
    return i32(clz(codeI ^ codeJ));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(leafCount[0]);
    let i = i32(gid.x);

    if (i >= n - 1) {
        return;
    }

    var first: i32;
    var last: i32;

    if (i == 0) {
        first = 0;
        last = n - 1;
    } else {
        let d = select(-1, 1, delta(i, i + 1, n) > delta(i, i - 1, n));

        let deltaMin = delta(i, i - d, n);

        var lmax = 2;
        for (var iter = 0; iter < 32; iter++) {
            if (delta(i, i + lmax * d, n) <= deltaMin) { break; }
            lmax *= 2;
        }

        var l = 0;
        var t = lmax / 2;
        for (var iter2 = 0; iter2 < 32; iter2++) {
            if (t < 1) { break; }
            if (delta(i, i + (l + t) * d, n) > deltaMin) {
                l += t;
            }
            t /= 2;
        }

        let j = i + l * d;
        first = min(i, j);
        last = max(i, j);
    }

    let deltaNode = delta(first, last, n);

    var gamma: i32;
    var split = first;
    var stride = last - first;

    for (var iter3 = 0; iter3 < 32; iter3++) {
        stride = (stride + 1) / 2;
        let middle = split + stride;

        if (middle < last) {
            let splitDelta = delta(first, middle, n);

            if (splitDelta > deltaNode) {
                split = middle;
            }
        }

        if (stride <= 1) {
            break;
        }
    }

    gamma = split;

    let leftIsLeaf = first == gamma;
    let rightIsLeaf = last == gamma + 1;

    var node: TreeNode;
    node.minX = AABB_SENTINEL;
    node.minY = AABB_SENTINEL;
    node.minZ = AABB_SENTINEL;
    node.maxX = -AABB_SENTINEL;
    node.maxY = -AABB_SENTINEL;
    node.maxZ = -AABB_SENTINEL;

    if (leftIsLeaf) {
        node.leftChild = u32(gamma) | LEAF_FLAG;
        parentIndices[u32(gamma)] = u32(i);
    } else {
        node.leftChild = u32(gamma);
        parentIndices[u32(n) + u32(gamma)] = u32(i);
    }

    if (rightIsLeaf) {
        node.rightChild = u32(gamma + 1) | LEAF_FLAG;
        parentIndices[u32(gamma + 1)] = u32(i);
    } else {
        node.rightChild = u32(gamma + 1);
        parentIndices[u32(n) + u32(gamma + 1)] = u32(i);
    }

    treeNodes[i] = node;
}
`;

const propagateShader = /* wgsl */ `
${INSTANCE_AABB_STRUCT_WGSL}
${LEAF_FLAG_WGSL}

const BOUNDS_SENTINEL: u32 = 0x7f800000u;

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read> sortedIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> treeNodesRaw: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> boundsFlags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> parentIndices: array<u32>;
@group(0) @binding(5) var<storage, read> leafCount: array<u32>;

${LEAF_FUNCTIONS_WGSL}

fn getLeafBounds(leafIdx: u32) -> array<vec3<f32>, 2> {
    let srcIdx = sortedIds[leafIdx];
    let aabb = leafAABBs[srcIdx];
    return array<vec3<f32>, 2>(
        vec3<f32>(aabb.minX, aabb.minY, aabb.minZ),
        vec3<f32>(aabb.maxX, aabb.maxY, aabb.maxZ)
    );
}

fn getParent(nodeIdx: u32, isLeafNode: bool, n: u32) -> u32 {
    if (isLeafNode) {
        return parentIndices[nodeIdx];
    } else {
        return parentIndices[n + nodeIdx];
    }
}

fn nodeBase(idx: u32) -> u32 {
    return idx * 8u;
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

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = leafCount[0];
    let leafIdx = gid.x;

    if (leafIdx >= n) {
        return;
    }

    let bounds = getLeafBounds(leafIdx);
    writeBounds(n - 1u + leafIdx, bounds[0], bounds[1]);

    var current = leafIdx;
    var isLeafNode = true;

    for (var iter = 0u; iter < ${MAX_PROPAGATION_ITERS}u; iter++) {
        let parent = getParent(current, isLeafNode, n);

        let oldFlag = atomicAdd(&boundsFlags[parent], 1u);

        if (oldFlag == 0u) {
            return;
        }

        let left = readLeftChild(parent);
        let right = readRightChild(parent);

        var leftMin: vec3<f32>;
        var leftMax: vec3<f32>;
        var rightMin: vec3<f32>;
        var rightMax: vec3<f32>;

        if (isLeaf(left)) {
            let leftBounds = getLeafBounds(leafIndex(left));
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        } else {
            let leftBounds = readChildBounds(left);
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        }

        if (isLeaf(right)) {
            let rightBounds = getLeafBounds(leafIndex(right));
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        } else {
            let rightBounds = readChildBounds(right);
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        }

        let newMin = min(leftMin, rightMin);
        let newMax = max(leftMax, rightMax);

        writeBounds(parent, newMin, newMax);

        current = parent;
        isLeafNode = false;

        if (parent == 0u) {
            break;
        }
    }
}
`;

function createLBVHBuffers(device: GPUDevice, maxLeaves: number, label: string) {
    const Storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const treeNodes = device.createBuffer({
        label: `${label}-treeNodes`,
        size: 2 * maxLeaves * TREE_NODE_SIZE,
        usage: Storage,
    });
    const mortonCodes = device.createBuffer({
        label: `${label}-mortonCodes`,
        size: maxLeaves * 4,
        usage: Storage,
    });
    const sortedIds = device.createBuffer({
        label: `${label}-sortedIds`,
        size: maxLeaves * 4,
        usage: Storage,
    });
    const sceneBounds = device.createBuffer({
        label: `${label}-sceneBounds`,
        size: 32,
        usage: Storage,
    });
    const parentIndices = device.createBuffer({
        label: `${label}-parentIndices`,
        size: 2 * maxLeaves * 4,
        usage: Storage,
    });
    const boundsFlags = device.createBuffer({
        label: `${label}-boundsFlags`,
        size: maxLeaves * 4,
        usage: Storage,
    });
    const mortonParamsBuffer = device.createBuffer({
        label: `${label}-mortonParams`,
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(mortonParamsBuffer, 0, new Uint32Array([maxLeaves]));

    return {
        treeNodes,
        mortonCodes,
        sortedIds,
        sceneBounds,
        parentIndices,
        boundsFlags,
        mortonParamsBuffer,
    };
}

function destroyLBVHBuffers(lbvh: LBVH): void {
    lbvh.treeNodes.destroy();
    lbvh.sortedIds.destroy();
    lbvh.mortonCodes.destroy();
    lbvh.sceneBounds.destroy();
    lbvh.parentIndices.destroy();
    lbvh.boundsFlags.destroy();
    lbvh.mortonParamsBuffer.destroy();
}

function buildLBVHBindGroups(
    device: GPUDevice,
    pipelines: LBVH["pipelines"],
    leafAABBs: GPUBuffer,
    countBuffer: GPUBuffer,
    bufs: ReturnType<typeof createLBVHBuffers>,
): LBVH["bindGroups"] {
    return {
        bounds: device.createBindGroup({
            layout: pipelines.bounds.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: bufs.sceneBounds } },
                { binding: 2, resource: { buffer: countBuffer } },
            ],
        }),
        morton: device.createBindGroup({
            layout: pipelines.morton.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: bufs.sceneBounds } },
                { binding: 2, resource: { buffer: bufs.mortonCodes } },
                { binding: 3, resource: { buffer: bufs.sortedIds } },
                { binding: 4, resource: { buffer: countBuffer } },
                { binding: 5, resource: { buffer: bufs.mortonParamsBuffer } },
            ],
        }),
        tree: device.createBindGroup({
            layout: pipelines.tree.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.mortonCodes } },
                { binding: 1, resource: { buffer: bufs.treeNodes } },
                { binding: 2, resource: { buffer: bufs.parentIndices } },
                { binding: 3, resource: { buffer: countBuffer } },
            ],
        }),
        propagate: device.createBindGroup({
            layout: pipelines.propagate.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: bufs.sortedIds } },
                { binding: 2, resource: { buffer: bufs.treeNodes } },
                { binding: 3, resource: { buffer: bufs.boundsFlags } },
                { binding: 4, resource: { buffer: bufs.parentIndices } },
                { binding: 5, resource: { buffer: countBuffer } },
            ],
        }),
    };
}

export async function createLBVH(device: GPUDevice, config: LBVHConfig): Promise<LBVH> {
    const { leafAABBs, countBuffer, maxLeaves, label } = config;

    const bufs = createLBVHBuffers(device, maxLeaves, label);

    const [boundsModule, mortonModule, treeModule, propagateModule] = await Promise.all([
        device.createShaderModule({ code: boundsShader }),
        device.createShaderModule({ code: mortonShader }),
        device.createShaderModule({ code: treeShader }),
        device.createShaderModule({ code: propagateShader }),
    ]);

    const [boundsPl, mortonPl, treePl, propagatePl, radixSort] = await Promise.all([
        device.createComputePipelineAsync({
            label: "bvh-bounds",
            layout: "auto",
            compute: { module: boundsModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            label: "bvh-morton",
            layout: "auto",
            compute: { module: mortonModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            label: "bvh-tree",
            layout: "auto",
            compute: { module: treeModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            label: "bvh-propagate",
            layout: "auto",
            compute: { module: propagateModule, entryPoint: "main" },
        }),
        createRadixSort(device, {
            keys: bufs.mortonCodes,
            values: bufs.sortedIds,
            count: maxLeaves,
        }),
    ]);

    const pipelines = { bounds: boundsPl, morton: mortonPl, tree: treePl, propagate: propagatePl };
    const bindGroups = buildLBVHBindGroups(device, pipelines, leafAABBs, countBuffer, bufs);

    return {
        ...bufs,
        radixSort,
        config,
        pipelines,
        bindGroups,
    };
}

export function rebuildLBVHBuffers(
    lbvh: LBVH,
    device: GPUDevice,
    newLeafAABBs: GPUBuffer,
    newMaxLeaves: number,
): void {
    destroyLBVHBuffers(lbvh);

    const { countBuffer, label } = lbvh.config;
    lbvh.config = { leafAABBs: newLeafAABBs, countBuffer, maxLeaves: newMaxLeaves, label };

    const bufs = createLBVHBuffers(device, newMaxLeaves, label);
    Object.assign(lbvh, bufs);

    rebuildRadixSort(lbvh.radixSort, bufs.mortonCodes, bufs.sortedIds, newMaxLeaves);
    lbvh.bindGroups = buildLBVHBindGroups(device, lbvh.pipelines, newLeafAABBs, countBuffer, bufs);
}

export function dispatchLBVH(
    lbvh: LBVH,
    encoder: GPUCommandEncoder,
    device: GPUDevice,
    cpuCount: number,
    ts?: (name: string) => GPUComputePassTimestampWrites | undefined,
): void {
    device.queue.writeBuffer(lbvh.sceneBounds, 0, initBounds);
    encoder.clearBuffer(lbvh.parentIndices);
    encoder.clearBuffer(lbvh.boundsFlags);

    const boundsWG = Math.ceil(cpuCount / WORKGROUP_SIZE);
    const mortonWG = Math.ceil(lbvh.config.maxLeaves / WORKGROUP_SIZE);
    const treeWG = Math.ceil(Math.max(cpuCount - 1, 1) / WORKGROUP_SIZE);
    const propagateWG = Math.ceil(cpuCount / WORKGROUP_SIZE);

    let pass = beginComputePass(encoder, ts?.("lbvh:bounds"));
    pass.setPipeline(lbvh.pipelines.bounds);
    pass.setBindGroup(0, lbvh.bindGroups.bounds);
    pass.dispatchWorkgroups(boundsWG);
    pass.end();

    pass = beginComputePass(encoder, ts?.("lbvh:morton"));
    pass.setPipeline(lbvh.pipelines.morton);
    pass.setBindGroup(0, lbvh.bindGroups.morton);
    pass.dispatchWorkgroups(mortonWG);
    pass.end();

    pass = beginComputePass(encoder, ts?.("lbvh:sort"));
    dispatchRadixSort(lbvh.radixSort, pass);
    pass.end();

    pass = beginComputePass(encoder, ts?.("lbvh:tree"));
    pass.setPipeline(lbvh.pipelines.tree);
    pass.setBindGroup(0, lbvh.bindGroups.tree);
    pass.dispatchWorkgroups(treeWG);
    pass.end();

    pass = beginComputePass(encoder, ts?.("lbvh:propagate"));
    pass.setPipeline(lbvh.pipelines.propagate);
    pass.setBindGroup(0, lbvh.bindGroups.propagate);
    pass.dispatchWorkgroups(propagateWG);
    pass.end();
}

export function disposeLBVH(lbvh: LBVH): void {
    destroyLBVHBuffers(lbvh);
    disposeRadixSort(lbvh.radixSort);
}
