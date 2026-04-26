import { capacity } from "../../../engine";
import {
    beginComputePass,
    type ComputeNode,
    type ExecutionContext,
} from "../../../standard/compute";
import {
    createLBVH,
    rebuildLBVHBuffers,
    dispatchLBVH,
    INSTANCE_AABB_STRUCT_WGSL,
    type LBVH,
} from "../../../standard/bvh";
import {
    TREE_NODE_STRUCT_WGSL,
    BVH_NODE_STRUCT_WGSL,
    LEAF_FLAG_WGSL,
    BVH_NODE_SIZE,
    AABB_SENTINEL_WGSL,
} from "./structs";
import { gbuf, type GBuf } from "../../../standard/compute";

const WORKGROUP_SIZE = 256;
const MAX_TREE_DEPTH = 32;

export interface TLASConfig {
    instanceAABBs: GBuf;
    instanceCount: GPUBuffer;
    getEntityCount: () => number;
}

export interface TLAS {
    lbvh: LBVH;
    compactAABBs: GBuf;
    bvhNodes: GBuf;
    entityIds: GBuf;
    pipelines: { compact: GPUComputePipeline; collapse: GPUComputePipeline };
    compactLayout: GPUBindGroupLayout;
    collapseLayout: GPUBindGroupLayout;
    bindGroups: { compact: GPUBindGroup | null; collapse: GPUBindGroup | null };
    cachedCapacity: number;
    config: TLASConfig;
}

const compactShader = /* wgsl */ `
${INSTANCE_AABB_STRUCT_WGSL}

@group(0) @binding(0) var<storage, read> instanceAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> compactAABBs: array<InstanceAABB>;
@group(0) @binding(3) var<storage, read> instanceCount: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let count = instanceCount[0];
    if (tid >= count) { return; }
    compactAABBs[tid] = instanceAABBs[entityIds[tid]];
}
`;

const collapseShader = /* wgsl */ `
${TREE_NODE_STRUCT_WGSL}
${BVH_NODE_STRUCT_WGSL}
${LEAF_FLAG_WGSL}
${AABB_SENTINEL_WGSL}

const INVALID_NODE: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read> treeNodes: array<TreeNode>;
@group(0) @binding(1) var<storage, read> instanceCount: array<u32>;
@group(0) @binding(2) var<storage, read> parentIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> bvhNodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read> sortedIds: array<u32>;
@group(0) @binding(5) var<storage, read> entityIds: array<u32>;

fn isLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn leafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}

fn resolveChild(child: u32) -> u32 {
    if (isLeaf(child)) {
        return entityIds[sortedIds[leafIndex(child)]] | LEAF_FLAG;
    }
    return child;
}

fn getDepth(nodeIdx: u32, n: u32) -> u32 {
    var depth = 0u;
    var current = nodeIdx;
    for (var iter = 0u; iter < ${MAX_TREE_DEPTH}u; iter++) {
        if (current == 0u) { break; }
        current = parentIndices[n + current];
        depth++;
    }
    return depth;
}

fn getChildBounds(child: u32, n: u32) -> array<vec3<f32>, 2> {
    if (isLeaf(child)) {
        let leafNodeIdx = n - 1u + leafIndex(child);
        let node = treeNodes[leafNodeIdx];
        return array<vec3<f32>, 2>(
            vec3<f32>(node.minX, node.minY, node.minZ),
            vec3<f32>(node.maxX, node.maxY, node.maxZ)
        );
    } else {
        let node = treeNodes[child];
        return array<vec3<f32>, 2>(
            vec3<f32>(node.minX, node.minY, node.minZ),
            vec3<f32>(node.maxX, node.maxY, node.maxZ)
        );
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = instanceCount[0];
    let nodeIdx = gid.x;

    if (n == 1u) {
        if (nodeIdx == 0u) {
            var out: BVHNode;
            out.child0 = resolveChild(0u | LEAF_FLAG);
            out.child1 = INVALID_NODE;
            out.child2 = INVALID_NODE;
            out.child3 = INVALID_NODE;

            let bounds = getChildBounds(0u | LEAF_FLAG, n);
            out.c0_minX = bounds[0].x; out.c0_minY = bounds[0].y; out.c0_minZ = bounds[0].z;
            out.c0_maxX = bounds[1].x; out.c0_maxY = bounds[1].y; out.c0_maxZ = bounds[1].z;
            out.c1_minX = AABB_SENTINEL; out.c1_minY = AABB_SENTINEL; out.c1_minZ = AABB_SENTINEL;
            out.c1_maxX = -AABB_SENTINEL; out.c1_maxY = -AABB_SENTINEL; out.c1_maxZ = -AABB_SENTINEL;
            out.c2_minX = AABB_SENTINEL; out.c2_minY = AABB_SENTINEL; out.c2_minZ = AABB_SENTINEL;
            out.c2_maxX = -AABB_SENTINEL; out.c2_maxY = -AABB_SENTINEL; out.c2_maxZ = -AABB_SENTINEL;
            out.c3_minX = AABB_SENTINEL; out.c3_minY = AABB_SENTINEL; out.c3_minZ = AABB_SENTINEL;
            out.c3_maxX = -AABB_SENTINEL; out.c3_maxY = -AABB_SENTINEL; out.c3_maxZ = -AABB_SENTINEL;

            bvhNodes[0] = out;
        }
        return;
    }

    if (nodeIdx >= n - 1u) {
        return;
    }

    let depth = getDepth(nodeIdx, n);
    let node = treeNodes[nodeIdx];
    let left = node.leftChild;
    let right = node.rightChild;

    var out: BVHNode;

    out.child0 = INVALID_NODE;
    out.child1 = INVALID_NODE;
    out.child2 = INVALID_NODE;
    out.child3 = INVALID_NODE;
    out.c0_minX = AABB_SENTINEL; out.c0_minY = AABB_SENTINEL; out.c0_minZ = AABB_SENTINEL;
    out.c0_maxX = -AABB_SENTINEL; out.c0_maxY = -AABB_SENTINEL; out.c0_maxZ = -AABB_SENTINEL;
    out.c1_minX = AABB_SENTINEL; out.c1_minY = AABB_SENTINEL; out.c1_minZ = AABB_SENTINEL;
    out.c1_maxX = -AABB_SENTINEL; out.c1_maxY = -AABB_SENTINEL; out.c1_maxZ = -AABB_SENTINEL;
    out.c2_minX = AABB_SENTINEL; out.c2_minY = AABB_SENTINEL; out.c2_minZ = AABB_SENTINEL;
    out.c2_maxX = -AABB_SENTINEL; out.c2_maxY = -AABB_SENTINEL; out.c2_maxZ = -AABB_SENTINEL;
    out.c3_minX = AABB_SENTINEL; out.c3_minY = AABB_SENTINEL; out.c3_minZ = AABB_SENTINEL;
    out.c3_maxX = -AABB_SENTINEL; out.c3_maxY = -AABB_SENTINEL; out.c3_maxZ = -AABB_SENTINEL;

    if ((depth & 1u) != 0u) {
        out.child0 = resolveChild(left);
        let bounds0 = getChildBounds(left, n);
        out.c0_minX = bounds0[0].x; out.c0_minY = bounds0[0].y; out.c0_minZ = bounds0[0].z;
        out.c0_maxX = bounds0[1].x; out.c0_maxY = bounds0[1].y; out.c0_maxZ = bounds0[1].z;

        out.child1 = resolveChild(right);
        let bounds1 = getChildBounds(right, n);
        out.c1_minX = bounds1[0].x; out.c1_minY = bounds1[0].y; out.c1_minZ = bounds1[0].z;
        out.c1_maxX = bounds1[1].x; out.c1_maxY = bounds1[1].y; out.c1_maxZ = bounds1[1].z;

        bvhNodes[nodeIdx] = out;
        return;
    }

    if (isLeaf(left)) {
        out.child0 = resolveChild(left);
        let bounds = getChildBounds(left, n);
        out.c0_minX = bounds[0].x; out.c0_minY = bounds[0].y; out.c0_minZ = bounds[0].z;
        out.c0_maxX = bounds[1].x; out.c0_maxY = bounds[1].y; out.c0_maxZ = bounds[1].z;
    } else {
        let leftNode = treeNodes[left];
        let ll = leftNode.leftChild;
        let lr = leftNode.rightChild;

        out.child0 = resolveChild(ll);
        let bounds0 = getChildBounds(ll, n);
        out.c0_minX = bounds0[0].x; out.c0_minY = bounds0[0].y; out.c0_minZ = bounds0[0].z;
        out.c0_maxX = bounds0[1].x; out.c0_maxY = bounds0[1].y; out.c0_maxZ = bounds0[1].z;

        out.child1 = resolveChild(lr);
        let bounds1 = getChildBounds(lr, n);
        out.c1_minX = bounds1[0].x; out.c1_minY = bounds1[0].y; out.c1_minZ = bounds1[0].z;
        out.c1_maxX = bounds1[1].x; out.c1_maxY = bounds1[1].y; out.c1_maxZ = bounds1[1].z;
    }

    if (isLeaf(right)) {
        out.child2 = resolveChild(right);
        let bounds = getChildBounds(right, n);
        out.c2_minX = bounds[0].x; out.c2_minY = bounds[0].y; out.c2_minZ = bounds[0].z;
        out.c2_maxX = bounds[1].x; out.c2_maxY = bounds[1].y; out.c2_maxZ = bounds[1].z;
    } else {
        let rightNode = treeNodes[right];
        let rl = rightNode.leftChild;
        let rr = rightNode.rightChild;

        out.child2 = resolveChild(rl);
        let bounds2 = getChildBounds(rl, n);
        out.c2_minX = bounds2[0].x; out.c2_minY = bounds2[0].y; out.c2_minZ = bounds2[0].z;
        out.c2_maxX = bounds2[1].x; out.c2_maxY = bounds2[1].y; out.c2_maxZ = bounds2[1].z;

        out.child3 = resolveChild(rr);
        let bounds3 = getChildBounds(rr, n);
        out.c3_minX = bounds3[0].x; out.c3_minY = bounds3[0].y; out.c3_minZ = bounds3[0].z;
        out.c3_maxX = bounds3[1].x; out.c3_maxY = bounds3[1].y; out.c3_maxZ = bounds3[1].z;
    }

    bvhNodes[nodeIdx] = out;
}
`;

function rebuildTLASBindGroups(tlas: TLAS, device: GPUDevice): void {
    tlas.bindGroups.compact = device.createBindGroup({
        layout: tlas.compactLayout,
        entries: [
            { binding: 0, resource: { buffer: tlas.config.instanceAABBs.buffer } },
            { binding: 1, resource: { buffer: tlas.entityIds.buffer } },
            { binding: 2, resource: { buffer: tlas.compactAABBs.buffer } },
            { binding: 3, resource: { buffer: tlas.config.instanceCount } },
        ],
    });

    tlas.bindGroups.collapse = device.createBindGroup({
        layout: tlas.collapseLayout,
        entries: [
            { binding: 0, resource: { buffer: tlas.lbvh.treeNodes } },
            { binding: 1, resource: { buffer: tlas.config.instanceCount } },
            { binding: 2, resource: { buffer: tlas.lbvh.parentIndices } },
            { binding: 3, resource: { buffer: tlas.bvhNodes.buffer } },
            { binding: 4, resource: { buffer: tlas.lbvh.sortedIds } },
            { binding: 5, resource: { buffer: tlas.entityIds.buffer } },
        ],
    });
}

function ensureTLASCapacity(tlas: TLAS, device: GPUDevice): void {
    const cap = capacity();
    if (cap === tlas.cachedCapacity) return;
    tlas.cachedCapacity = cap;

    rebuildLBVHBuffers(tlas.lbvh, device, tlas.compactAABBs.buffer, cap);
    rebuildTLASBindGroups(tlas, device);
}

export async function createTLAS(device: GPUDevice, config: TLASConfig): Promise<TLAS> {
    const compactAABBs = gbuf(
        device,
        "tlas-compact-aabbs",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        (cap) => cap * 32,
    );
    const bvhNodes = gbuf(
        device,
        "tlas-bvh-nodes",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        (cap) => cap * BVH_NODE_SIZE,
    );
    const entityIds = gbuf(
        device,
        "tlas-entity-ids",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        (cap) => cap * 4,
    );

    const lbvh = await createLBVH(device, {
        leafAABBs: compactAABBs.buffer,
        countBuffer: config.instanceCount,
        maxLeaves: capacity(),
        label: "tlas",
    });

    const [compactModule, collapseModule] = await Promise.all([
        device.createShaderModule({ code: compactShader }),
        device.createShaderModule({ code: collapseShader }),
    ]);

    const [compactPl, collapsePl] = await Promise.all([
        device.createComputePipelineAsync({
            label: "tlas-compact",
            layout: "auto",
            compute: { module: compactModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            label: "tlas-collapse",
            layout: "auto",
            compute: { module: collapseModule, entryPoint: "main" },
        }),
    ]);

    const tlas: TLAS = {
        lbvh,
        compactAABBs,
        bvhNodes,
        entityIds,
        pipelines: { compact: compactPl, collapse: collapsePl },
        compactLayout: compactPl.getBindGroupLayout(0),
        collapseLayout: collapsePl.getBindGroupLayout(0),
        bindGroups: { compact: null, collapse: null },
        cachedCapacity: capacity(),
        config,
    };

    rebuildTLASBindGroups(tlas, device);

    return tlas;
}

export function refitTLAS(
    tlas: TLAS,
    encoder: GPUCommandEncoder,
    device: GPUDevice,
    entityCount: number,
    ts?: (name: string) => GPURenderPassTimestampWrites | undefined,
): void {
    ensureTLASCapacity(tlas, device);

    encoder.clearBuffer(tlas.lbvh.boundsFlags);

    const propagateWG = Math.ceil(entityCount / WORKGROUP_SIZE);
    const propagatePass = beginComputePass(encoder, ts?.("tlas-propagate"));
    propagatePass.setPipeline(tlas.lbvh.pipelines.propagate);
    propagatePass.setBindGroup(0, tlas.lbvh.bindGroups.propagate);
    propagatePass.dispatchWorkgroups(propagateWG);
    propagatePass.end();

    const collapsePass = beginComputePass(encoder, ts?.("tlas-collapse"));
    collapsePass.setPipeline(tlas.pipelines.collapse);
    collapsePass.setBindGroup(0, tlas.bindGroups.collapse!);
    collapsePass.dispatchWorkgroups(Math.ceil(Math.max(entityCount - 1, 1) / WORKGROUP_SIZE));
    collapsePass.end();
}

function executeTLAS(tlas: TLAS, ctx: ExecutionContext): void {
    const { device, encoder } = ctx;
    const ts = ctx.timestampWrites;
    const entityCount = tlas.config.getEntityCount();

    ensureTLASCapacity(tlas, device);

    const workgroups = Math.ceil(entityCount / WORKGROUP_SIZE);

    const compactPass = beginComputePass(encoder, ts?.("tlas-compact"));
    compactPass.setPipeline(tlas.pipelines.compact);
    compactPass.setBindGroup(0, tlas.bindGroups.compact!);
    compactPass.dispatchWorkgroups(workgroups);
    compactPass.end();

    dispatchLBVH(
        tlas.lbvh,
        encoder,
        device,
        entityCount,
        (name) => ts?.(name) as GPUComputePassTimestampWrites | undefined,
    );

    const collapsePass = beginComputePass(encoder, ts?.("tlas-collapse"));
    collapsePass.setPipeline(tlas.pipelines.collapse);
    collapsePass.setBindGroup(0, tlas.bindGroups.collapse!);
    collapsePass.dispatchWorkgroups(Math.ceil(Math.max(entityCount - 1, 1) / WORKGROUP_SIZE));
    collapsePass.end();
}

export function createTLASNode(tlas: TLAS, isActive?: () => boolean): ComputeNode {
    return {
        name: "tlas",
        scope: "frame",
        sync: true,
        inputs: ["instance-aabbs", "instance-count"],
        outputs: ["tlas-bvh-nodes", "tlas-morton-codes", "tlas-instance-ids"],

        execute(ctx: ExecutionContext) {
            if (isActive?.() === false) return;
            executeTLAS(tlas, ctx);
        },
    };
}
