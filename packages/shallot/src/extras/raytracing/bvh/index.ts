import { resource, capacity, type State } from "../../../engine";
import { SHAPE_AABB_STRIDE } from "./structs";
import { Compute } from "../../../standard/compute";
import type { ComputeNode } from "../../../standard/compute";
import { gbuf, type GBuf } from "../../../standard/compute";
import { WorldTransform } from "../../../standard/transforms";
import { Part, Dynamic, Render, meshRegistry } from "../../../standard/render";
import { getMeshId, type SurfaceData } from "../../../standard/render/core";
import { surfaceRegistry } from "../../../standard/render";
import { createBLASAtlas, type BLASAtlas } from "./blas";
import { createTLAS, createTLASNode, type TLAS } from "./tlas";
import { createBLASRefitNode, type DynamicShapeInfo } from "./refit";
import { createInstanceNode } from "./instance";

export {
    type BLASAtlas,
    type BLASMeta,
    createBLASAtlas,
    extractShapeTriangles,
    buildShapeBLAS,
} from "./blas";
export { type TLAS, type TLASConfig, createTLAS, createTLASNode, refitTLAS } from "./tlas";
export { createBLASRefitNode } from "./refit";
export { createInstanceNode } from "./instance";
export { createRadixSortNode, dispatchRadixSort } from "../../../standard/radix";
export {
    BVH_UTILS_WGSL,
    BVH_STRUCTS,
    TLAS_BLAS_STRUCTS,
    TLAS_BLAS_BINDINGS,
    TLAS_BLAS_TRAVERSAL,
    TLAS_BLAS_SHADOW,
    BLAS_SHADOW_WGSL,
    ANALYTIC_SHADOW_WGSL,
    ANALYTIC_INTERSECTION_WGSL,
} from "./traverse";
export {
    LEAF_FLAG,
    isLeaf,
    leafIndex,
    TREE_NODE_SIZE,
    BVH_NODE_SIZE,
    BLAS_TRIANGLE_SIZE,
    TREE_NODE_STRIDE,
    BLAS_META_STRIDE,
    SHAPE_AABB_STRIDE,
    RAY_EPSILON,
    SAFE_INVERSE_EPSILON,
    AABB_SENTINEL,
    MORTON_QUANTIZATION,
    MAX_PROPAGATION_ITERS,
    OCT_ENCODING_SCALE,
    INVALID_NODE,
    LEAF_FLAG_WGSL,
    TREE_NODE_STRUCT_WGSL,
    BVH_NODE_STRUCT_WGSL,
    BLAS_NODE_STRUCT_WGSL,
    BLAS_TRIANGLE_STRUCT_WGSL,
    RAY_STRUCT_WGSL,
    HIT_RESULT_STRUCT_WGSL,
    OCT_DECODE_WGSL,
    AABB_SENTINEL_WGSL,
} from "./structs";

export interface BVHState {
    instanceAABBs: GBuf;
    instanceInverses: GBuf;
    instanceCount: GPUBuffer;
    tlas: TLAS;
    blasAtlas: BLASAtlas;
    blasVersion: number;
    dynamicInstanceVersion: number;
    activeChecks: (() => boolean)[];
}

export function isBVHActive(bvh: BVHState): boolean {
    for (const check of bvh.activeChecks) {
        if (check()) return true;
    }
    return false;
}

export const BVH = resource<BVHState>("bvh");

export async function initializeBVH(state: State): Promise<void> {
    if (BVH.from(state)) return;

    const compute = Compute.from(state);
    const render = Render.from(state);
    if (!compute || !render) return;

    const { device } = compute;

    const blasAtlas = createBLASAtlas(device, (id) => meshRegistry.get(id));

    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const instanceAABBs = gbuf(
        device,
        "instanceAABBs",
        storageUsage,
        (cap) => cap * SHAPE_AABB_STRIDE * 4,
    );
    const instanceCount = device.createBuffer({
        label: "instanceCount",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const tlas = await createTLAS(device, {
        instanceAABBs,
        instanceCount,
        getEntityCount: () => render.entityCount,
    });

    const bvhState: BVHState = {
        instanceAABBs,
        instanceInverses: gbuf(device, "instanceInverses", storageUsage, (cap) => cap * 64),
        instanceCount,
        tlas,
        blasAtlas,
        blasVersion: meshRegistry.version,
        dynamicInstanceVersion: 0,
        activeChecks: [],
    };

    state.setResource(BVH, bvhState);

    let entityIds = new Uint32Array(capacity());
    let cachedUploadCapacity = capacity();
    const countBuf = new Uint32Array(1);

    const dynamicShapes = new Map<number, DynamicShapeInfo>();
    const infoPool: DynamicShapeInfo[] = [];
    let poolUsed = 0;
    let dynHash = 0;

    const dynamicSurface: SurfaceData = {};

    const rebuildDynamicShapes = () => {
        dynamicShapes.clear();
        poolUsed = 0;
        let hash = 0;
        const surfaces = surfaceRegistry.all();
        for (const eid of state.query([Part])) {
            const shapeId = getMeshId(eid);
            const surfaceType = Part.surface[eid];
            const surfaceData = surfaces[surfaceType];
            const hasDynamic = state.hasComponent(eid, Dynamic);
            const hasVertexCode = !!surfaceData?.vertex;
            if (!hasDynamic && !hasVertexCode) continue;
            const surface = hasVertexCode ? surfaceData! : dynamicSurface;
            const existing = dynamicShapes.get(shapeId);
            if (existing) {
                existing.entities.push(eid);
            } else {
                let info = infoPool[poolUsed];
                if (info) {
                    info.surface = surface;
                    info.entities.length = 0;
                    info.entities.push(eid);
                } else {
                    info = { surface, entities: [eid] };
                    infoPool[poolUsed] = info;
                }
                poolUsed++;
                dynamicShapes.set(shapeId, info);
            }
            hash = (hash * 31 + shapeId) | 0;
            hash = (hash * 31 + eid) | 0;
        }
        dynHash = hash;
    };

    const uploadNode: ComputeNode = {
        name: "bvh-upload",
        scope: "frame",
        inputs: [],
        outputs: ["instance-count"],
        execute(ctx) {
            rebuildDynamicShapes();
            if (!isBVHActive(bvhState)) return;

            if (capacity() !== cachedUploadCapacity) {
                cachedUploadCapacity = capacity();
                entityIds = new Uint32Array(cachedUploadCapacity);
                bvhState.blasVersion = -1;
            }

            const currentMeshVersion = meshRegistry.version;

            const needsRebuild =
                currentMeshVersion !== bvhState.blasVersion ||
                dynHash !== bvhState.dynamicInstanceVersion;

            if (needsRebuild) {
                bvhState.blasAtlas.triIdsBuffer.destroy();
                bvhState.blasAtlas.trianglesBuffer.destroy();
                bvhState.blasAtlas.nodesBuffer.destroy();
                bvhState.blasAtlas.parentIndicesBuffer.destroy();
                bvhState.blasAtlas.baseTrianglesBuffer.destroy();
                bvhState.blasAtlas.boundsFlagsBuffer.destroy();
                bvhState.blasAtlas.entityBlasMetaBuffer.destroy();
                bvhState.blasAtlas.perEntityShapeAABBs.destroy();
                bvhState.blasAtlas.shapeDataBuffer.destroy();

                const dynMap = new Map<number, number[]>();
                for (const [shapeId, info] of dynamicShapes) {
                    dynMap.set(shapeId, info.entities);
                }
                bvhState.blasAtlas = createBLASAtlas(
                    ctx.device,
                    (id) => meshRegistry.get(id),
                    dynMap,
                );
                bvhState.blasVersion = currentMeshVersion;
                bvhState.dynamicInstanceVersion = dynHash;
            }

            const meshEntities = state.query([Part, WorldTransform]);
            let entityCount = 0;

            for (const eid of meshEntities) {
                entityIds[entityCount++] = eid;
            }

            ctx.device.queue.writeBuffer(
                bvhState.tlas.entityIds.buffer,
                0,
                entityIds,
                0,
                Math.max(entityCount, 1),
            );
            countBuf[0] = entityCount;
            ctx.device.queue.writeBuffer(bvhState.instanceCount, 0, countBuf);
        },
    };
    compute.graph.add(uploadNode);

    const isActive = () => isBVHActive(bvhState);

    const blasRefitNode = createBLASRefitNode(bvhState, render, () => dynamicShapes, isActive);
    compute.graph.add(blasRefitNode);

    const instanceNode = createInstanceNode(
        bvhState,
        render,
        ["matrices", "blas-nodes", "batched"],
        isActive,
    );
    compute.graph.add(instanceNode);

    const tlasNode = createTLASNode(tlas, isActive);
    compute.graph.add(tlasNode);
}
