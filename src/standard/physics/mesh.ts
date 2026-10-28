import type { PhysicsMesh } from "../../core/physics";
import type { Resource, World } from "../../engine";
import type { AABB } from "./common/records";
import { buildMesh } from "./geometry-builds";
import { kernel } from "./kernel/kernel";

/** A b3MeshNode's mesh-local bounds in meters and its position in the tree. */
export interface PhysicsMeshNode {
    readonly bounds: AABB;
    /** Root is level zero. */
    readonly level: number;
    readonly leaf: boolean;
}

/** A snapshot of Box3D's b3MeshData reports and mesh-local tree. */
export interface PhysicsMeshData {
    readonly triangleCount: number;
    readonly vertexCount: number;
    readonly degenerateCount: number;
    /** Sum of the retained triangles' areas in square meters, not the tree nodes' surface area. */
    readonly surfaceArea: number;
    readonly treeHeight: number;
    /** Nodes in native storage order, with the root first. */
    readonly nodes: readonly PhysicsMeshNode[];
    /** Mesh-local bounds in meters, root at level zero and siblings in left-to-right order. */
    readonly nodeBounds: readonly (readonly AABB[])[];
}

const builds: Resource<Map<PhysicsMeshData, number>> = {
    create(world) {
        const live = new Map<PhysicsMeshData, number>();
        const k = kernel(world);
        world.onDispose(() => {
            for (const pointer of live.values()) k.nonconvexDestroy(pointer);
            live.clear();
        });
        return live;
    },
};

/**
 * Box3D's b3CreateMesh: build independently of PhysicsMeshes and its shape cache.
 * Each call owns a new kernel allocation until destroyPhysicsMesh or World disposal.
 * Reports are snapshots and remain readable after release. Invalid definitions throw as
 * they do when a Shape builds the same definition; this call does not register a shape.
 */
export function createPhysicsMesh(world: World, definition: PhysicsMesh): PhysicsMeshData {
    const live = world.resource(builds);
    const mesh = buildMesh(world, definition);
    const nodeBounds: AABB[][] = [];
    const nodes: PhysicsMeshNode[] = [];
    const visit = (index: number, level: number): void => {
        const node = mesh.nodes[index]!;
        const bounds = { lowerBound: node.lowerBound, upperBound: node.upperBound };
        nodes[index] = { bounds, level, leaf: node.leaf };
        (nodeBounds[level] ??= []).push(bounds);
        if (!node.leaf) {
            visit(index + 1, level + 1);
            visit(index + node.childOffset, level + 1);
        }
    };
    if (mesh.nodes.length) visit(0, 0);
    const report: PhysicsMeshData = {
        triangleCount: mesh.triangles.length,
        vertexCount: mesh.vertices.length,
        degenerateCount: mesh.degenerateCount,
        surfaceArea: mesh.surfaceArea,
        treeHeight: mesh.treeHeight,
        nodes,
        nodeBounds,
    };
    live.set(report, kernel(world).nonconvexRetain());
    return report;
}

/** Box3D's b3DestroyMesh. Refuses a foreign or already released build. */
export function destroyPhysicsMesh(world: World, mesh: PhysicsMeshData): void {
    const live = world.resource(builds);
    const pointer = live.get(mesh);
    if (pointer === undefined)
        throw new Error("destroyPhysicsMesh: build is not live in this World");
    kernel(world).nonconvexDestroy(pointer);
    live.delete(mesh);
}
