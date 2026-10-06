import type { Transform, Vec3 } from "../common/math";
import { ShapeType } from "../common/types";
import { kernel } from "../kernel/kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "../kernel/shapecolumns";
import type { WorldState } from "../world/world";
import type { Capsule, Sphere } from "./geometry";
import type { HeightFieldData } from "./heightfield";
import { type HullData, readHullAt } from "./hull";
import type { Mesh, MeshData, MeshNode, MeshTriangle } from "./mesh";

export type CompoundDrawChild = {
    type:
        | typeof ShapeType.Sphere
        | typeof ShapeType.Capsule
        | typeof ShapeType.Hull
        | typeof ShapeType.Mesh;
    transform: Transform;
    geometry: Sphere | Capsule | HullData | Mesh;
};

function vector(v: DataView, offset: number): Vec3 {
    return {
        x: v.getFloat32(offset, true),
        y: v.getFloat32(offset + 4, true),
        z: v.getFloat32(offset + 8, true),
    };
}
function readMeshData(world: WorldState, pointer: number): MeshData {
    const memory = kernel(world.ecsState).memory.buffer;
    const header = new DataView(memory, pointer, 96);
    const byteCount = header.getUint32(16, true);
    const v = new DataView(memory, pointer, byteCount);
    const nodeOffset = v.getUint32(56, true),
        nodeCount = v.getUint32(60, true),
        vertexOffset = v.getUint32(64, true),
        vertexCount = v.getUint32(68, true),
        triangleOffset = v.getUint32(72, true),
        triangleCount = v.getUint32(76, true),
        materialOffset = v.getUint32(80, true),
        materialCount = v.getUint32(84, true),
        flagsOffset = v.getUint32(88, true);
    const nodes: MeshNode[] = [];
    for (let i = 0; i < nodeCount; ++i) {
        const o = nodeOffset + i * 32,
            data = v.getUint32(o + 12, true),
            leaf = (data & 3) === 3;
        nodes.push({
            lowerBound: vector(v, o),
            upperBound: vector(v, o + 16),
            leaf,
            axis: leaf ? 0 : data & 3,
            childOffset: leaf ? 0 : data >>> 2,
            triangleCount: leaf ? data >>> 2 : 0,
            triangleOffset: v.getUint32(o + 28, true),
        });
    }
    const vertices: Vec3[] = [];
    for (let i = 0; i < vertexCount; ++i) vertices.push(vector(v, vertexOffset + i * 12));
    const triangles: MeshTriangle[] = [];
    for (let i = 0; i < triangleCount; ++i) {
        const o = triangleOffset + i * 12;
        triangles.push({
            index1: v.getUint32(o, true),
            index2: v.getUint32(o + 4, true),
            index3: v.getUint32(o + 8, true),
        });
    }
    const materialIndices = new Array<number>(triangleCount),
        flags = new Array<number>(triangleCount);
    for (let i = 0; i < triangleCount; ++i) {
        materialIndices[i] = v.getUint8(materialOffset + i);
        flags[i] = v.getUint8(flagsOffset + i);
    }
    return {
        bounds: { lowerBound: vector(v, 20), upperBound: vector(v, 32) },
        surfaceArea: v.getFloat32(44, true),
        treeHeight: v.getUint32(48, true),
        degenerateCount: v.getUint32(52, true),
        nodes,
        vertices,
        triangles,
        materialIndices,
        materialCount,
        flags,
    };
}

export function readShapeMesh(world: WorldState, shape: number): Mesh {
    world.shapeStore.refreshViews();
    const o = shape * SHAPE_STRIDE + S_GEO_REFERENCE,
        pointer = world.shapeStore.shapeU[o],
        f = world.shapeStore.shapeF;
    return {
        data: readMeshData(world, pointer),
        scale: { x: f[o + 1], y: f[o + 2], z: f[o + 3] },
    };
}

export function readShapeHeightField(world: WorldState, shape: number): HeightFieldData {
    world.shapeStore.refreshViews();
    const pointer = world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    const header = new DataView(kernel(world.ecsState).memory.buffer, pointer, 96);
    const byteCount = header.getUint32(16, true);
    const v = new DataView(kernel(world.ecsState).memory.buffer, pointer, byteCount),
        columnCount = v.getUint32(68, true),
        rowCount = v.getUint32(72, true),
        heightsOffset = v.getUint32(76, true),
        materialOffset = v.getUint32(80, true),
        flagsOffset = v.getUint32(84, true),
        cellCount = (columnCount - 1) * (rowCount - 1);
    const compressedHeights = new Array<number>(columnCount * rowCount),
        materialIndices = new Array<number>(cellCount),
        flags = new Array<number>(cellCount * 2);
    for (let i = 0; i < compressedHeights.length; ++i)
        compressedHeights[i] = v.getUint16(heightsOffset + i * 2, true);
    for (let i = 0; i < cellCount; ++i) materialIndices[i] = v.getUint8(materialOffset + i);
    for (let i = 0; i < flags.length; ++i) flags[i] = v.getUint8(flagsOffset + i);
    return {
        aabb: { lowerBound: vector(v, 20), upperBound: vector(v, 32) },
        minHeight: v.getFloat32(44, true),
        maxHeight: v.getFloat32(48, true),
        heightScale: v.getFloat32(52, true),
        scale: vector(v, 56),
        columnCount,
        rowCount,
        compressedHeights,
        materialIndices,
        flags,
        clockwise: v.getUint8(88) !== 0,
    };
}

function readTransform(v: DataView, o: number): Transform {
    return {
        p: vector(v, o),
        q: {
            v: vector(v, o + 12),
            s: v.getFloat32(o + 24, true),
        },
    };
}

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

export function readCompoundChildren(world: WorldState, shape: number): CompoundDrawChild[] {
    world.shapeStore.refreshViews();
    const pointer = world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    const header = new DataView(kernel(world.ecsState).memory.buffer, pointer, 128);
    const byteCount = header.getUint32(8, true),
        v = new DataView(kernel(world.ecsState).memory.buffer, pointer, byteCount),
        capsuleOffset = v.getUint32(76, true),
        capsuleCount = v.getUint32(80, true),
        hullOffset = v.getUint32(84, true),
        hullCount = v.getUint32(88, true),
        meshOffset = v.getUint32(96, true),
        meshCount = v.getUint32(100, true),
        sphereOffset = v.getUint32(108, true),
        sphereCount = v.getUint32(112, true);
    const out: CompoundDrawChild[] = [];
    for (let i = 0; i < capsuleCount; ++i) {
        const o = capsuleOffset + i * 32;
        out.push({
            type: ShapeType.Capsule,
            transform: identityTransform(),
            geometry: {
                center1: vector(v, o),
                center2: vector(v, o + 12),
                radius: v.getFloat32(o + 24, true),
            },
        });
    }
    for (let i = 0; i < hullCount; ++i) {
        const o = hullOffset + i * 36,
            nested = pointer + v.getUint32(o + 28, true);
        out.push({
            type: ShapeType.Hull,
            transform: readTransform(v, o),
            geometry: readHullAt(world, nested),
        });
    }
    for (let i = 0; i < meshCount; ++i) {
        const o = meshOffset + i * 60,
            nested = pointer + v.getUint32(o + 40, true);
        out.push({
            type: ShapeType.Mesh,
            transform: readTransform(v, o),
            geometry: {
                data: readMeshData(world, nested),
                scale: vector(v, o + 28),
            },
        });
    }
    for (let i = 0; i < sphereCount; ++i) {
        const o = sphereOffset + i * 20;
        out.push({
            type: ShapeType.Sphere,
            transform: identityTransform(),
            geometry: { center: vector(v, o), radius: v.getFloat32(o + 12, true) },
        });
    }
    return out;
}
