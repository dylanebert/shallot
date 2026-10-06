import type { Transform, Vec3 } from "../common/math";
import { ShapeType } from "../common/types";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import type { Capsule, Sphere } from "./geometry";
import type { HeightFieldData } from "./heightfield";
import type { HullData } from "./hull";
import type { Mesh, MeshData, MeshNode, MeshTriangle } from "./mesh";

export function drawVector(v: DataView, o: number): Vec3 {
    return { x: v.getFloat32(o, true), y: v.getFloat32(o + 4, true), z: v.getFloat32(o + 8, true) };
}
export function drawTransform(v: DataView, o: number): Transform {
    return { p: drawVector(v, o), q: { v: drawVector(v, o + 12), s: v.getFloat32(o + 24, true) } };
}
export function drawSphere(v: DataView, o: number): Sphere {
    return { center: drawVector(v, o), radius: v.getFloat32(o + 12, true) };
}
export function drawCapsule(v: DataView, o: number): Capsule {
    return {
        center1: drawVector(v, o),
        center2: drawVector(v, o + 12),
        radius: v.getFloat32(o + 24, true),
    };
}
export function drawHull(v: DataView, p: number): HullData {
    const vertexCount = v.getUint32(p + 100, true),
        edgeCount = v.getUint32(p + 112, true),
        faceCount = v.getUint32(p + 120, true);
    const vertices = [],
        points = [],
        edges = [],
        faces = [],
        planes = [];
    for (let i = 0; i < vertexCount; ++i) {
        vertices.push({ edge: v.getUint8(p + v.getUint32(p + 104, true) + i) });
        points.push(drawVector(v, p + v.getUint32(p + 108, true) + 12 * i));
    }
    for (let i = 0; i < edgeCount; ++i) {
        const o = p + v.getUint32(p + 116, true) + 4 * i;
        edges.push({
            next: v.getUint8(o),
            twin: v.getUint8(o + 1),
            origin: v.getUint8(o + 2),
            face: v.getUint8(o + 3),
        });
    }
    for (let i = 0; i < faceCount; ++i) {
        faces.push({ edge: v.getUint8(p + v.getUint32(p + 128, true) + i) });
        const o = p + v.getUint32(p + 124, true) + 16 * i;
        planes.push({ normal: drawVector(v, o), offset: v.getFloat32(o + 12, true) });
    }
    return {
        aabb: { lowerBound: drawVector(v, p + 16), upperBound: drawVector(v, p + 28) },
        surfaceArea: v.getFloat32(p + 40, true),
        volume: v.getFloat32(p + 44, true),
        innerRadius: v.getFloat32(p + 48, true),
        center: drawVector(v, p + 52),
        centralInertia: {
            cx: drawVector(v, p + 64),
            cy: drawVector(v, p + 76),
            cz: drawVector(v, p + 88),
        },
        vertexCount,
        edgeCount,
        faceCount,
        vertices,
        points,
        edges,
        faces,
        planes,
        hash: v.getBigUint64(p + 8, true),
    };
}
function meshData(v: DataView, p: number): MeshData {
    const nodeOffset = v.getUint32(p + 56, true),
        nodeCount = v.getUint32(p + 60, true),
        vertexOffset = v.getUint32(p + 64, true),
        vertexCount = v.getUint32(p + 68, true),
        triangleOffset = v.getUint32(p + 72, true),
        triangleCount = v.getUint32(p + 76, true),
        materialOffset = v.getUint32(p + 80, true),
        materialCount = v.getUint32(p + 84, true),
        flagsOffset = v.getUint32(p + 88, true);
    const nodes: MeshNode[] = [],
        vertices: Vec3[] = [],
        triangles: MeshTriangle[] = [];
    for (let i = 0; i < nodeCount; ++i) {
        const o = p + nodeOffset + i * 32,
            data = v.getUint32(o + 12, true),
            leaf = (data & 3) === 3;
        nodes.push({
            lowerBound: drawVector(v, o),
            upperBound: drawVector(v, o + 16),
            leaf,
            axis: leaf ? 0 : data & 3,
            childOffset: leaf ? 0 : data >>> 2,
            triangleCount: leaf ? data >>> 2 : 0,
            triangleOffset: v.getUint32(o + 28, true),
        });
    }
    for (let i = 0; i < vertexCount; ++i) vertices.push(drawVector(v, p + vertexOffset + i * 12));
    for (let i = 0; i < triangleCount; ++i) {
        const o = p + triangleOffset + i * 12;
        triangles.push({
            index1: v.getUint32(o, true),
            index2: v.getUint32(o + 4, true),
            index3: v.getUint32(o + 8, true),
        });
    }
    const materialIndices = new Array<number>(triangleCount),
        flags = new Array<number>(triangleCount);
    for (let i = 0; i < triangleCount; ++i) {
        materialIndices[i] = v.getUint8(p + materialOffset + i);
        flags[i] = v.getUint8(p + flagsOffset + i);
    }
    return {
        bounds: { lowerBound: drawVector(v, p + 20), upperBound: drawVector(v, p + 32) },
        surfaceArea: v.getFloat32(p + 44, true),
        treeHeight: v.getUint32(p + 48, true),
        degenerateCount: v.getUint32(p + 52, true),
        nodes,
        vertices,
        triangles,
        materialIndices,
        materialCount,
        flags,
    };
}
export function drawMesh(v: DataView, o: number): Mesh {
    return { scale: drawVector(v, o), data: meshData(v, o + 12) };
}
export function drawHeightField(v: DataView, p: number): HeightFieldData {
    const columnCount = v.getUint32(p + 68, true),
        rowCount = v.getUint32(p + 72, true),
        heightsOffset = v.getUint32(p + 76, true),
        materialOffset = v.getUint32(p + 80, true),
        flagsOffset = v.getUint32(p + 84, true),
        cellCount = (columnCount - 1) * (rowCount - 1);
    const compressedHeights = new Array<number>(columnCount * rowCount),
        materialIndices = new Array<number>(cellCount),
        flags = new Array<number>(cellCount * 2);
    for (let i = 0; i < compressedHeights.length; ++i)
        compressedHeights[i] = v.getUint16(p + heightsOffset + i * 2, true);
    for (let i = 0; i < cellCount; ++i) materialIndices[i] = v.getUint8(p + materialOffset + i);
    for (let i = 0; i < flags.length; ++i) flags[i] = v.getUint8(p + flagsOffset + i);
    return {
        aabb: { lowerBound: drawVector(v, p + 20), upperBound: drawVector(v, p + 32) },
        minHeight: v.getFloat32(p + 44, true),
        maxHeight: v.getFloat32(p + 48, true),
        heightScale: v.getFloat32(p + 52, true),
        scale: drawVector(v, p + 56),
        columnCount,
        rowCount,
        compressedHeights,
        materialIndices,
        flags,
        clockwise: v.getUint8(p + 88) !== 0,
    };
}
export type CompoundDrawChild = {
    type: number;
    transform: Transform;
    geometry: Sphere | Capsule | HullData | Mesh;
};

// Requested observations use the same kernel primitive encoder as the public draw.
export function readShapeMesh(world: WorldState, shape: number): Mesh {
    const k = kernel(world.ecsState);
    const start = k.worldDrawShape(world.worldId, shape);
    try {
        return drawMesh(new DataView(k.memory.buffer), k.worldDrawPtr() + start * 4 + 40);
    } finally {
        k.worldDrawRelease(start);
    }
}
export function readShapeHeightField(world: WorldState, shape: number): HeightFieldData {
    const k = kernel(world.ecsState);
    const start = k.worldDrawShape(world.worldId, shape);
    try {
        return drawHeightField(new DataView(k.memory.buffer), k.worldDrawPtr() + start * 4 + 52);
    } finally {
        k.worldDrawRelease(start);
    }
}
export function readCompoundChildren(world: WorldState, shape: number): CompoundDrawChild[] {
    const k = kernel(world.ecsState);
    const start = k.worldDrawShape(world.worldId, shape);
    try {
        const v = new DataView(k.memory.buffer),
            end = k.worldDrawPtr() + k.worldDrawLen() * 4;
        const children: CompoundDrawChild[] = [];
        for (let o = k.worldDrawPtr() + start * 4; o < end; o += v.getUint32(o + 4, true) * 4) {
            const type = v.getUint32(o, true),
                p = o + 40;
            const geometry =
                type === ShapeType.Sphere
                    ? drawSphere(v, p)
                    : type === ShapeType.Capsule
                      ? drawCapsule(v, p)
                      : type === ShapeType.Hull
                        ? drawHull(v, p + 12)
                        : drawMesh(v, p);
            children.push({ type, transform: drawTransform(v, o + 12), geometry });
        }
        return children;
    } finally {
        k.worldDrawRelease(start);
    }
}
