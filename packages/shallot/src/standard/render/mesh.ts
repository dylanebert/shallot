import { registry, type Registry } from "../../engine";
import { Shape, linearToSrgb, srgbToLinear } from "../../engine/utils";
import { buf, traits, type Buf, CHUNK_MASK, CHUNK_SHIFT } from "../../engine";
import { createFieldProxy, formatHex, type FieldProxy } from "../../engine/ecs/core";
import { Transform } from "../transforms";
import { SurfaceType, surfaceRegistry } from "./surface";

export const MAX_SURFACES = 32;
export const MAX_SHAPES = 256;
export const MAX_BATCH_SLOTS = MAX_SHAPES * MAX_SURFACES;
const INVALID_SHAPE = 0xffffffff;

export interface MeshData {
    vertices: Float32Array<ArrayBuffer>;
    indices: Uint16Array<ArrayBuffer>;
    vertexCount: number;
    indexCount: number;
}

export const meshRegistry: Registry<MeshData> = registry(MAX_SHAPES);

function ensureBuiltIns(): void {
    if (meshRegistry.count() === 0) {
        meshRegistry.add(createBox(), "box");
        meshRegistry.add(createSphere(), "sphere");
        meshRegistry.add(createCapsule(), "capsule");
        meshRegistry.add(createPlane(), "plane");
    }
}

export const MeshShape = {
    Box: 0,
    Sphere: 1,
    Capsule: 2,
    Plane: 3,
} as const;

export function mesh(data: MeshData, name?: string): number {
    ensureBuiltIns();
    return meshRegistry.add(data, name);
}

export function getMeshByName(name: string): number | undefined {
    ensureBuiltIns();
    return meshRegistry.getByName(name);
}

export function getMeshName(id: number): string | undefined {
    ensureBuiltIns();
    return meshRegistry.getName(id);
}

export function getMesh(id: number): MeshData | undefined {
    ensureBuiltIns();
    return meshRegistry.get(id);
}

export function getMeshVersion(): number {
    return meshRegistry.version;
}

export function meshCount(): number {
    ensureBuiltIns();
    return meshRegistry.count();
}

export function clearMeshes(): void {
    meshRegistry.clear();
    _dynamics.clear();
}

export const PartShapes = buf(Uint8Array, 1, 0);
export const PartColors = buf(Float32Array, 4, 0);
export const PartSizes = buf(Float32Array, 4, 0);
export const PartPBR = buf(Float32Array, 4, 0);
export const PartEmission = buf(Float32Array, 4, 0);
export const PartVolumes = buf(Uint8Array, 1, 0);
export const PartGeometry = buf(Uint32Array, 1, INVALID_SHAPE);

export const Volume = {
    Solid: 0,
    HalfSpace: 1,
} as const;

function hexColorProxy(ref: Buf<Float32Array>): FieldProxy {
    const chunks = ref.chunks;
    function getValue(eid: number): number {
        const chunk = chunks[eid >>> CHUNK_SHIFT];
        const o = (eid & CHUNK_MASK) * 4;
        const r = Math.round(linearToSrgb(chunk[o]) * 255);
        const g = Math.round(linearToSrgb(chunk[o + 1]) * 255);
        const b = Math.round(linearToSrgb(chunk[o + 2]) * 255);
        return (r << 16) | (g << 8) | b;
    }

    function setValue(eid: number, value: number): void {
        const chunk = chunks[eid >>> CHUNK_SHIFT];
        const o = (eid & CHUNK_MASK) * 4;
        chunk[o] = srgbToLinear(((value >> 16) & 0xff) / 255);
        chunk[o + 1] = srgbToLinear(((value >> 8) & 0xff) / 255);
        chunk[o + 2] = srgbToLinear((value & 0xff) / 255);
    }

    return new Proxy([] as unknown as FieldProxy, {
        get(_, prop) {
            if (prop === "get") return getValue;
            if (prop === "set") return setValue;
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return getValue(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            setValue(eid, value);
            return true;
        },
    });
}

const SHAPE_NAMES: Record<string, number> = {
    box: Shape.Box,
    sphere: Shape.Sphere,
    capsule: Shape.Capsule,
    plane: Shape.Plane,
    mesh: Shape.Mesh,
};

function parseShape(name: string): number | undefined {
    return SHAPE_NAMES[name];
}

function formatShape(value: number): string | undefined {
    for (const [name, v] of Object.entries(SHAPE_NAMES)) {
        if (v === value) return name;
    }
    return undefined;
}

export const PartSurfaces = buf(Uint16Array, 1, 0);

export const Part: {
    shape: FieldProxy;
    surface: FieldProxy;
    volume: FieldProxy;
    color: FieldProxy;
    colorR: FieldProxy;
    colorG: FieldProxy;
    colorB: FieldProxy;
    opacity: FieldProxy;
    sizeX: FieldProxy;
    sizeY: FieldProxy;
    sizeZ: FieldProxy;
    shadows: FieldProxy;
    roughness: FieldProxy;
    reflectivity: FieldProxy;
    emission: FieldProxy;
    emissionIntensity: FieldProxy;
} = {
    shape: createFieldProxy(PartShapes, 1, 0),
    surface: createFieldProxy(PartSurfaces, 1, 0),
    volume: createFieldProxy(PartVolumes, 1, 0),
    color: hexColorProxy(PartColors),
    colorR: createFieldProxy(PartColors, 4, 0),
    colorG: createFieldProxy(PartColors, 4, 1),
    colorB: createFieldProxy(PartColors, 4, 2),
    opacity: createFieldProxy(PartColors, 4, 3),
    sizeX: createFieldProxy(PartSizes, 4, 0),
    sizeY: createFieldProxy(PartSizes, 4, 1),
    sizeZ: createFieldProxy(PartSizes, 4, 2),
    shadows: createFieldProxy(PartSizes, 4, 3),
    roughness: createFieldProxy(PartPBR, 4, 0),
    reflectivity: createFieldProxy(PartPBR, 4, 1),
    emission: hexColorProxy(PartEmission),
    emissionIntensity: createFieldProxy(PartEmission, 4, 3),
};

traits(Part, {
    requires: [Transform],
    defaults: () => ({
        shape: Shape.Box,
        surface: SurfaceType.Default,
        color: 0xffffff,
        opacity: 1.0,
        sizeX: 1,
        sizeY: 1,
        sizeZ: 1,
        shadows: 1,
        roughness: 1.0,
        reflectivity: 0.0,
        emission: 0x000000,
        emissionIntensity: 0.0,
        volume: Volume.Solid,
    }),
    parse: { shape: parseShape, surface: (name: string) => surfaceRegistry.getByName(name) },
    format: {
        shape: formatShape,
        surface: (id: number) => surfaceRegistry.getName(id),
        color: formatHex,
        emission: formatHex,
    },
    enums: { surface: SurfaceType, volume: Volume, shape: Shape },
});

export const MeshGeometryData = buf(Uint32Array, 1, 0);

export const Mesh: {
    geometry: FieldProxy;
} = {
    geometry: createFieldProxy(MeshGeometryData, 1, 0),
};

traits(Mesh, {
    requires: [Part],
    defaults: () => ({
        geometry: MeshShape.Box,
    }),
    parse: { geometry: getMeshByName },
    format: { geometry: getMeshName },
});

export const Dynamic = {};

traits(Dynamic, {
    requires: [Part],
});

export interface DynamicInfo {
    baseFloatOffset: number;
    atlasFloatOffset: number;
    atlasIndexOffset: number;
    vertexCount: number;
}

interface DynamicEntry {
    meshId: number;
    priorShape: number;
    priorGeometry: number;
    baseFloatOffset: number;
    atlasFloatOffset: number;
    atlasIndexOffset: number;
    vertexCount: number;
}

const _dynamics = new Map<number, DynamicEntry>();
const _unboundedShapes = new Set<number>();

export function isUnboundedShape(shapeId: number): boolean {
    return _unboundedShapes.has(shapeId);
}

export function dynamicInfo(eid: number): DynamicInfo | undefined {
    const d = _dynamics.get(eid);
    if (!d || d.atlasFloatOffset < 0) return undefined;
    return {
        baseFloatOffset: d.baseFloatOffset,
        atlasFloatOffset: d.atlasFloatOffset,
        atlasIndexOffset: d.atlasIndexOffset,
        vertexCount: d.vertexCount,
    };
}

export function isDynamic(eid: number): boolean {
    return _dynamics.has(eid);
}

export function allocateDynamic(eid: number): void {
    if (_dynamics.has(eid)) return;
    const baseShapeId = getMeshId(eid);
    const baseMesh = getMesh(baseShapeId);
    if (!baseMesh) return;
    const cloned: MeshData = {
        vertices: new Float32Array(baseMesh.vertices),
        indices: new Uint16Array(baseMesh.indices),
        vertexCount: baseMesh.vertexCount,
        indexCount: baseMesh.indexCount,
    };
    const meshId = mesh(cloned);
    _unboundedShapes.add(meshId);
    _dynamics.set(eid, {
        meshId,
        priorShape: Part.shape[eid],
        priorGeometry: Mesh.geometry[eid],
        baseFloatOffset: -1,
        atlasFloatOffset: -1,
        atlasIndexOffset: -1,
        vertexCount: baseMesh.vertexCount,
    });
    Mesh.geometry[eid] = meshId;
    Part.shape[eid] = Shape.Mesh;
}

export function deallocateDynamic(eid: number): void {
    const d = _dynamics.get(eid);
    if (d) {
        _unboundedShapes.delete(d.meshId);
        Part.shape[eid] = d.priorShape;
        Mesh.geometry[eid] = d.priorGeometry;
    }
    _dynamics.delete(eid);
}

export function getMeshId(eid: number): number {
    const shape = Part.shape[eid];
    switch (shape) {
        case Shape.Box:
            return MeshShape.Box;
        case Shape.Sphere:
            return MeshShape.Sphere;
        case Shape.Capsule:
            return MeshShape.Capsule;
        case Shape.Plane:
            return MeshShape.Plane;
        case Shape.Mesh:
            return Mesh.geometry[eid];
        default:
            return MeshShape.Box;
    }
}

export interface AABB {
    minX: number;
    minY: number;
    minZ: number;
    maxX: number;
    maxY: number;
    maxZ: number;
}

export function computeShapeAABB(mesh: MeshData): AABB {
    const { vertices, vertexCount } = mesh;
    const stride = 8;

    if (vertexCount === 0) {
        return { minX: 0, minY: 0, minZ: 0, maxX: 0, maxY: 0, maxZ: 0 };
    }

    let minX = vertices[0];
    let minY = vertices[1];
    let minZ = vertices[2];
    let maxX = vertices[0];
    let maxY = vertices[1];
    let maxZ = vertices[2];

    for (let i = 1; i < vertexCount; i++) {
        const x = vertices[i * stride];
        const y = vertices[i * stride + 1];
        const z = vertices[i * stride + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    return { minX, minY, minZ, maxX, maxY, maxZ };
}

export function createBox(): MeshData {
    // prettier-ignore
    const vertices = new Float32Array([
        // +Z face
        -0.5, -0.5, 0.5, 0, 0, 1, 0, 0, 0.5, -0.5, 0.5, 0, 0, 1, 1, 0, 0.5, 0.5, 0.5, 0, 0, 1, 1, 1,
        -0.5, 0.5, 0.5, 0, 0, 1, 0, 1,
        // -Z face
        0.5, -0.5, -0.5, 0, 0, -1, 0, 0, -0.5, -0.5, -0.5, 0, 0, -1, 1, 0, -0.5, 0.5, -0.5, 0, 0,
        -1, 1, 1, 0.5, 0.5, -0.5, 0, 0, -1, 0, 1,
        // +Y face
        -0.5, 0.5, 0.5, 0, 1, 0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 0, 0.5, 0.5, -0.5, 0, 1, 0, 1, 1,
        -0.5, 0.5, -0.5, 0, 1, 0, 0, 1,
        // -Y face
        -0.5, -0.5, -0.5, 0, -1, 0, 0, 0, 0.5, -0.5, -0.5, 0, -1, 0, 1, 0, 0.5, -0.5, 0.5, 0, -1, 0,
        1, 1, -0.5, -0.5, 0.5, 0, -1, 0, 0, 1,
        // +X face
        0.5, -0.5, 0.5, 1, 0, 0, 0, 0, 0.5, -0.5, -0.5, 1, 0, 0, 1, 0, 0.5, 0.5, -0.5, 1, 0, 0, 1,
        1, 0.5, 0.5, 0.5, 1, 0, 0, 0, 1,
        // -X face
        -0.5, -0.5, -0.5, -1, 0, 0, 0, 0, -0.5, -0.5, 0.5, -1, 0, 0, 1, 0, -0.5, 0.5, 0.5, -1, 0, 0,
        1, 1, -0.5, 0.5, -0.5, -1, 0, 0, 0, 1,
    ]);

    const indices = new Uint16Array([
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18,
        16, 18, 19, 20, 21, 22, 20, 22, 23,
    ]);

    return { vertices, indices, vertexCount: 24, indexCount: 36 };
}

export function createSphere(segments = 32, rings = 16): MeshData {
    const vertices: number[] = [];
    const indices: number[] = [];
    const radius = 0.5;

    for (let y = 0; y <= rings; y++) {
        const v = y / rings;
        const theta = v * Math.PI;

        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;

            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);

            vertices.push(nx * radius, ny * radius, nz * radius, nx, ny, nz, u, v);
        }
    }

    for (let y = 0; y < rings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = y * (segments + 1) + x;
            const b = a + segments + 1;

            indices.push(a, a + 1, b);
            indices.push(a + 1, b + 1, b);
        }
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint16Array(indices),
        vertexCount: (rings + 1) * (segments + 1),
        indexCount: rings * segments * 6,
    };
}

export function createCapsule(segments = 32, rings = 16): MeshData {
    const vertices: number[] = [];
    const indices: number[] = [];
    const radius = 0.5;
    const halfHeight = 0.5;
    const halfRings = rings / 2;

    for (let y = 0; y <= halfRings; y++) {
        const theta = (y / halfRings) * (Math.PI / 2);
        const v = (y / halfRings) * 0.25;
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            vertices.push(nx * radius, ny * radius + halfHeight, nz * radius, nx, ny, nz, u, v);
        }
    }

    for (let x = 0; x <= segments; x++) {
        const u = x / segments;
        const phi = u * Math.PI * 2;
        const nx = Math.cos(phi);
        const nz = Math.sin(phi);
        vertices.push(nx * radius, halfHeight, nz * radius, nx, 0, nz, u, 0.25);
    }
    for (let x = 0; x <= segments; x++) {
        const u = x / segments;
        const phi = u * Math.PI * 2;
        const nx = Math.cos(phi);
        const nz = Math.sin(phi);
        vertices.push(nx * radius, -halfHeight, nz * radius, nx, 0, nz, u, 0.75);
    }

    for (let y = 0; y <= halfRings; y++) {
        const theta = (y / halfRings) * (Math.PI / 2);
        const v = 0.75 + (y / halfRings) * 0.25;
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = -Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            vertices.push(nx * radius, ny * radius - halfHeight, nz * radius, nx, ny, nz, u, v);
        }
    }

    const stride = segments + 1;

    for (let y = 0; y < halfRings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = y * stride + x;
            const b = a + stride;
            indices.push(a, a + 1, b);
            indices.push(a + 1, b + 1, b);
        }
    }

    const cylTop = (halfRings + 1) * stride;
    const cylBot = cylTop + stride;
    for (let x = 0; x < segments; x++) {
        const a = cylTop + x;
        const b = cylBot + x;
        indices.push(a, a + 1, b);
        indices.push(a + 1, b + 1, b);
    }

    const botStart = cylBot + stride;
    for (let y = 0; y < halfRings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = botStart + y * stride + x;
            const b = a + stride;
            indices.push(a, b, a + 1);
            indices.push(a + 1, b, b + 1);
        }
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint16Array(indices),
        vertexCount: vertices.length / 8,
        indexCount: indices.length,
    };
}

export function createPlane(): MeshData {
    // prettier-ignore
    const vertices = new Float32Array([
        -0.5, 0, 0.5, 0, 1, 0, 0, 0, 0.5, 0, 0.5, 0, 1, 0, 1, 0, 0.5, 0, -0.5, 0, 1, 0, 1, 1, -0.5,
        0, -0.5, 0, 1, 0, 0, 1,
    ]);

    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

    return { vertices, indices, vertexCount: 4, indexCount: 6 };
}

export function createCone(segments = 16): MeshData {
    const vertices: number[] = [];
    const indices: number[] = [];
    const radius = 0.5;
    const height = 1.0;

    vertices.push(0, height / 2, 0, 0, 1, 0, 0.5, 0);

    for (let i = 0; i <= segments; i++) {
        const u = i / segments;
        const angle = u * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const ny = radius / height;
        const len = Math.sqrt(x * x + ny * ny + z * z);
        vertices.push(x, -height / 2, z, x / len, ny / len, z / len, u, 1);
    }

    for (let i = 0; i < segments; i++) {
        indices.push(0, i + 2, i + 1);
    }

    const baseCenterIndex = vertices.length / 8;
    vertices.push(0, -height / 2, 0, 0, -1, 0, 0.5, 0.5);

    const baseStartIndex = vertices.length / 8;
    for (let i = 0; i <= segments; i++) {
        const u = i / segments;
        const angle = u * Math.PI * 2;
        vertices.push(
            Math.cos(angle) * radius,
            -height / 2,
            Math.sin(angle) * radius,
            0,
            -1,
            0,
            u,
            1,
        );
    }

    for (let i = 0; i < segments; i++) {
        indices.push(baseCenterIndex, baseStartIndex + i, baseStartIndex + i + 1);
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint16Array(indices),
        vertexCount: vertices.length / 8,
        indexCount: indices.length,
    };
}

interface ShapeMeta {
    vertexOffset: number;
    indexOffset: number;
    triCount: number;
}

export interface ShapeAtlas {
    vertices: GPUBuffer;
    indices: GPUBuffer;
    meta: GPUBuffer;
    baseVertices: GPUBuffer;
    shapeCount: number;
    maxTriangles: number;
    vertexCapacity: number;
    indexCapacity: number;
    baseVertexCapacity: number;
    dynOffsets: Map<number, number>;
}

function packMeshData(): {
    verticesData: Float32Array;
    indicesData: Uint32Array;
    metaData: Uint32Array;
    baseVerticesData: Float32Array;
    shapeCount: number;
    maxTriangles: number;
    dynOffsets: Map<number, number>;
} {
    const allVertices: number[] = [];
    const allIndices: number[] = [];
    const baseVertices: number[] = [];
    const shapeMetas: ShapeMeta[] = [];
    const dynOffsets = new Map<number, number>();
    const dynMeshIds = new Map<number, DynamicEntry>();
    for (const d of _dynamics.values()) dynMeshIds.set(d.meshId, d);

    let vertexOffset = 0;
    let indexOffset = 0;
    let maxTriangles = 0;
    let baseVertexOffset = 0;

    const shapeCount = meshCount();
    for (let shapeId = 0; shapeId < shapeCount; shapeId++) {
        const m = getMesh(shapeId);
        if (!m) {
            shapeMetas.push({ vertexOffset: 0, indexOffset: 0, triCount: 0 });
            continue;
        }

        const triCount = m.indexCount / 3;
        shapeMetas.push({ vertexOffset, indexOffset, triCount });

        const dynEntry = dynMeshIds.get(shapeId);
        if (dynEntry) {
            dynOffsets.set(shapeId, vertexOffset * 4);
            dynEntry.atlasFloatOffset = vertexOffset;
            dynEntry.atlasIndexOffset = indexOffset;
            dynEntry.baseFloatOffset = baseVertexOffset;
            for (let i = 0; i < m.vertices.length; i++) {
                baseVertices.push(m.vertices[i]);
            }
            baseVertexOffset += m.vertices.length;
        }

        for (let i = 0; i < m.vertices.length; i++) {
            allVertices.push(m.vertices[i]);
        }

        for (let i = 0; i < m.indices.length; i++) {
            allIndices.push(m.indices[i]);
        }

        vertexOffset += m.vertices.length;
        indexOffset += m.indices.length;
        maxTriangles += triCount;
    }

    const verticesData = new Float32Array(allVertices);
    const indicesData = new Uint32Array(allIndices);
    const baseVerticesData = new Float32Array(baseVertices);
    const metaData = new Uint32Array(MAX_SHAPES * 4);

    for (let i = 0; i < shapeMetas.length; i++) {
        metaData[i * 4] = shapeMetas[i].vertexOffset;
        metaData[i * 4 + 1] = shapeMetas[i].indexOffset;
        metaData[i * 4 + 2] = shapeMetas[i].triCount;
        metaData[i * 4 + 3] = 0;
    }

    return {
        verticesData,
        indicesData,
        metaData,
        baseVerticesData,
        shapeCount: shapeMetas.filter((m) => m.triCount > 0).length,
        maxTriangles,
        dynOffsets,
    };
}

function atlasBufferSize(dataLength: number): number {
    return Math.max(dataLength * 2, 256) * 4;
}

export function createShapeAtlas(device: GPUDevice): ShapeAtlas {
    const {
        verticesData,
        indicesData,
        metaData,
        baseVerticesData,
        shapeCount,
        maxTriangles,
        dynOffsets,
    } = packMeshData();

    const vertexCapacity = atlasBufferSize(verticesData.length);
    const indexCapacity = atlasBufferSize(indicesData.length);
    const baseVertexCapacity = atlasBufferSize(Math.max(baseVerticesData.length, 1));

    const vertices = device.createBuffer({
        label: "unified-vertices",
        size: vertexCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertices, 0, verticesData as Float32Array<ArrayBuffer>);

    const indices = device.createBuffer({
        label: "unified-indices",
        size: indexCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indices, 0, indicesData as Uint32Array<ArrayBuffer>);

    const meta = device.createBuffer({
        label: "unified-meta",
        size: MAX_SHAPES * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(meta, 0, metaData as Uint32Array<ArrayBuffer>);

    const baseVerticesBuffer = device.createBuffer({
        label: "dynamic-base-vertices",
        size: baseVertexCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (baseVerticesData.length > 0) {
        device.queue.writeBuffer(
            baseVerticesBuffer,
            0,
            baseVerticesData as Float32Array<ArrayBuffer>,
        );
    }

    return {
        vertices,
        indices,
        meta,
        baseVertices: baseVerticesBuffer,
        shapeCount,
        maxTriangles,
        vertexCapacity,
        indexCapacity,
        baseVertexCapacity,
        dynOffsets,
    };
}

export function updateShapeAtlas(device: GPUDevice, atlas: ShapeAtlas): void {
    const { verticesData, indicesData, metaData, baseVerticesData, dynOffsets } = packMeshData();

    const needsVertexGrow = verticesData.byteLength > atlas.vertexCapacity;
    const needsIndexGrow = indicesData.byteLength > atlas.indexCapacity;
    const needsBaseGrow = baseVerticesData.byteLength > atlas.baseVertexCapacity;

    if (needsVertexGrow) {
        atlas.vertices.destroy();
        atlas.vertexCapacity = atlasBufferSize(verticesData.length);
        atlas.vertices = device.createBuffer({
            label: "unified-vertices",
            size: atlas.vertexCapacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    if (needsIndexGrow) {
        atlas.indices.destroy();
        atlas.indexCapacity = atlasBufferSize(indicesData.length);
        atlas.indices = device.createBuffer({
            label: "unified-indices",
            size: atlas.indexCapacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
    }

    if (needsBaseGrow) {
        atlas.baseVertices.destroy();
        atlas.baseVertexCapacity = atlasBufferSize(Math.max(baseVerticesData.length, 1));
        atlas.baseVertices = device.createBuffer({
            label: "dynamic-base-vertices",
            size: atlas.baseVertexCapacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    atlas.dynOffsets = dynOffsets;
    device.queue.writeBuffer(atlas.vertices, 0, verticesData as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(atlas.indices, 0, indicesData as Uint32Array<ArrayBuffer>);
    device.queue.writeBuffer(atlas.meta, 0, metaData as Uint32Array<ArrayBuffer>);
    if (baseVerticesData.length > 0) {
        device.queue.writeBuffer(
            atlas.baseVertices,
            0,
            baseVerticesData as Float32Array<ArrayBuffer>,
        );
    }
}
