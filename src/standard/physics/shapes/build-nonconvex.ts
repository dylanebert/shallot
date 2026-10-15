import type { World } from "../../../engine";
import type { Vec3 } from "../common/records";
import { type Kernel, kernel } from "../kernel/kernel";
import type { HeightFieldData, HeightFieldDef } from "./heightfield";
import type { MeshData, MeshDef } from "./mesh";
import { drawHeightField, drawMeshData } from "./readgeometry";

type Scratch = { floats: Float32Array; words: Uint32Array; view: DataView };
const scratch = new WeakMap<Kernel, Scratch>();
function prepare(world: World | undefined, count = 0): [Kernel, Scratch] {
    const k = kernel(world);
    const ptr = k.nonconvexBuildInputPtr(count);
    let s = scratch.get(k);
    if (!s) {
        s = {
            floats: new Float32Array(0),
            words: new Uint32Array(0),
            view: new DataView(k.memory.buffer),
        };
        scratch.set(k, s);
    }
    if (
        s.words.buffer !== k.memory.buffer ||
        s.words.byteOffset !== ptr ||
        s.words.length < 16 + count
    ) {
        s.words = new Uint32Array(k.memory.buffer, ptr, 16 + count);
        s.floats = new Float32Array(k.memory.buffer, ptr, 16 + count);
    }
    return [k, s];
}
function result(k: Kernel, operation: number): MeshData | HeightFieldData | null {
    const ptr = k.nonconvexBuild(operation);
    if (ptr === 0) return null;
    const s = scratch.get(k)!;
    if (s.view.buffer !== k.memory.buffer) s.view = new DataView(k.memory.buffer);
    return operation < 10 ? drawMeshData(s.view, ptr) : drawHeightField(s.view, ptr);
}
/** Builds independent mesh data; degenerate triangles are culled and invalid bounds return null. */
export function createMesh(def: MeshDef, world?: World): MeshData | null {
    const nv = def.vertices.length,
        ni = 3 * Math.trunc(def.indices.length / 3);
    const hasMaterials = def.materialIndices != null;
    const [k, s] = prepare(world, 3 * nv + ni + (hasMaterials ? ni / 3 : 0));
    s.floats[0] = def.weldTolerance ?? 0;
    s.words[1] = Number(def.weldVertices ?? false);
    s.words[2] = Number(def.useMedianSplit ?? false);
    s.words[3] = Number(def.identifyEdges ?? false);
    s.words[4] = nv;
    s.words[5] = ni;
    s.words[6] = Number(hasMaterials);
    for (let i = 0; i < nv; ++i) {
        s.floats[16 + 3 * i] = def.vertices[i].x;
        s.floats[17 + 3 * i] = def.vertices[i].y;
        s.floats[18 + 3 * i] = def.vertices[i].z;
    }
    for (let i = 0; i < ni; ++i) s.words[16 + 3 * nv + i] = def.indices[i];
    if (hasMaterials)
        for (let i = 0; i < ni / 3; ++i) s.words[16 + 3 * nv + ni + i] = def.materialIndices![i];
    return result(k, 0) as MeshData | null;
}
export function createBoxMesh(
    center: Vec3,
    extent: Vec3,
    identifyEdges: boolean,
    world?: World,
): MeshData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = center.x;
    a[1] = center.y;
    a[2] = center.z;
    a[3] = extent.x;
    a[4] = extent.y;
    a[5] = extent.z;
    a[6] = Number(identifyEdges);
    return result(k, 1) as MeshData;
}
export function createHollowBoxMesh(center: Vec3, extent: Vec3, world?: World): MeshData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = center.x;
    a[1] = center.y;
    a[2] = center.z;
    a[3] = extent.x;
    a[4] = extent.y;
    a[5] = extent.z;
    return result(k, 2) as MeshData;
}
export function createGridMesh(
    xCount: number,
    zCount: number,
    cellWidth: number,
    materialCount: number,
    identifyEdges: boolean,
    world?: World,
): MeshData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = xCount;
    a[1] = zCount;
    a[2] = cellWidth;
    a[3] = materialCount;
    a[4] = Number(identifyEdges);
    return result(k, 3) as MeshData;
}
export function createWaveMesh(
    xCount: number,
    zCount: number,
    cellWidth: number,
    amplitude: number,
    rowFrequency: number,
    columnFrequency: number,
    world?: World,
): MeshData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = xCount;
    a[1] = zCount;
    a[2] = cellWidth;
    a[3] = amplitude;
    a[4] = rowFrequency;
    a[5] = columnFrequency;
    return result(k, 4) as MeshData;
}
export function createTorusMesh(
    radialResolution: number,
    tubularResolution: number,
    radius: number,
    thickness: number,
    world?: World,
): MeshData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = radialResolution;
    a[1] = tubularResolution;
    a[2] = radius;
    a[3] = thickness;
    return result(k, 5) as MeshData;
}
/** Quantizes heights and derives adjacency flags from the decompressed field. */
export function createHeightField(def: HeightFieldDef, world?: World): HeightFieldData {
    const count = def.countX * def.countZ,
        cells = (def.countX - 1) * (def.countZ - 1);
    const [k, s] = prepare(world, count + (def.materialIndices === null ? 0 : cells));
    const a = s.floats;
    a[0] = def.countX;
    a[1] = def.countZ;
    a[2] = def.scale.x;
    a[3] = def.scale.y;
    a[4] = def.scale.z;
    a[5] = def.globalMinimumHeight;
    a[6] = def.globalMaximumHeight;
    a[7] = Number(def.clockwiseWinding);
    s.words[8] = Number(def.materialIndices !== null);
    for (let i = 0; i < count; ++i) a[16 + i] = def.heights[i];
    if (def.materialIndices !== null)
        for (let i = 0; i < cells; ++i) s.words[16 + count + i] = def.materialIndices[i];
    return result(k, 10) as HeightFieldData;
}
export function createGrid(
    rowCount: number,
    columnCount: number,
    scale: Vec3,
    makeHoles: boolean,
    world?: World,
): HeightFieldData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = rowCount;
    a[1] = columnCount;
    a[2] = scale.x;
    a[3] = scale.y;
    a[4] = scale.z;
    a[5] = Number(makeHoles);
    return result(k, 11) as HeightFieldData;
}
export function createWave(
    rowCount: number,
    columnCount: number,
    scale: Vec3,
    rowFrequency: number,
    columnFrequency: number,
    makeHoles: boolean,
    world?: World,
): HeightFieldData {
    const [k, s] = prepare(world);
    const a = s.floats;
    a[0] = rowCount;
    a[1] = columnCount;
    a[2] = scale.x;
    a[3] = scale.y;
    a[4] = scale.z;
    a[5] = rowFrequency;
    a[6] = columnFrequency;
    a[7] = Number(makeHoles);
    return result(k, 12) as HeightFieldData;
}
