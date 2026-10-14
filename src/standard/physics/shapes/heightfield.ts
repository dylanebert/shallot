import { type AABB, type Vec3, vec3 } from "../common/math";
import { MeshEdgeFlags, type Triangle } from "./mesh";

export { createGrid, createHeightField, createWave } from "./build-nonconvex";

const f32 = Math.fround;

/** Reserved material index designating a hole (skipped cell) in a height field (B3_HEIGHT_FIELD_HOLE). */
export const HEIGHT_FIELD_HOLE = 0xff;

/** Data used to create a height field (b3HeightFieldDef). */
export type HeightFieldDef = {
    /** Grid point heights, row-major, length = countX * countZ. */
    heights: number[];
    /** Per-cell material index (0xFF = hole), length = (countX-1)*(countZ-1); null → all zero. */
    materialIndices: number[] | null;
    /** Overall scale; all components must be positive. */
    scale: Vec3;
    /** Number of grid lines along the x-axis (columns). */
    countX: number;
    /** Number of grid lines along the z-axis (rows). */
    countZ: number;
    /** Global min/max heights for quantization; all heights clamp to this range (unscaled). */
    globalMinimumHeight: number;
    globalMaximumHeight: number;
    /** Clockwise winding effectively inverts the field along the y-axis. */
    clockwiseWinding: boolean;
};

/** A height field with compressed storage (b3HeightFieldData), modeled as a struct of arrays. */
export type HeightFieldData = {
    aabb: AABB;
    minHeight: number;
    maxHeight: number;
    heightScale: number;
    scale: Vec3;
    columnCount: number;
    rowCount: number;
    /** uint16 quantized height per grid point (columnCount * rowCount). */
    compressedHeights: number[];
    /** uint8 material index per cell ((columnCount-1) * (rowCount-1)). */
    materialIndices: number[];
    /** uint8 edge flags per triangle (2 * cellCount). */
    flags: number[];
    clockwise: boolean;
};

// Decode the four corner vertices of a cell into local space (b3GetHeightFieldCellCorners).
// corners: [ (col,row), (col+1,row), (col,row+1), (col+1,row+1) ].
function getCellCorners(
    hf: HeightFieldData,
    row: number,
    column: number,
): [Vec3, Vec3, Vec3, Vec3] {
    const columnCount = hf.columnCount;
    const index11 = row * columnCount + column;
    const index12 = index11 + 1;
    const index21 = (row + 1) * columnCount + column;
    const index22 = index21 + 1;

    const minHeight = hf.minHeight;
    const heightScale = hf.heightScale;
    const heights = hf.compressedHeights;

    const height11 = f32(minHeight + f32(heightScale * heights[index11]));
    const height12 = f32(minHeight + f32(heightScale * heights[index12]));
    const height21 = f32(minHeight + f32(heightScale * heights[index21]));
    const height22 = f32(minHeight + f32(heightScale * heights[index22]));

    const x1 = f32(column);
    const x2 = f32(column + 1);
    const z1 = f32(row);
    const z2 = f32(row + 1);

    const scale = hf.scale;
    return [
        vec3.mul(scale, { x: x1, y: height11, z: z1 }),
        vec3.mul(scale, { x: x2, y: height12, z: z1 }),
        vec3.mul(scale, { x: x1, y: height21, z: z2 }),
        vec3.mul(scale, { x: x2, y: height22, z: z2 }),
    ];
}

/**
 * Fetch a height-field triangle by index (b3GetHeightFieldTriangle). The two triangles of a cell share
 * the diagonal (edge2); clockwise winding swaps vertices 1/2 and remaps edge1 <-> edge3 flags.
 */
export function getHeightFieldTriangle(hf: HeightFieldData, triangleIndex: number): Triangle {
    let flags = hf.flags[triangleIndex];

    const columnCount = hf.columnCount;
    const quadIndex = triangleIndex >> 1;
    const row = Math.trunc(quadIndex / (columnCount - 1));
    const column = quadIndex - row * (columnCount - 1);

    const index11 = row * columnCount + column;
    const index12 = index11 + 1;
    const index21 = (row + 1) * columnCount + column;
    const index22 = index21 + 1;

    const corners = getCellCorners(hf, row, column);

    let vertices: [Vec3, Vec3, Vec3];
    let i1: number;
    let i2: number;
    let i3: number;

    if ((triangleIndex & 1) === 0) {
        vertices = [corners[0], corners[2], corners[1]];
        i1 = index11;
        i2 = index21;
        i3 = index12;
    } else {
        vertices = [corners[3], corners[1], corners[2]];
        i1 = index22;
        i2 = index12;
        i3 = index21;
    }

    if (hf.clockwise) {
        const tmp = vertices[1];
        vertices[1] = vertices[2];
        vertices[2] = tmp;
        const ti = i2;
        i2 = i3;
        i3 = ti;

        // Reversing winding swaps edge1 and edge3; edge2 (the diagonal) is preserved.
        const edge1Bits = flags & (MeshEdgeFlags.ConcaveEdge1 | MeshEdgeFlags.InverseConcaveEdge1);
        const edge3Bits = flags & (MeshEdgeFlags.ConcaveEdge3 | MeshEdgeFlags.InverseConcaveEdge3);
        flags &= ~(
            MeshEdgeFlags.ConcaveEdge1 |
            MeshEdgeFlags.ConcaveEdge3 |
            MeshEdgeFlags.InverseConcaveEdge1 |
            MeshEdgeFlags.InverseConcaveEdge3
        );
        flags |= edge1Bits << 2;
        flags |= edge3Bits >> 2;
    }

    return { vertices, i1, i2, i3, flags };
}

/** Per-triangle material index of a height field (b3GetHeightFieldMaterial). */
export function getHeightFieldMaterial(hf: HeightFieldData, triangleIndex: number): number {
    return hf.materialIndices[triangleIndex >> 1];
}
