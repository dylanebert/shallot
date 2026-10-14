import type { AABB, Vec3 } from "../common/math";

export { createGrid, createHeightField, createWave } from "./build-nonconvex";
export { getHeightFieldTriangle } from "./read-geometry";

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

/** Per-triangle material index of a height field (b3GetHeightFieldMaterial). */
export function getHeightFieldMaterial(hf: HeightFieldData, triangleIndex: number): number {
    return hf.materialIndices[triangleIndex >> 1];
}
