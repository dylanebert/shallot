// Height field: a regular grid of heights forming a static triangle terrain. Ported from Box3D's
// height_field.c (Erin Catto, MIT). Heights are quantized to uint16 for storage; the collision
// triangles use the *decompressed* heights, so the quantization is load-bearing for bit-exactness and
// is replicated exactly. Per-cell convexity flags mark flat/concave internal edges (used by the mesh-
// contact ghost culling). Height fields are static-only collision geometry.
//
// The C stores the field as one byte blob with offsets into the height/material/flag arrays; the port
// models it as a plain struct of arrays. The raw bytes are never hashed by the sim (only body/contact
// state is), so the representation is free — the compression, the flag bits, and the query's triangle-
// visit order are what must stay bit-exact. fround discipline per the README.

import { LINEAR_SLOP, MAX_AABB_MARGIN, OVERLAP_SLOP } from "./core";
import {
    type CastOutput,
    computeProxyAABB,
    type DistanceInput,
    emptyCache,
    emptyCastOutput,
    makeLocalProxy,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeCastPairInput,
    type ShapeProxy,
    shapeCast,
    shapeDistance,
} from "./distance";
import type { Capsule } from "./geometry";
import {
    type AABB,
    aabb,
    clampf,
    FLT_MAX,
    intersectRayTriangle,
    maxf,
    minf,
    type Plane,
    plane,
    quat,
    rayCastAABB,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "./math";
import { MeshEdgeFlags, type Triangle, testBoundsTriangleOverlap } from "./mesh";
import type { PlaneResult } from "./mover";

const f32 = Math.fround;
const UINT16_MAX = 65535;

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

/**
 * Build a height field from a def (b3CreateHeightField): quantize heights to uint16, derive per-cell
 * convexity flags from the decompressed heights + neighbour planes, and compute the local AABB.
 */
export function createHeightField(data: HeightFieldDef): HeightFieldData {
    const columnCount = data.countX;
    const rowCount = data.countZ;

    const heightCount = columnCount * rowCount;
    const cellCount = (columnCount - 1) * (rowCount - 1);
    const triangleCount = 2 * cellCount;

    const compressedHeights = new Array<number>(heightCount);
    const materialIndices = new Array<number>(cellCount);
    const flags = new Array<number>(triangleCount).fill(0);

    const minHeight = data.globalMinimumHeight;
    const maxHeight = data.globalMaximumHeight;

    const height = maxf(f32(maxHeight - minHeight), LINEAR_SLOP);
    const heightScale = f32(height / UINT16_MAX);
    const scale = data.scale;

    let lowerHeightBound = maxHeight;
    let upperHeightBound = minHeight;

    const invHeightScale = f32(1.0 / heightScale);
    for (let i = 0; i < heightCount; ++i) {
        const clampedHeight = clampf(data.heights[i], minHeight, maxHeight);
        const scaledHeight = f32(f32(clampedHeight - minHeight) * invHeightScale);
        // C: (uint16_t)b3MinFloat(scaledHeight, 65535). scaledHeight >= 0, so trunc == floor.
        compressedHeights[i] = Math.trunc(minf(scaledHeight, UINT16_MAX));

        lowerHeightBound = minf(lowerHeightBound, clampedHeight);
        upperHeightBound = maxf(upperHeightBound, clampedHeight);
    }

    // Decompressed heights give accurate convexity metrics.
    const heights = new Array<number>(heightCount);
    for (let i = 0; i < heightCount; ++i) {
        heights[i] = f32(minHeight + f32(heightScale * compressedHeights[i]));
    }

    if (data.materialIndices !== null) {
        for (let i = 0; i < cellCount; ++i) {
            materialIndices[i] = data.materialIndices[i];
        }
    } else {
        for (let i = 0; i < cellCount; ++i) {
            materialIndices[i] = 0;
        }
    }

    const box: AABB = {
        lowerBound: { x: 0.0, y: f32(scale.y * lowerHeightBound), z: 0.0 },
        upperBound: {
            x: f32(scale.x * (columnCount - 1)),
            y: f32(scale.y * upperHeightBound),
            z: f32(scale.z * (rowCount - 1)),
        },
    };

    const cos5Deg = f32(0.9962);

    let triangleIndex = 0;
    for (let row = 0; row < rowCount - 1; ++row) {
        for (let column = 0; column < columnCount - 1; ++column) {
            const triangleIndex1 = triangleIndex;
            const triangleIndex2 = triangleIndex + 1;
            triangleIndex += 2;

            const cellIndex = row * (columnCount - 1) + column;

            if (materialIndices[cellIndex] === HEIGHT_FIELD_HOLE) {
                continue;
            }

            let flags1 = 0;
            let flags2 = 0;

            const index11 = row * columnCount + column;
            const index12 = index11 + 1;
            const index21 = (row + 1) * columnCount + column;
            const index22 = index21 + 1;

            // Two triangles of the cell, plus the diagonal (edge2) convexity between them.
            const height11 = heights[index11];
            const height12 = heights[index12];
            const height21 = heights[index21];
            const height22 = heights[index22];

            const cx1 = f32(column);
            const cx2 = f32(column + 1);
            const cz1 = f32(row);
            const cz2 = f32(row + 1);

            // triangle 0 : 11, 21, 12
            const t0v0 = vec3.mul(scale, { x: cx1, y: height11, z: cz1 });
            const t0v1 = vec3.mul(scale, { x: cx1, y: height21, z: cz2 });
            const t0v2 = vec3.mul(scale, { x: cx2, y: height12, z: cz1 });
            const plane1 = plane.fromPoints(t0v0, t0v1, t0v2);

            // triangle 1 : 22, 12, 21
            const t1v0 = vec3.mul(scale, { x: cx2, y: height22, z: cz2 });
            const t1v1 = vec3.mul(scale, { x: cx2, y: height12, z: cz1 });
            const t1v2 = vec3.mul(scale, { x: cx1, y: height21, z: cz2 });
            const plane2 = plane.fromPoints(t1v0, t1v1, t1v2);

            {
                const separation = plane.separation(plane1, t1v0);
                const cosAngle = vec3.dot(plane1.normal, plane2.normal);
                if (separation > 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.ConcaveEdge2;
                    flags2 |= MeshEdgeFlags.ConcaveEdge2;
                }
                if (separation < 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.InverseConcaveEdge2;
                    flags2 |= MeshEdgeFlags.InverseConcaveEdge2;
                }
            }

            // top neighbour: edge3 of triangle 0
            const topCellIndex = (row - 1) * (columnCount - 1) + column;
            if (row > 0 && materialIndices[topCellIndex] !== HEIGHT_FIELD_HOLE) {
                const r = row - 1;
                const c = column;
                const i11 = r * columnCount + c;
                const i12 = i11 + 1;
                const i21 = (r + 1) * columnCount + c;
                const i22 = i21 + 1;

                const h12 = heights[i12];
                const h21 = heights[i21];
                const h22 = heights[i22];

                const x1 = f32(c);
                const x2 = f32(c + 1);
                const z1 = f32(r);
                const z2 = f32(r + 1);

                const vs0 = vec3.mul(scale, { x: x2, y: h22, z: z2 });
                const vs1 = vec3.mul(scale, { x: x2, y: h12, z: z1 });
                const vs2 = vec3.mul(scale, { x: x1, y: h21, z: z2 });
                const n = normalFromPoints(vs0, vs1, vs2);

                const separation = plane.separation(plane1, vs1);
                const cosAngle = vec3.dot(plane1.normal, n);
                if (separation > 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.ConcaveEdge3;
                }
                if (separation < 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.InverseConcaveEdge3;
                }
            }

            // bottom neighbour: edge3 of triangle 1
            const bottomCellIndex = (row + 1) * (columnCount - 1) + column;
            if (row + 1 < rowCount - 1 && materialIndices[bottomCellIndex] !== HEIGHT_FIELD_HOLE) {
                const r = row + 1;
                const c = column;
                const i11 = r * columnCount + c;
                const i12 = i11 + 1;
                const i21 = (r + 1) * columnCount + c;

                const h11 = heights[i11];
                const h12 = heights[i12];
                const h21 = heights[i21];

                const x1 = f32(c);
                const x2 = f32(c + 1);
                const z1 = f32(r);
                const z2 = f32(r + 1);

                const vs0 = vec3.mul(scale, { x: x1, y: h11, z: z1 });
                const vs1 = vec3.mul(scale, { x: x1, y: h21, z: z2 });
                const vs2 = vec3.mul(scale, { x: x2, y: h12, z: z1 });
                const n = normalFromPoints(vs0, vs1, vs2);

                const separation = plane.separation(plane2, vs1);
                const cosAngle = vec3.dot(plane2.normal, n);
                if (separation > 0.0 || cosAngle > cos5Deg) {
                    flags2 |= MeshEdgeFlags.ConcaveEdge3;
                }
                if (separation < 0.0 || cosAngle > cos5Deg) {
                    flags2 |= MeshEdgeFlags.InverseConcaveEdge3;
                }
            }

            // left neighbour: edge1 of triangle 0
            const leftCellIndex = row * (columnCount - 1) + column - 1;
            if (column - 1 >= 0 && materialIndices[leftCellIndex] !== HEIGHT_FIELD_HOLE) {
                const r = row;
                const c = column - 1;
                const i11 = r * columnCount + c;
                const i12 = i11 + 1;
                const i21 = (r + 1) * columnCount + c;
                const i22 = i21 + 1;

                const h12 = heights[i12];
                const h21 = heights[i21];
                const h22 = heights[i22];

                const x1 = f32(c);
                const x2 = f32(c + 1);
                const z1 = f32(r);
                const z2 = f32(r + 1);

                const vs0 = vec3.mul(scale, { x: x2, y: h22, z: z2 });
                const vs1 = vec3.mul(scale, { x: x2, y: h12, z: z1 });
                const vs2 = vec3.mul(scale, { x: x1, y: h21, z: z2 });
                const n = normalFromPoints(vs0, vs1, vs2);

                const separation = plane.separation(plane1, vs2);
                const cosAngle = vec3.dot(plane1.normal, n);
                if (separation > 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.ConcaveEdge1;
                }
                if (separation < 0.0 || cosAngle > cos5Deg) {
                    flags1 |= MeshEdgeFlags.InverseConcaveEdge1;
                }
            }

            // right neighbour: edge1 of triangle 1
            const rightCellIndex = row * (columnCount - 1) + column + 1;
            if (
                column + 1 < columnCount - 1 &&
                materialIndices[rightCellIndex] !== HEIGHT_FIELD_HOLE
            ) {
                const r = row;
                const c = column + 1;
                const i11 = r * columnCount + c;
                const i12 = i11 + 1;
                const i21 = (r + 1) * columnCount + c;

                const h11 = heights[i11];
                const h12 = heights[i12];
                const h21 = heights[i21];

                const x1 = f32(c);
                const x2 = f32(c + 1);
                const z1 = f32(r);
                const z2 = f32(r + 1);

                const vs0 = vec3.mul(scale, { x: x1, y: h11, z: z1 });
                const vs1 = vec3.mul(scale, { x: x1, y: h21, z: z2 });
                const vs2 = vec3.mul(scale, { x: x2, y: h12, z: z1 });
                const n = normalFromPoints(vs0, vs1, vs2);

                const separation = plane.separation(plane2, vs2);
                const cosAngle = vec3.dot(plane2.normal, n);
                if (separation > 0.0 || cosAngle > cos5Deg) {
                    flags2 |= MeshEdgeFlags.ConcaveEdge1;
                }
                if (separation < 0.0 || cosAngle > cos5Deg) {
                    flags2 |= MeshEdgeFlags.InverseConcaveEdge1;
                }
            }

            flags[triangleIndex1] = flags1;
            flags[triangleIndex2] = flags2;
        }
    }

    return {
        aabb: box,
        minHeight,
        maxHeight,
        heightScale,
        scale,
        columnCount,
        rowCount,
        compressedHeights,
        materialIndices,
        flags,
        clockwise: data.clockwiseWinding,
    };
}

// b3Normalize(b3Cross(p2-p1, p3-p1)) — the adjacency-neighbour normal (no offset needed).
const normalFromPoints = (p1: Vec3, p2: Vec3, p3: Vec3): Vec3 =>
    vec3.normalize(vec3.cross(vec3.sub(p2, p1), vec3.sub(p3, p1)));

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

/** AABB of a height field under a transform (b3ComputeHeightFieldAABB = b3AABB_Transform of the box). */
export function computeHeightFieldAABB(hf: HeightFieldData, transform: Transform): AABB {
    return aabb.transform(transform, hf.aabb);
}

/**
 * Ray vs height field (b3RayCastHeightField). A ray is a degenerate shape cast — a single-point,
 * zero-radius proxy swept along the translation — so this defers to {@link shapeCastHeightField}.
 * Point/normal are in the field's local frame (the caller lifts them to world).
 */
export function rayCastHeightField(hf: HeightFieldData, input: RayCastInput): CastOutput {
    return shapeCastHeightField(hf, {
        proxy: { points: [input.origin], count: 1, radius: 0 },
        translation: input.translation,
        maxFraction: input.maxFraction,
        canEncroach: false,
    });
}

/**
 * Shape cast (or ray cast, when the proxy is a single zero-radius point) vs height field
 * (b3ShapeCastHeightField). Rasterizes the swept shape bounds across the grid cells with a DDA walk,
 * testing the two triangles of each visited cell; returns the nearest hit. Point/normal are local.
 */
export function shapeCastHeightField(hf: HeightFieldData, input: ShapeCastInput): CastOutput {
    const shapeBounds = aabb.make(input.proxy.points, input.proxy.count, input.proxy.radius);
    const shapeTranslation = input.translation;
    const scale = hf.scale;

    const shapeStart = aabb.center(shapeBounds);
    const shapeDelta = vec3.scale(input.maxFraction, shapeTranslation);
    const shapeEnd = vec3.add(shapeStart, shapeDelta);

    let result = emptyCastOutput();

    const shapeExtents = aabb.extents(shapeBounds);
    const margin: Vec3 = { x: MAX_AABB_MARGIN, y: MAX_AABB_MARGIN, z: MAX_AABB_MARGIN };
    const combinedBounds: AABB = {
        lowerBound: vec3.sub(vec3.sub(hf.aabb.lowerBound, shapeExtents), margin),
        upperBound: vec3.add(vec3.add(hf.aabb.upperBound, shapeExtents), margin),
    };

    const hit = rayCastAABB(combinedBounds, shapeStart, shapeEnd);
    if (hit.hit === false) return result;
    const minFraction = hit.minFraction;
    const maxFraction = hit.maxFraction;

    // These drive the grid DDA; the triangle cast uses the unclamped ray + fraction.
    const clampedStart = vec3.mulAdd(shapeStart, minFraction, shapeDelta);
    const clampedDelta = vec3.scale(f32(maxFraction - minFraction), shapeDelta);

    // The center sweep must stay on the shape path; clampedStart gets pushed to the leading corner.
    const centerStart = clampedStart;
    const centerEnd = vec3.add(clampedStart, clampedDelta);

    // Push the grid-walk start out to the leading shape-bounds corner.
    let startX = clampedStart.x;
    let startZ = clampedStart.z;
    let signX: number;
    let signZ: number;
    if (shapeTranslation.x >= 0) {
        startX = f32(startX + shapeExtents.x);
        signX = 1;
    } else {
        startX = f32(startX - shapeExtents.x);
        signX = -1;
    }
    if (shapeTranslation.z >= 0) {
        startZ = f32(startZ + shapeExtents.z);
        signZ = 1;
    } else {
        startZ = f32(startZ - shapeExtents.z);
        signZ = -1;
    }
    const endX = f32(startX + clampedDelta.x);
    const endZ = f32(startZ + clampedDelta.z);

    const columnStart = Math.floor(f32(startX / scale.x));
    const columnEnd = Math.floor(f32(endX / scale.x));
    const rowStart = Math.floor(f32(startZ / scale.z));
    const rowEnd = Math.floor(f32(endZ / scale.z));

    const absClampedDelta = vec3.abs(clampedDelta);

    let deltaAlphaX: number;
    let nextFractionX: number;
    let deltaColumn: number;
    if (columnStart < columnEnd) {
        deltaAlphaX = f32(scale.x / absClampedDelta.x);
        nextFractionX = f32(f32(f32(scale.x * (columnStart + 1)) - startX) / absClampedDelta.x);
        deltaColumn = 1;
    } else if (columnEnd < columnStart) {
        deltaAlphaX = f32(scale.x / absClampedDelta.x);
        nextFractionX = f32(f32(startX - f32(scale.x * columnStart)) / absClampedDelta.x);
        deltaColumn = -1;
    } else {
        deltaAlphaX = 0;
        nextFractionX = FLT_MAX;
        deltaColumn = 0;
    }

    let deltaAlphaZ: number;
    let nextFractionZ: number;
    let deltaRow: number;
    if (rowStart < rowEnd) {
        deltaAlphaZ = f32(scale.z / absClampedDelta.z);
        nextFractionZ = f32(f32(f32(scale.z * (rowStart + 1)) - startZ) / absClampedDelta.z);
        deltaRow = 1;
    } else if (rowEnd < rowStart) {
        deltaAlphaZ = f32(scale.z / absClampedDelta.z);
        nextFractionZ = f32(f32(startZ - f32(scale.z * rowStart)) / absClampedDelta.z);
        deltaRow = -1;
    } else {
        deltaAlphaZ = 0;
        nextFractionZ = FLT_MAX;
        deltaRow = 0;
    }

    let boxColumnHead = columnStart;
    let boxRowHead = rowStart;
    let boxColumnTail = Math.floor(
        f32(f32(startX - f32(f32(2 * signX) * shapeExtents.x)) / scale.x),
    );
    let boxRowTail = Math.floor(f32(f32(startZ - f32(f32(2 * signZ) * shapeExtents.z)) / scale.z));

    let bestFraction = input.maxFraction;

    // nextFraction* advance in clamped-sweep space; map into input-translation space before comparing
    // against bestFraction, else the loop can exit early and miss a nearer hit in a later cell.
    const gridFractionScale = f32(input.maxFraction * f32(maxFraction - minFraction));
    const gridFractionOffset = f32(input.maxFraction * minFraction);

    const rowCount = hf.rowCount;
    const columnCount = hf.columnCount;

    const pairInput: ShapeCastPairInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: input.proxy,
        transform: xf.identity(),
        translationB: input.translation,
        maxFraction: bestFraction,
        canEncroach: input.canEncroach,
    };

    const castBounds: AABB = {
        lowerBound: vec3.sub(vec3.min(centerStart, centerEnd), shapeExtents),
        upperBound: vec3.add(vec3.max(centerStart, centerEnd), shapeExtents),
    };

    const isRay = input.proxy.count === 1 && input.proxy.radius === 0;

    while (true) {
        const column1 = boxColumnTail < boxColumnHead ? boxColumnTail : boxColumnHead;
        const column2 = boxColumnTail < boxColumnHead ? boxColumnHead : boxColumnTail;
        const row1 = boxRowTail < boxRowHead ? boxRowTail : boxRowHead;
        const row2 = boxRowTail < boxRowHead ? boxRowHead : boxRowTail;

        for (let row = row1; row <= row2; ++row) {
            if (row < 0 || rowCount - 1 <= row) continue;
            for (let column = column1; column <= column2; ++column) {
                if (column < 0 || columnCount - 1 <= column) continue;

                const cellIndex = row * (columnCount - 1) + column;
                const materialIndex = hf.materialIndices[cellIndex];
                if (materialIndex === HEIGHT_FIELD_HOLE) continue;

                const corners = getCellCorners(hf, row, column);
                const point11 = corners[0];
                const point12 = corners[1];
                const point21 = corners[2];
                const point22 = corners[3];

                const cellBounds: AABB = {
                    lowerBound: vec3.min(vec3.min(point11, point12), vec3.min(point21, point22)),
                    upperBound: vec3.max(vec3.max(point11, point12), vec3.max(point21, point22)),
                };
                if (aabb.overlaps(castBounds, cellBounds) === false) continue;

                const quadIndex = row * (columnCount - 1) + column;
                const triangleIndex1 = 2 * quadIndex;
                const triangleIndex2 = triangleIndex1 + 1;

                if (isRay) {
                    {
                        const v1 = point11;
                        const v2 = hf.clockwise ? point12 : point21;
                        const v3 = hf.clockwise ? point21 : point12;
                        const alpha = intersectRayTriangle(
                            shapeStart,
                            shapeTranslation,
                            v1,
                            v2,
                            v3,
                        );
                        if (alpha < bestFraction) {
                            const edge1 = vec3.sub(point21, point11);
                            const edge2 = vec3.sub(point12, point11);
                            const normal = hf.clockwise
                                ? vec3.cross(edge2, edge1)
                                : vec3.cross(edge1, edge2);
                            result.point = vec3.mulAdd(shapeStart, alpha, shapeTranslation);
                            result.normal = vec3.normalize(normal);
                            result.fraction = alpha;
                            result.triangleIndex = triangleIndex1;
                            result.materialIndex = materialIndex;
                            result.hit = true;
                            bestFraction = alpha;
                        }
                    }
                    {
                        const v1 = point22;
                        const v2 = hf.clockwise ? point21 : point12;
                        const v3 = hf.clockwise ? point12 : point21;
                        const alpha = intersectRayTriangle(
                            shapeStart,
                            shapeTranslation,
                            v1,
                            v2,
                            v3,
                        );
                        if (alpha < bestFraction) {
                            const edge1 = vec3.sub(point22, point21);
                            const edge2 = vec3.sub(point12, point21);
                            const normal = hf.clockwise
                                ? vec3.cross(edge2, edge1)
                                : vec3.cross(edge1, edge2);
                            result.point = vec3.mulAdd(shapeStart, alpha, shapeTranslation);
                            result.normal = vec3.normalize(normal);
                            result.fraction = alpha;
                            result.triangleIndex = triangleIndex2;
                            result.materialIndex = materialIndex;
                            result.hit = true;
                            bestFraction = alpha;
                        }
                    }
                } else {
                    {
                        const origin = point11;
                        pairInput.proxyA = {
                            points: [
                                vec3.zero(),
                                vec3.sub(point21, origin),
                                vec3.sub(point12, origin),
                            ],
                            count: 3,
                            radius: 0,
                        };
                        pairInput.maxFraction = bestFraction;
                        pairInput.transform = { p: vec3.neg(origin), q: quat.identity() };
                        const pairOutput = shapeCast(pairInput);
                        if (pairOutput.hit) {
                            bestFraction = pairOutput.fraction;
                            result = pairOutput;
                            result.point = vec3.add(result.point, origin);
                            result.triangleIndex = triangleIndex1;
                            result.materialIndex = materialIndex;
                        }
                    }
                    {
                        const origin = point21;
                        pairInput.proxyA = {
                            points: [
                                vec3.zero(),
                                vec3.sub(point22, origin),
                                vec3.sub(point12, origin),
                            ],
                            count: 3,
                            radius: 0,
                        };
                        pairInput.maxFraction = bestFraction;
                        pairInput.transform = { p: vec3.neg(origin), q: quat.identity() };
                        const pairOutput = shapeCast(pairInput);
                        if (pairOutput.hit) {
                            bestFraction = pairOutput.fraction;
                            result = pairOutput;
                            result.point = vec3.add(result.point, origin);
                            result.triangleIndex = triangleIndex2;
                            result.materialIndex = materialIndex;
                        }
                    }
                }
            }
        }

        // Advance the grid walk. Map the next cell-crossing fractions into input space first.
        const inputFractionX =
            nextFractionX === FLT_MAX
                ? FLT_MAX
                : f32(gridFractionOffset + f32(nextFractionX * gridFractionScale));
        const inputFractionZ =
            nextFractionZ === FLT_MAX
                ? FLT_MAX
                : f32(gridFractionOffset + f32(nextFractionZ * gridFractionScale));
        if (inputFractionX > bestFraction && inputFractionZ > bestFraction) break;

        if (nextFractionX <= nextFractionZ) {
            if (boxColumnHead === columnEnd) break;
            boxColumnHead += deltaColumn;
            boxColumnTail = boxColumnHead;
            if (shapeExtents.z === 0) {
                boxRowTail = boxRowHead;
            } else {
                const rowIntercept = f32(startZ + f32(nextFractionX * clampedDelta.z));
                boxRowTail = Math.floor(
                    f32(f32(rowIntercept - f32(f32(2 * signZ) * shapeExtents.z)) / scale.z),
                );
            }
            nextFractionX = f32(nextFractionX + deltaAlphaX);
        } else {
            if (boxRowHead === rowEnd) break;
            boxRowHead += deltaRow;
            boxRowTail = boxRowHead;
            if (shapeExtents.x === 0) {
                boxColumnTail = boxColumnHead;
            } else {
                const columnIntercept = f32(startX + f32(nextFractionZ * clampedDelta.x));
                boxColumnTail = Math.floor(
                    f32(f32(columnIntercept - f32(f32(2 * signX) * shapeExtents.x)) / scale.x),
                );
            }
            nextFractionZ = f32(nextFractionZ + deltaAlphaZ);
        }
    }

    return result;
}

/**
 * True if `proxy` (in world space, with the field at `transform`) overlaps the height field
 * (b3OverlapHeightField). Pulls the proxy into local space, tests the two triangles of every cell its
 * bounds cover with a SAT pre-cull then GJK, and reports a hit within the overlap slop.
 */
export function overlapHeightField(
    hf: HeightFieldData,
    transform: Transform,
    proxy: ShapeProxy,
): boolean {
    const localProxy = makeLocalProxy(proxy, transform);
    const box = computeProxyAABB(localProxy);
    const scale = hf.scale;

    const minRow = Math.floor(f32(box.lowerBound.z / scale.z));
    const maxRow = Math.floor(f32(box.upperBound.z / scale.z));
    const minCol = Math.floor(f32(box.lowerBound.x / scale.x));
    const maxCol = Math.floor(f32(box.upperBound.x / scale.x));

    const boundsCenter = vec3.scale(0.5, vec3.add(box.lowerBound, box.upperBound));
    const boundsExtent = vec3.sub(box.upperBound, boundsCenter);

    const input: DistanceInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: localProxy,
        transform: xf.identity(),
        useRadii: true,
    };
    const cache = emptyCache();

    for (let row = minRow; row <= maxRow; ++row) {
        if (row < 0 || hf.rowCount - 1 <= row) continue;
        for (let column = minCol; column <= maxCol; ++column) {
            if (column < 0 || hf.columnCount - 1 <= column) continue;

            const cellIndex = row * (hf.columnCount - 1) + column;
            if (hf.materialIndices[cellIndex] === HEIGHT_FIELD_HOLE) continue;

            const corners = getCellCorners(hf, row, column);
            const point11 = corners[0];
            const point12 = corners[1];
            const point21 = corners[2];
            const point22 = corners[3];

            if (testBoundsTriangleOverlap(boundsCenter, boundsExtent, point11, point21, point12)) {
                input.proxyA = { points: [point11, point21, point12], count: 3, radius: 0 };
                cache.count = 0;
                if (shapeDistance(input, cache).distance < OVERLAP_SLOP) return true;
            }
            if (testBoundsTriangleOverlap(boundsCenter, boundsExtent, point21, point22, point12)) {
                input.proxyA = { points: [point22, point12, point21], count: 3, radius: 0 };
                cache.count = 0;
                if (shapeDistance(input, cache).distance < OVERLAP_SLOP) return true;
            }
        }
    }

    return false;
}

/**
 * Collision planes between a capsule mover and a height field (b3CollideMoverAndHeightField), in the
 * field's frame. Walks the grid cells the mover's bounds cover in ascending triangle order, tests the
 * two triangles of each against the mover's core segment, and emits one plane per triangle within
 * reach. Deep overlap is dropped (no SAT for movers). Stops at `capacity` planes.
 */
export function collideMoverAndHeightField(
    shape: HeightFieldData,
    capacity: number,
    mover: Capsule,
): PlaneResult[] {
    const planes: PlaneResult[] = [];

    const input: DistanceInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: { points: [mover.center1, mover.center2], count: 2, radius: 0 },
        transform: xf.identity(),
        useRadii: false,
    };
    const cache = emptyCache();

    const r = { x: mover.radius, y: mover.radius, z: mover.radius };
    const boundsMin = vec3.sub(vec3.min(mover.center1, mover.center2), r);
    const boundsMax = vec3.add(vec3.max(mover.center1, mover.center2), r);
    const boundsCenter = vec3.scale(0.5, vec3.add(boundsMin, boundsMax));
    const boundsExtent = vec3.sub(boundsMax, boundsCenter);

    const scale = shape.scale;
    const minRow = Math.floor(f32(boundsMin.z / scale.z));
    const maxRow = Math.floor(f32(boundsMax.z / scale.z));
    const minCol = Math.floor(f32(boundsMin.x / scale.x));
    const maxCol = Math.floor(f32(boundsMax.x / scale.x));

    // Outer loop rows, inner loop columns so triangle indices increase monotonically.
    for (let row = minRow; row <= maxRow; ++row) {
        if (row < 0 || shape.rowCount - 1 <= row) continue;
        for (let column = minCol; column <= maxCol; ++column) {
            if (column < 0 || shape.columnCount - 1 <= column) continue;

            const cellIndex = row * (shape.columnCount - 1) + column;
            if (shape.materialIndices[cellIndex] === HEIGHT_FIELD_HOLE) continue;

            const corners = getCellCorners(shape, row, column);
            const point11 = corners[0];
            const point12 = corners[1];
            const point21 = corners[2];
            const point22 = corners[3];

            if (testBoundsTriangleOverlap(boundsCenter, boundsExtent, point11, point21, point12)) {
                input.proxyA = { points: [point11, point21, point12], count: 3, radius: 0 };
                cache.count = 0;
                const output = shapeDistance(input, cache);
                if (output.distance !== 0 && output.distance <= mover.radius) {
                    const pl: Plane = {
                        normal: output.normal,
                        offset: f32(mover.radius - output.distance),
                    };
                    planes.push({ plane: pl, point: output.pointA });
                    if (planes.length === capacity) return planes;
                }
            }

            if (testBoundsTriangleOverlap(boundsCenter, boundsExtent, point21, point22, point12)) {
                input.proxyA = { points: [point22, point12, point21], count: 3, radius: 0 };
                cache.count = 0;
                const output = shapeDistance(input, cache);
                if (output.distance !== 0 && output.distance <= mover.radius) {
                    const pl: Plane = {
                        normal: output.normal,
                        offset: f32(mover.radius - output.distance),
                    };
                    planes.push({ plane: pl, point: output.pointA });
                    if (planes.length === capacity) return planes;
                }
            }
        }
    }

    return planes;
}

/** Callback for {@link queryHeightField}: triangle vertices + triangle index. */
export type HeightFieldQueryFcn = (a: Vec3, b: Vec3, c: Vec3, triangleIndex: number) => void;

/**
 * Visit every height-field triangle whose cell AABB overlaps `bounds` (in the field's local frame), in
 * ascending triangle-index order (b3QueryHeightField). The two triangles of an overlapping cell are
 * emitted with collision-facing winding (flipped when the field is clockwise).
 */
export function queryHeightField(
    hf: HeightFieldData,
    bounds: AABB,
    fcn: HeightFieldQueryFcn,
): void {
    const scale = hf.scale;

    const minRow = Math.floor(f32(bounds.lowerBound.z / scale.z));
    const maxRow = Math.floor(f32(bounds.upperBound.z / scale.z));
    const minCol = Math.floor(f32(bounds.lowerBound.x / scale.x));
    const maxCol = Math.floor(f32(bounds.upperBound.x / scale.x));

    // Outer loop rows, inner loop columns so triangle indices increase monotonically.
    for (let row = minRow; row <= maxRow; ++row) {
        if (row < 0 || hf.rowCount - 1 <= row) {
            continue;
        }

        for (let column = minCol; column <= maxCol; ++column) {
            if (column < 0 || hf.columnCount - 1 <= column) {
                continue;
            }

            const cellIndex = row * (hf.columnCount - 1) + column;
            const material = hf.materialIndices[cellIndex];
            if (material === HEIGHT_FIELD_HOLE) {
                continue;
            }

            const corners = getCellCorners(hf, row, column);
            const point11 = corners[0];
            const point12 = corners[1];
            const point21 = corners[2];
            const point22 = corners[3];

            const cellBound: AABB = {
                lowerBound: vec3.min(vec3.min(point11, point12), vec3.min(point21, point22)),
                upperBound: vec3.max(vec3.max(point11, point12), vec3.max(point21, point22)),
            };

            if (aabb.overlaps(bounds, cellBound)) {
                const quadIndex = row * (hf.columnCount - 1) + column;
                const triangleIndex = 2 * quadIndex;

                if (hf.clockwise) {
                    fcn(point11, point12, point21, triangleIndex);
                    fcn(point22, point21, point12, triangleIndex + 1);
                } else {
                    fcn(point11, point21, point12, triangleIndex);
                    fcn(point22, point12, point21, triangleIndex + 1);
                }
            }
        }
    }
}

/**
 * A flat grid height field (b3CreateGrid): all heights zero, quantized to the global [-256, 256] range.
 * `makeHoles` punches a hole every 16th cell.
 */
export function createGrid(
    rowCount: number,
    columnCount: number,
    scale: Vec3,
    makeHoles: boolean,
): HeightFieldData {
    const heightCount = rowCount * columnCount;
    const heights = new Array<number>(heightCount).fill(0.0);

    const cellCount = (rowCount - 1) * (columnCount - 1);
    const materialIndices = new Array<number>(cellCount);
    for (let i = 0; i < rowCount - 1; ++i) {
        for (let j = 0; j < columnCount - 1; ++j) {
            const k = i * (columnCount - 1) + j;
            materialIndices[k] = makeHoles && k > 0 && k % 16 === 0 ? HEIGHT_FIELD_HOLE : 0;
        }
    }

    return createHeightField({
        heights,
        materialIndices,
        scale,
        countX: columnCount,
        countZ: rowCount,
        globalMinimumHeight: -256.0,
        globalMaximumHeight: 256.0,
        clockwiseWinding: false,
    });
}

/**
 * A sinusoidal wave height field (b3CreateWave): height = sin(omegaZ*row) * sin(omegaX*col). Uses JS
 * `Math.sin`, so it is NOT part of the bit-exact sim path — for authoring/visual scenes only.
 */
export function createWave(
    rowCount: number,
    columnCount: number,
    scale: Vec3,
    rowFrequency: number,
    columnFrequency: number,
    makeHoles: boolean,
): HeightFieldData {
    const heightCount = rowCount * columnCount;
    const heights = new Array<number>(heightCount);

    const omegaZ = f32(2.0 * Math.PI * rowFrequency);
    const omegaX = f32(2.0 * Math.PI * columnFrequency);

    for (let i = 0; i < rowCount; ++i) {
        const rowHeight = f32(Math.sin(f32(omegaZ * i)));
        for (let j = 0; j < columnCount; ++j) {
            const k = i * columnCount + j;
            const columnHeight = f32(Math.sin(f32(omegaX * j)));
            heights[k] = f32(rowHeight * columnHeight);
        }
    }

    const cellCount = (rowCount - 1) * (columnCount - 1);
    const materialIndices = new Array<number>(cellCount);
    for (let i = 0; i < rowCount - 1; ++i) {
        for (let j = 0; j < columnCount - 1; ++j) {
            const k = i * (columnCount - 1) + j;
            materialIndices[k] = makeHoles && k > 0 && k % 16 === 0 ? HEIGHT_FIELD_HOLE : 0;
        }
    }

    return createHeightField({
        heights,
        materialIndices,
        scale,
        countX: columnCount,
        countZ: rowCount,
        globalMinimumHeight: -256.0,
        globalMaximumHeight: 256.0,
        clockwiseWinding: false,
    });
}
