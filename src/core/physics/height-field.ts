import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";

/**
 * Box3D's `b3HeightFieldDef`: a grid of `countX` by `countZ` points in the shape's xz plane, one unit
 * apart before `scale`, starting at the shape origin. A solver shares a build per definition and body
 * Transform.scale in each World, quantizing
 * each height to 16 bits between the global bounds. It refuses a count below 2, a `heights` length
 * other than `countX * countZ`, a `materialIndices` length other than the cell count, a
 * non-positive `scale`, and `globalMinimumHeight` above `globalMaximumHeight`. `Shape.scale` does
 * not apply; the body's Transform.scale multiplies the definition scale and must be positive and
 * finite. Body-scale changes recreate the shape at the next fixed sync. Replace a height field
 * by adding a new definition and assigning its handle, never by mutation.
 */
export interface HeightField {
    name: string;
    /** Height of each grid point before `scale`, row by row: point (x, z) is `heights[z * countX + x]`. */
    heights: Float32Array;
    /** One index per cell, `(countX - 1) * (countZ - 1)` row by row, into the Shape's `ShapeMaterials` set; 255 makes the cell a hole. Default all 0. */
    materialIndices?: Uint8Array;
    /** Meters per grid unit along x, per height unit along y and per grid unit along z; each positive. */
    scale: [number, number, number];
    /** Grid points along x, at least 2. */
    countX: number;
    /** Grid points along z, at least 2. */
    countZ: number;
    /**
     * Lowest height the field quantizes to, before `scale`; lower heights clamp to it. Fields that
     * share both bounds line up exactly at their seams.
     */
    globalMinimumHeight: number;
    /** Highest height the field quantizes to, before `scale`; higher heights clamp to it. */
    globalMaximumHeight: number;
    /** Whether the field winds clockwise, inverting it along y; default false. */
    clockwiseWinding?: boolean;
}

/** World-owned height fields. `add` returns a handle for `Shape.geometry`; names are display labels. */
export const HeightFields: Resource<PhysicsAssets<HeightField>> = {
    create: () => new PhysicsAssets(),
};
