import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";

/**
 * Box3D's `b3HeightFieldDef`: a grid of `countX` by `countZ` points in the shape's xz plane, one unit
 * apart before `scale`, starting at the shape origin. A solver shares a build per definition and body
 * Transform.scale in each World, quantizing
 * each height to 16 bits between the global bounds. It refuses a count below 1, a `heights` length
 * other than `countX * countZ`, a `materialIndices` length other than the cell count, a
 * non-positive horizontal scale, negative vertical scale, and `globalMinimumHeight` above
 * `globalMaximumHeight`. An axis with one point has no cells; zero vertical scale makes a flat
 * surface. `Shape.scale` does
 * not apply; the body's Transform.scale multiplies the definition scale and must be positive and
 * finite. Body-scale changes recreate the shape at the next fixed sync. Replace a height field
 * by adding a new definition and assigning its handle, never by mutation.
 */
export type HeightField =
    | {
          name: string;
          /** Box3D's b3CreateWave; scale is meters per grid/height unit and frequencies are radians. */
          wave: {
              rowCount: number;
              columnCount: number;
              scale: [number, number, number];
              rowFrequency: number;
              columnFrequency: number;
              makeHoles: boolean;
          };
      }
    | HeightFieldGrid;

export interface HeightFieldGrid {
    name: string;
    /** Height of each grid point before `scale`, row by row: point (x, z) is `heights[z * countX + x]`. */
    heights: Float32Array;
    /** One index per cell, `(countX - 1) * (countZ - 1)` row by row, into the Shape's `ShapeMaterials` set; 255 makes the cell a hole. Default all 0. */
    materialIndices?: Uint8Array;
    /** Meters per grid unit along xz (positive) and per height unit along y (nonnegative). */
    scale: [number, number, number];
    /** Grid points along x, at least 1. */
    countX: number;
    /** Grid points along z, at least 1. */
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
