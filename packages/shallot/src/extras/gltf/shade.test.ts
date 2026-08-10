import { describe, expect, test } from "bun:test";
import type { GltfMaterial } from "./gltf";
import { MAP_ALL, MAP_EMIS, MAP_MR, MAP_NORMAL, MAP_OCC, mapSet } from "./shade";

// mapSet turns a decoded material into the map-set key (a mesh's `variant`). The bit↔field mapping is
// load-bearing: a wrong bit specializes a pipeline that then samples a map the material doesn't carry (or
// skips one it does), and the set bit must agree with the palette's `*Layer >= 0` (which keys off the same
// `*Image` field). So this pins each data-map field to its bit, and that baseColor (`image`) is not a bit.
describe("mapSet — material → specialization key", () => {
    const mat = (o: Partial<GltfMaterial>): GltfMaterial => ({
        color: [1, 1, 1, 1],
        metallic: 1,
        roughness: 1,
        emissive: [0, 0, 0],
        normalScale: 1,
        occStrength: 1,
        alphaMode: "OPAQUE",
        cutoff: 0.5,
        ...o,
    });

    test("no material / no maps is 0 (albedo-only)", () => {
        expect(mapSet(undefined)).toBe(0);
        expect(mapSet(mat({}))).toBe(0);
        // baseColor is always present on a textured surface, so it's not a map-set bit
        expect(mapSet(mat({ image: 0 }))).toBe(0);
    });

    test("each data-map field maps to its own bit", () => {
        expect(mapSet(mat({ normalImage: 0 }))).toBe(MAP_NORMAL);
        expect(mapSet(mat({ mrImage: 0 }))).toBe(MAP_MR);
        expect(mapSet(mat({ occImage: 0 }))).toBe(MAP_OCC);
        expect(mapSet(mat({ emissiveImage: 0 }))).toBe(MAP_EMIS);
    });

    test("all four maps present is MAP_ALL", () => {
        expect(mapSet(mat({ normalImage: 1, mrImage: 2, occImage: 3, emissiveImage: 4 }))).toBe(
            MAP_ALL,
        );
    });
});
