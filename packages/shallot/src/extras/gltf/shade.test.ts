import { describe, expect, test } from "bun:test";
import type { GltfMaterial } from "./gltf";
import { MAP_ALL, MAP_EMIS, MAP_MR, MAP_NORMAL, MAP_OCC, mapSet, materialPreamble } from "./shade";

// the per-map sample each bit gates — the marker its presence/absence is read by below
const SAMPLE: Record<number, string> = {
    [MAP_NORMAL]: "textureSample(normalTex",
    [MAP_MR]: "textureSample(mr,",
    [MAP_OCC]: "textureSample(occlusion",
    [MAP_EMIS]: "textureSample(emissive",
};
const BITS = [MAP_NORMAL, MAP_MR, MAP_OCC, MAP_EMIS];

describe("materialPreamble — map-set specialization", () => {
    test("every variant defines the four shade helpers + the albedo switch", () => {
        for (let mapset = 0; mapset <= MAP_ALL; mapset++) {
            const wgsl = materialPreamble(mapset);
            for (const fn of ["pbrOf", "emissiveOf", "perturbNormal", "shadePbr", "sampleAlbedo"]) {
                expect(wgsl).toContain(`fn ${fn}(`);
            }
            // albedo is unconditional (always present on a textured surface), via SAMPLE_ALBEDO
            expect(wgsl).toContain("textureSampleGrad(");
        }
    });

    test("a bit is set iff its data-map sample is emitted", () => {
        for (let mapset = 0; mapset <= MAP_ALL; mapset++) {
            const wgsl = materialPreamble(mapset);
            for (const bit of BITS) {
                const present = (mapset & bit) !== 0;
                expect(wgsl.includes(SAMPLE[bit])).toBe(present);
            }
        }
    });

    test("albedo-only (mapset 0) emits no data-map sample, no factor select, no tangent recon", () => {
        const wgsl = materialPreamble(0);
        for (const bit of BITS) expect(wgsl).not.toContain(SAMPLE[bit]);
        // homogeneous bins mean no runtime `*Layer >= 0` gate is needed — the absent path is the factor
        expect(wgsl).not.toContain("select(");
        // no normal map → no screen-space derivative tangent frame (dpdx(uv) is the albedo gradient)
        expect(wgsl).not.toContain("dpdx(world)");
        expect(wgsl).toContain("return N;");
    });

    test("the all-maps variant samples every data map and reconstructs the tangent frame", () => {
        const wgsl = materialPreamble(MAP_ALL);
        for (const bit of BITS) expect(wgsl).toContain(SAMPLE[bit]);
        expect(wgsl).toContain("dpdx(world)");
        // the bitmask guarantees presence, so a present sample indexes its layer directly (no max gate)
        expect(wgsl).not.toContain("max(md.mrLayer");
    });

    test("the normal-map bit alone reconstructs the frame without the other samples", () => {
        const wgsl = materialPreamble(MAP_NORMAL);
        expect(wgsl).toContain(SAMPLE[MAP_NORMAL]);
        expect(wgsl).toContain("dpdx(world)");
        for (const bit of [MAP_MR, MAP_OCC, MAP_EMIS]) expect(wgsl).not.toContain(SAMPLE[bit]);
    });
});

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
