import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import type { GltfMaterial } from "./gltf";
import { MATERIAL_STRIDE, MaterialData, materialDataWgsl, packMaterials } from "./palette";

// the CPU↔GPU palette boundary: `packMaterials` writes the MaterialData struct byte layout the surfaces read,
// so the offsets are pinned here (a wrong offset reads e.g. roughness from the metallic slot — plausible on
// screen, wrong everywhere). The struct + packer share `palette.ts` so this is their one source of truth.

function material(over: Partial<GltfMaterial>): GltfMaterial {
    return {
        color: [1, 1, 1, 1],
        metallic: 1,
        roughness: 1,
        normalScale: 1,
        occStrength: 1,
        emissive: [0, 0, 0],
        cutoff: 0.5,
        alphaMode: "OPAQUE",
        ...over,
    };
}

const layers = (
    albedo: number,
    mr: number,
    normal: number,
    occ: number,
    emis: number,
    bucket = 0,
) => ({
    albedo: Int32Array.of(albedo),
    mr: Int32Array.of(mr),
    normal: Int32Array.of(normal),
    occ: Int32Array.of(occ),
    emis: Int32Array.of(emis),
    albedoBucket: Uint32Array.of(bucket),
});

// The schema is now the layout's one source of truth, so its byte offsets are pinned against the numbers
// the hand-written `MaterialData` WGSL struct had — literals here, never re-derived from the schema (that
// would test a constant against itself). A shipped shader reads these offsets and `packMaterials` writes
// them, so a reorder or an alignment change has to fail here.
describe("MaterialData layout", () => {
    const offset = (field: (m: d.Infer<typeof MaterialData>) => unknown) =>
        d.memoryLayoutOf(MaterialData, field).offset;

    test("four vec4 rows, 64 bytes, each field at the offset the WGSL struct had", () => {
        expect(d.sizeOf(MaterialData)).toBe(64);
        expect(MATERIAL_STRIDE).toBe(64);
        // row 0: layer / cutoff / metallic / roughness
        expect(offset((m) => m.layer)).toBe(0);
        expect(offset((m) => m.cutoff)).toBe(4);
        expect(offset((m) => m.metallic)).toBe(8);
        expect(offset((m) => m.roughness)).toBe(12);
        // row 1: the four data-slot layers
        expect(offset((m) => m.mrLayer)).toBe(16);
        expect(offset((m) => m.normalLayer)).toBe(20);
        expect(offset((m) => m.occLayer)).toBe(24);
        expect(offset((m) => m.emisLayer)).toBe(28);
        // row 2: emissive is a vec3f, so its 16-byte alignment is what keeps the row packed at 32
        expect(offset((m) => m.emissive)).toBe(32);
        expect(offset((m) => m.normalScale)).toBe(44);
        // row 3: occStrength / albedoBucket + the two pad words
        expect(offset((m) => m.occStrength)).toBe(48);
        expect(offset((m) => m.albedoBucket)).toBe(52);
        expect(offset((m) => m.pad1)).toBe(56);
        expect(offset((m) => m.pad2)).toBe(60);
    });

    test("the chunk emits the struct under its authored name, with the mixed lane types intact", () => {
        const wgsl = materialDataWgsl();
        expect(wgsl).toContain("struct MaterialData {");
        // the `-1` slot sentinels are signed and the bucket is unsigned — a lane typed f32 would read
        // -1 as garbage in the shader's `layer < 0` guard / `switch albedoBucket`
        expect(wgsl).toContain("layer: i32,");
        expect(wgsl).toContain("albedoBucket: u32,");
        expect(wgsl).toContain("emissive: vec3f,");
    });
});

describe("packMaterials", () => {
    test("writes each field to its MaterialData struct offset, 64 bytes/material", () => {
        const m = material({
            cutoff: 0.25,
            metallic: 0.2,
            roughness: 0.7,
            normalScale: 0.8,
            occStrength: 0.5,
            emissive: [1, 2, 3],
        });
        const buf = packMaterials([m], layers(4, 5, 6, 7, 8, 2));
        expect(buf.byteLength).toBe(MATERIAL_STRIDE);
        const u = new Uint32Array(buf);
        const i = new Int32Array(buf);
        const f = new Float32Array(buf);
        expect(u[0]).toBe(4); // layer
        expect(u[13]).toBe(2); // albedoBucket (the size-bucket sampleAlbedo switches on)
        expect(f[1]).toBe(0.25); // cutoff (power-of-2 fraction, exact)
        expect(f[2]).toBeCloseTo(0.2, 6); // metallic
        expect(f[3]).toBeCloseTo(0.7, 6); // roughness
        expect(i[4]).toBe(5); // mrLayer
        expect(i[5]).toBe(6); // normalLayer
        expect(i[6]).toBe(7); // occLayer
        expect(i[7]).toBe(8); // emisLayer
        expect(f[8]).toBe(1); // emissive.x
        expect(f[9]).toBe(2); // emissive.y
        expect(f[10]).toBe(3); // emissive.z
        expect(f[11]).toBeCloseTo(0.8, 6); // normalScale
        expect(f[12]).toBeCloseTo(0.5, 6); // occStrength
    });

    test("a factor-only material keeps its -1 albedo layer, like the data slots", () => {
        const buf = packMaterials([material({})], layers(-1, -1, -1, -1, -1));
        expect(new Int32Array(buf)[0]).toBe(-1); // albedo -1 preserved — sampleAlbedo returns white for it (skin path)
        expect(new Int32Array(buf)[4]).toBe(-1); // mrLayer stays -1
        expect(new Int32Array(buf)[7]).toBe(-1); // emisLayer stays -1
    });

    test("strides 64 bytes/material — the second material's fields don't alias the first", () => {
        const buf = packMaterials([material({ metallic: 0.1 }), material({ metallic: 0.9 })], {
            albedo: Int32Array.of(0, 1),
            mr: Int32Array.of(-1, -1),
            normal: Int32Array.of(-1, -1),
            occ: Int32Array.of(-1, -1),
            emis: Int32Array.of(-1, -1),
            albedoBucket: Uint32Array.of(0, 3),
        });
        expect(buf.byteLength).toBe(2 * MATERIAL_STRIDE);
        const f = new Float32Array(buf);
        expect(f[2]).toBeCloseTo(0.1, 6); // material 0 metallic
        expect(f[16 + 2]).toBeCloseTo(0.9, 6); // material 1 metallic, +16 floats (64 B)
        expect(new Uint32Array(buf)[16]).toBe(1); // material 1 albedo layer
        expect(new Uint32Array(buf)[16 + 13]).toBe(3); // material 1 albedoBucket, +16 floats
    });
});
