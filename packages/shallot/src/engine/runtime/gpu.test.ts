import { describe, expect, test } from "bun:test";
import {
    checkStorageBinding,
    checkTextureLimits,
    deviceLimits,
    resolveFeatures,
    UnsupportedError,
} from "./gpu";

// a mobile-shaped adapter: the core WebGPU 1.0 limits are present, but the 2024 split-stage storage
// limits are absent (undefined), as reported by older mobile WebGPU implementations.
function limits(overrides: Record<string, number> = {}): GPUSupportedLimits {
    return {
        maxTextureDimension2D: 8192,
        maxTextureArrayLayers: 256,
        maxStorageBufferBindingSize: 134_217_728,
        maxBufferSize: 268_435_456,
        ...overrides,
    } as unknown as GPUSupportedLimits;
}

describe("deviceLimits", () => {
    test("drops absent split-stage limits instead of forwarding undefined as NaN", () => {
        const result = deviceLimits(limits());
        // forwarding an absent limit reaches requestDevice as NaN and rejects with
        // "Value NaN is outside the range [0, 9007199254740991]". Every requested value must
        // be a finite number.
        for (const value of Object.values(result)) {
            expect(Number.isFinite(value)).toBe(true);
        }
        expect("maxStorageBuffersInVertexStage" in result).toBe(false);
        expect("maxStorageTexturesInFragmentStage" in result).toBe(false);
        expect(result.maxStorageBufferBindingSize).toBe(134_217_728);
        expect(result.maxStorageBuffersPerShaderStage).toBe(10);
    });

    test("forwards split-stage limits when the adapter reports them", () => {
        const result = deviceLimits(
            limits({
                maxStorageBuffersInVertexStage: 8,
                maxStorageBuffersInFragmentStage: 10,
                maxStorageTexturesInVertexStage: 4,
                maxStorageTexturesInFragmentStage: 8,
            }),
        );
        expect(result.maxStorageBuffersInVertexStage).toBe(8);
        expect(result.maxStorageTexturesInFragmentStage).toBe(8);
    });
});

describe("resolveFeatures", () => {
    // an adapter exposing exactly `subgroups` + the base floor it cares about here.
    const adapter = (...present: GPUFeatureName[]) => new Set<GPUFeatureName>(present);

    test("a required feature the adapter has is granted nothing extra but never missing", () => {
        const { granted, missing } = resolveFeatures(adapter("shader-f16"), ["shader-f16"], []);
        expect(missing).toEqual([]);
        expect(granted).toEqual([]);
    });

    test("a required feature the adapter lacks is missing (the caller throws)", () => {
        const { missing } = resolveFeatures(adapter(), ["subgroups"], []);
        expect(missing).toEqual(["subgroups"]);
    });

    test("a preferred feature the adapter has is granted, and never missing", () => {
        const { granted, missing } = resolveFeatures(adapter("subgroups"), [], ["subgroups"]);
        expect(granted).toEqual(["subgroups"]);
        expect(missing).toEqual([]);
    });

    test("a preferred feature the adapter lacks is silently dropped — never missing", () => {
        // the whole point: a no-subgroup device (WebKit) loads a physics app, takes the LDS arm.
        const { granted, missing } = resolveFeatures(adapter(), [], ["subgroups"]);
        expect(granted).toEqual([]);
        expect(missing).toEqual([]);
    });

    test("a feature both required and preferred isn't double-counted into granted", () => {
        const { granted, missing } = resolveFeatures(
            adapter("subgroups"),
            ["subgroups"],
            ["subgroups"],
        );
        expect(missing).toEqual([]);
        expect(granted).toEqual([]); // already in required; the device-request set dedupes anyway
    });
});

const MB = 1 << 20;

describe("checkStorageBinding", () => {
    test("a buffer past the per-binding limit throws a named, loud + clear UnsupportedError", () => {
        let caught: unknown;
        try {
            checkStorageBinding("[bvh] the node buffer", 200 * MB, 128 * MB, "Lower maxPrims.");
        } catch (e) {
            caught = e;
        }
        // a named UnsupportedError, not a generic Error — the consumer's diagnostic boundary.
        expect(caught).toBeInstanceOf(UnsupportedError);
        const msg = (caught as Error).message;
        expect(msg).toContain("[bvh] the node buffer"); // names the buffer
        expect(msg).toContain("200 MB"); // needed
        expect(msg).toContain("128 MB"); // available
        expect(msg).toContain("maxStorageBufferBindingSize"); // the limit it tripped
        expect(msg).toContain("Lower maxPrims"); // the remedy
    });

    test("a buffer under the limit does not throw", () => {
        expect(() => checkStorageBinding("[bvh] x", 64 * MB, 128 * MB, "remedy")).not.toThrow();
    });

    test("the boundary is exclusive: exactly at the limit fits, one byte over throws", () => {
        expect(() => checkStorageBinding("x", 128 * MB, 128 * MB, "r")).not.toThrow();
        expect(() => checkStorageBinding("x", 128 * MB + 1, 128 * MB, "r")).toThrow(
            UnsupportedError,
        );
    });
});

describe("checkTextureLimits", () => {
    test("a width past maxTextureDimension2D throws a named UnsupportedError", () => {
        let caught: unknown;
        try {
            checkTextureLimits(
                "[gltf] a skinned mesh's VAT",
                { width: 9000, height: 4 },
                limits(),
                "Reduce the vertex count.",
            );
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(UnsupportedError);
        const msg = (caught as Error).message;
        expect(msg).toContain("[gltf] a skinned mesh's VAT");
        expect(msg).toContain("9000×4"); // the extent
        expect(msg).toContain("maxTextureDimension2D"); // the limit it tripped
        expect(msg).toContain("Reduce the vertex count."); // the remedy
    });

    test("a height past maxTextureDimension2D throws (the dimension check is the larger axis)", () => {
        expect(() => checkTextureLimits("vat", { width: 4, height: 9000 }, limits(), "r")).toThrow(
            UnsupportedError,
        );
    });

    test("layers past maxTextureArrayLayers throw a named UnsupportedError", () => {
        let caught: unknown;
        try {
            checkTextureLimits(
                "[render] an image array",
                { width: 256, height: 256, layers: 300 },
                limits(),
                "Reduce the distinct textures.",
            );
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(UnsupportedError);
        const msg = (caught as Error).message;
        expect(msg).toContain("[render] an image array");
        expect(msg).toContain("300 array layers");
        expect(msg).toContain("maxTextureArrayLayers");
        expect(msg).toContain("Reduce the distinct textures.");
    });

    test("a texture within both limits does not throw; layers default to 1 (a plain 2D texture)", () => {
        expect(() =>
            checkTextureLimits("vat", { width: 8192, height: 8192 }, limits(), "r"),
        ).not.toThrow();
        expect(() =>
            checkTextureLimits("array", { width: 256, height: 256, layers: 256 }, limits(), "r"),
        ).not.toThrow();
    });

    test("the boundaries are exclusive: exactly at each limit fits, one over throws", () => {
        expect(() => checkTextureLimits("x", { width: 8193, height: 1 }, limits(), "r")).toThrow(
            UnsupportedError,
        );
        expect(() =>
            checkTextureLimits("x", { width: 1, height: 1, layers: 257 }, limits(), "r"),
        ).toThrow(UnsupportedError);
    });
});
