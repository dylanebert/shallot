import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import { GltfPlugin } from "../../extras/gltf";
import { ProfilePlugin } from "../../extras/profile";
import {
    BASE_FEATURES,
    Compute,
    checkStorageBinding,
    checkTextureLimits,
    checkTgsl,
    deviceLimits,
    precompile,
    precompileAll,
    requestGPU,
    resolveFeatures,
    tgslCanary,
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

describe("BASE_FEATURES — the floor conformance oracle", () => {
    // an adapter offering every WebGPU feature except one named absentee
    const without = (absent: GPUFeatureName) => ({ has: (f: GPUFeatureName) => f !== absent });
    // the device the whole audit is about, named independently of BASE_FEATURES so putting an entry
    // back on the floor goes red here: it lacks everything the opt-in plugins moved off, nothing else.
    const decommissioned = [
        "timestamp-query",
        "texture-compression-bc",
        "texture-compression-etc2",
        "texture-compression-astc",
    ];
    const modest = { has: (f: GPUFeatureName) => !decommissioned.includes(f) };
    const profileFeatures = ProfilePlugin.features ?? [];
    const gltfPreferred = GltfPlugin.preferredFeatures ?? [];

    // The floor gates device acquisition for *every* app, before any plugin loads, so an entry only an
    // opt-in plugin needs rejects hardware that would have run the scene fine. That's the governing
    // rule: a floor entry earns its place only by being a `DEFAULT_PLUGINS` need. Neither `ProfilePlugin`
    // nor `GltfPlugin` is in that set, so nothing either declares may sit on the floor.
    test("no feature an opt-in plugin declares sits on the floor", () => {
        expect(profileFeatures).toEqual(["timestamp-query"]);
        expect(gltfPreferred.length).toBe(3);
        for (const f of [...profileFeatures, ...gltfPreferred, ...(GltfPlugin.features ?? [])]) {
            expect(BASE_FEATURES).not.toContain(f);
        }
    });

    // `shader-f16` isn't a floor need either: sear's `material` mirror binds `vec2<u32>` +
    // `unpack2x16float` (core WGSL), so nothing on the default path declares an `f16` type. The
    // shader-side half is `standard/sear/surface.test.ts`.
    test("a device lacking shader-f16 still meets the floor", () => {
        expect(BASE_FEATURES).not.toContain("shader-f16");
        expect(resolveFeatures(without("shader-f16"), BASE_FEATURES, []).missing).toEqual([]);
    });

    test("a device with no timestamp-query and no texture compression still meets the floor", () => {
        expect(resolveFeatures(modest, BASE_FEATURES, []).missing).toEqual([]);
    });

    // the compression families are preferred, never required: that same device loads a glTF app, and
    // the per-image gate in `gltf/assets.ts` is what fails, only on a KTX2 image.
    test("GltfPlugin's compression families never gate the device", () => {
        const required = [...BASE_FEATURES, ...(GltfPlugin.features ?? [])] as GPUFeatureName[];
        expect(resolveFeatures(modest, required, gltfPreferred).missing).toEqual([]);
        // and the trap they exist for: `target.ts` branches on `device.features.has`, false for a family
        // the device never *requested* — so each family must be requested against an adapter that offers
        // that one alone. Naming them here (not reading them off `gltfPreferred`) is what makes dropping
        // one go red: against an all-true adapter `granted` is `preferred` for any list at all.
        for (const f of [
            "texture-compression-bc",
            "texture-compression-etc2",
            "texture-compression-astc",
        ] as GPUFeatureName[]) {
            const base: readonly string[] = BASE_FEATURES;
            const only = { has: (x: GPUFeatureName) => x === f || base.includes(x) };
            const { granted, missing } = resolveFeatures(only, BASE_FEATURES, gltfPreferred);
            expect(granted).toEqual([f]);
            expect(missing).toEqual([]);
        }
    });

    // ProfilePlugin's is required, not preferred: an explicitly added debug plugin fails loud rather
    // than silently reporting no GPU spans.
    test("a ProfilePlugin app on that same device fails loud, naming timestamp-query", () => {
        const required = [...BASE_FEATURES, ...profileFeatures] as GPUFeatureName[];
        expect(resolveFeatures(modest, required, []).missing).toEqual(["timestamp-query"]);
    });
});

describe("resolveFeatures", () => {
    // an adapter exposing exactly `subgroups` + the base floor it cares about here.
    const adapter = (...present: GPUFeatureName[]) => new Set<GPUFeatureName>(present);

    test("a required feature the adapter has is granted nothing extra but never missing", () => {
        const { granted, missing } = resolveFeatures(
            adapter("float32-filterable"),
            ["float32-filterable"],
            [],
        );
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

describe("TGSL metadata", () => {
    test("the canary resolves to its WGSL body when the build ran the typegpu transform", () => {
        // the whole distribution contract in one assertion: a body the transform must have
        // transpiled at build time. Delete the bunfig preload and this goes red.
        const wgsl = tgpu.resolve([tgslCanary]);
        expect(wgsl).toContain("(x + 1u)");
        expect(wgsl).toContain("-> u32");
    });

    // the throwing arm can't be built here — the transform runs over this file too, so any fn
    // declared in it carries metadata. Its red proof is the no-plugin arm of `bun run test:install`
    test("checkTgsl passes under a transformed build", () => {
        expect(() => checkTgsl()).not.toThrow();
    });

    // the other failure checkTgsl guards: a second typegpu copy loaded after the engine's, observable
    // only via the version snapshot taken at module load — mutate the global read at check time and
    // restore it, rather than the snapshot itself (which no test can rewind).
    test("checkTgsl throws when a second typegpu copy's version differs from the engine's", () => {
        const globals = globalThis as unknown as Record<string, string | undefined>;
        const prior = globals.__TYPEGPU_VERSION__;
        globals.__TYPEGPU_VERSION__ = `${prior ?? "0.11.9"}-other-copy`;
        try {
            expect(() => checkTgsl()).toThrow(/Two copies of typegpu are loaded/);
        } finally {
            globals.__TYPEGPU_VERSION__ = prior;
        }
    });
});

// a device stand-in: `initFromDevice` only stores the handle and nothing here submits work, so root
// identity and the build boundary are testable with no adapter (testing.md — never bind a device in
// `bun test`). `Compute` is a process singleton every other test file also writes, so each test that
// drives `requestGPU` restores what it found.
const fakeDevice = () => ({ queue: {}, features: new Set(), limits: {} }) as unknown as GPUDevice;

describe("the TypeGPU root", () => {
    test("is memoized per device; a rebuild keeps it, a new device mints a new one", async () => {
        const saved = { ...Compute };
        try {
            const device = fakeDevice();
            await requestGPU(device);
            const root = Compute.root;
            expect(root).toBeDefined();

            // the rebuild case: a host reusing one device across builds (the cross-build memo law)
            // must find its typed resources still alive, so the root has to survive the rebuild —
            // while the publish maps wipe, typed beside raw
            Compute.typed.set("transforms", {} as never);
            await requestGPU(device);
            expect(Compute.root).toBe(root);
            expect(Compute.typed.size).toBe(0);

            await requestGPU(fakeDevice());
            expect(Compute.root).not.toBe(root);
        } finally {
            Object.assign(Compute, saved);
        }
    });
});

// One test, walked in order, because the queue is module state whose "before the drain" half exists
// only until the first `precompileAll` — `requestGPU` is what re-opens it, once per build. Splitting
// these into separate tests would make them pass or fail on file order.
describe("precompile", () => {
    test("the queue's lifecycle: held through warm, drained once, late arrivals run on the spot", async () => {
        const saved = { ...Compute };
        // a build began: that, and only that, re-opens the queue
        await requestGPU(fakeDevice());
        const order: string[] = [];
        precompile("a", () => order.push("a"));
        precompile("b", () => order.push("b"));
        // nothing runs during warm — a dispatch here could hit a bind group whose dependency another
        // plugin's warm hasn't published yet
        expect(order).toEqual([]);

        precompile("broken", () => {
            throw new Error("createComputePipeline failed");
        });
        precompile("c", () => order.push("c"));

        // the throw names the pipeline, so a build failure points at the kernel, not at `build`
        expect(() => precompileAll()).toThrow(/precompile "broken" failed/);
        // "c" was never shifted off, so the throw costs one pipeline, not the rest of the queue
        expect(order).toEqual(["a", "b"]);
        precompileAll();
        expect(order).toEqual(["a", "b", "c"]);
        precompileAll();
        expect(order).toEqual(["a", "b", "c"]);

        // past the drain a lazily-built pipeline compiles on arrival: late beats silently dropped,
        // which would surface as a multi-second first-frame stall with nothing pointing at it
        precompile("late", () => order.push("late"));
        expect(order).toEqual(["a", "b", "c", "late"]);
        expect(() => {
            precompile("late-broken", () => {
                throw new Error("createComputePipeline failed");
            });
        }).toThrow(/precompile "late-broken" failed/);
        Object.assign(Compute, saved);
    });
});
