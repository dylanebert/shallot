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
    GpuDiagnosticError,
    observeDevice,
    precompile,
    precompileAll,
    precompileScope,
    requestGPU,
    resolveFeatures,
    type ShaderArtifact,
    shaderHash,
    tgslCanary,
    UnsupportedError,
    validateGpu,
} from "./gpu";

describe("device failure listeners", () => {
    test("an adopted device composes uncaptured errors and loss once without replacing the host", async () => {
        const host = () => {};
        let listener: ((event: GPUUncapturedErrorEvent) => void) | undefined;
        let lose!: (info: GPUDeviceLostInfo) => void;
        const device = {
            lost: new Promise<GPUDeviceLostInfo>((resolve) => {
                lose = resolve;
            }),
            onuncapturederror: host,
            addEventListener(type: string, callback: (event: GPUUncapturedErrorEvent) => void) {
                if (type === "uncapturederror") listener = callback;
            },
        } as unknown as GPUDevice;
        const reported: string[] = [];

        observeDevice(device, (message) => reported.push(message));
        observeDevice(device, (message) => reported.push(`duplicate:${message}`));

        expect(device.onuncapturederror).toBe(host);
        listener?.({ error: new Error("bad binding") } as unknown as GPUUncapturedErrorEvent);
        lose({ reason: "destroyed", message: "host closed" } as GPUDeviceLostInfo);
        await Promise.resolve();
        expect(reported).toEqual([
            "GPU uncaptured Error: bad binding",
            "GPU device lost destroyed: host closed",
        ]);
    });
});

describe("GPU validation scopes", () => {
    test("a thrown or scoped validation failure pops the balanced scope and names one diagnostic", async () => {
        const events: string[] = [];
        let scoped: Error | null = null;
        const device = {
            pushErrorScope(filter: GPUErrorFilter) {
                events.push(`push:${filter}`);
            },
            async popErrorScope() {
                events.push("pop");
                return scoped;
            },
        } as unknown as GPUDevice;

        const rejected = validateGpu(device, "forward", async () => {
            throw Object.assign(new Error("pipeline rejected"), { name: "OperationError" });
        });
        await expect(rejected).rejects.toMatchObject({
            name: "GpuDiagnosticError",
            label: "forward",
            errorClass: "OperationError",
            message: 'GPU "forward" OperationError: pipeline rejected',
        });
        expect(events).toEqual(["push:validation", "pop"]);

        events.length = 0;
        scoped = Object.assign(new Error("binding mismatch"), { name: "GPUValidationError" });
        const invalid = validateGpu(device, "shadow", async () => true);
        await expect(invalid).rejects.toBeInstanceOf(GpuDiagnosticError);
        await expect(invalid).rejects.toMatchObject({
            label: "shadow",
            errorClass: "GPUValidationError",
            message: 'GPU "shadow" GPUValidationError: binding mismatch',
        });
        expect(events).toEqual(["push:validation", "pop"]);
    });

    test("nested scopes stay balanced and preserve the inner diagnostic", async () => {
        const events: string[] = [];
        const results = [
            Object.assign(new Error("bad inner pipeline"), { name: "GPUValidationError" }),
            null,
        ];
        const device = {
            pushErrorScope(filter: GPUErrorFilter) {
                events.push(`push:${filter}`);
            },
            async popErrorScope() {
                events.push("pop");
                return results.shift() ?? null;
            },
        } as unknown as GPUDevice;

        const nested = validateGpu(device, "outer", () =>
            validateGpu(device, "inner", async () => true),
        );
        await expect(nested).rejects.toMatchObject({
            name: "GpuDiagnosticError",
            label: "inner",
            errorClass: "GPUValidationError",
        });
        expect(events).toEqual(["push:validation", "push:validation", "pop", "pop"]);
    });
});

describe("shader artifacts", () => {
    test("source interception is opt-in, hashed, compilation-aware, and bounded", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: unknown[] };
        };
        const prior = globals.__gpuDiagnostics;
        const original = (() => ({
            getCompilationInfo: async () => ({ messages: [] }),
        })) as unknown as GPUDevice["createShaderModule"];
        const offDevice = {
            ...fakeDevice(),
            createShaderModule: original,
        } as unknown as GPUDevice;
        try {
            delete globals.__gpuDiagnostics;
            await requestGPU(offDevice);
            expect(offDevice.createShaderModule).toBe(original);

            globals.__gpuDiagnostics = { artifacts: [] };
            let compilationCalls = 0;
            const device = {
                ...fakeDevice(),
                createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({
                    label: descriptor.label,
                    getCompilationInfo: async () => {
                        compilationCalls++;
                        return {
                            messages:
                                descriptor.label === "shader-0"
                                    ? [
                                          {
                                              type: "error",
                                              message: "invalid shader-0",
                                              lineNum: 2,
                                              linePos: 3,
                                              offset: 4,
                                              length: 5,
                                          },
                                      ]
                                    : [],
                        };
                    },
                }),
            } as unknown as GPUDevice;
            await requestGPU(device);
            await requestGPU(device);
            const source = "@compute @workgroup_size(1) fn main() {}";
            await expect(
                validateGpu(device, "seed-error", () => {
                    device.createShaderModule({ label: "shader-0", code: source });
                }),
            ).rejects.toMatchObject({ label: "shader-0", errorClass: "GPUCompilationError" });
            await validateGpu(device, "success-flood", () => {
                for (let i = 1; i < 19; i++) {
                    device.createShaderModule({ label: `shader-${i}`, code: source });
                }
            });

            // Once compilation proves an error it is monotonic/pinned; a later pending/success flood
            // stays bounded by evicting non-errors while preserving the exact failing source.
            expect(compilationCalls).toBe(19);
            const artifacts = globals.__gpuDiagnostics.artifacts as ShaderArtifact[];
            expect(artifacts).toHaveLength(16);
            expect(artifacts[0]).toMatchObject({
                label: "shader-0",
                source,
                messages: [{ type: "error", message: "invalid shader-0" }],
            });
            expect(artifacts.some((artifact) => artifact.label === "shader-1")).toBe(false);
            expect(artifacts.at(-1)).toEqual({
                label: "shader-18",
                stage: "compute",
                source,
                hash: shaderHash(source),
                messages: [],
            });
            expect(shaderHash(source)).toBe("e372f38edd1acebc");
            expect(shaderHash(`${source}\n`)).not.toBe(shaderHash(source));
        } finally {
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a broad active scope admits and retains a failing module beyond the cap", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        const source = "@compute @workgroup_size(1) fn main() {}";
        let resolveLast!: (info: GPUCompilationInfo) => void;
        const lastInfo = new Promise<GPUCompilationInfo>((resolve) => {
            resolveLast = resolve;
        });
        const device = {
            ...fakeDevice(),
            createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({
                getCompilationInfo: () =>
                    descriptor.label === "broad-19"
                        ? lastInfo
                        : Promise.resolve({ messages: [] } as unknown as GPUCompilationInfo),
            }),
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            const validation = validateGpu(device, "broad-warm", () => {
                for (let i = 0; i < 20; i++) {
                    device.createShaderModule({ label: `broad-${i}`, code: source });
                }
            });
            await Promise.resolve();
            await Promise.resolve();
            expect(globals.__gpuDiagnostics.artifacts).toHaveLength(16);
            expect(
                globals.__gpuDiagnostics.artifacts.some(
                    (artifact) => artifact.label === "broad-19",
                ),
            ).toBe(false);
            resolveLast({
                messages: [
                    {
                        type: "error",
                        message: "last module failed",
                        lineNum: 1,
                        linePos: 1,
                        offset: 0,
                        length: 1,
                    },
                ],
            } as unknown as GPUCompilationInfo);

            await expect(validation).rejects.toMatchObject({
                name: "GpuDiagnosticError",
                label: "broad-19",
                errorClass: "GPUCompilationError",
            });
            expect(globals.__gpuDiagnostics.artifacts).toHaveLength(16);
            expect(globals.__gpuDiagnostics.artifacts.at(-1)).toMatchObject({
                label: "broad-19",
                source,
                messages: [{ type: "error", message: "last module failed" }],
            });
            expect(
                globals.__gpuDiagnostics.artifacts.some((artifact) => artifact.label === "broad-0"),
            ).toBe(false);
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a failing pipeline retains its cross-scope module when all labels differ", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        const source = "@compute @workgroup_size(1) fn main() {}";
        let popCount = 0;
        const device = {
            ...fakeDevice(),
            createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({
                label: descriptor.label,
                getCompilationInfo: async () => ({ messages: [] }),
            }),
            createComputePipeline: (descriptor: GPUComputePipelineDescriptor) => descriptor,
            popErrorScope: async () => {
                popCount++;
                return popCount === 1
                    ? null
                    : Object.assign(new Error("pipeline validation failed"), {
                          name: "GPUValidationError",
                      });
            },
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            let module!: GPUShaderModule;
            await validateGpu(device, "module-scope", () => {
                module = device.createShaderModule({
                    label: "module-label",
                    code: source,
                });
            });
            await expect(
                validateGpu(device, "precompile-label", () => {
                    device.createComputePipeline({
                        label: "pipeline-label",
                        layout: "auto",
                        compute: { module, entryPoint: "main" },
                    });
                    for (let i = 0; i < 20; i++) {
                        device.createShaderModule({
                            label: `unrelated-${i}`,
                            code: source,
                        });
                    }
                }),
            ).rejects.toMatchObject({
                label: "precompile-label",
                errorClass: "GPUValidationError",
            });

            expect(globals.__gpuDiagnostics.artifacts).toHaveLength(16);
            expect(
                globals.__gpuDiagnostics.artifacts.find(
                    (artifact) => artifact.label === "module-label",
                ),
            ).toMatchObject({ source, hash: shaderHash(source) });
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("an active scope pins its correlated artifact until a delayed pipeline failure arrives", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        const source = "@compute @workgroup_size(1) fn correlated() {}";
        let resolvePop!: (error: GPUError | null) => void;
        let popStarted = false;
        const device = {
            ...fakeDevice(),
            createShaderModule: () => ({
                getCompilationInfo: async () => ({ messages: [] }),
            }),
            popErrorScope: () => {
                popStarted = true;
                return new Promise<GPUError | null>((resolve) => {
                    resolvePop = resolve;
                });
            },
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            const validation = validateGpu(device, "correlated", () => {
                device.createShaderModule({ label: "correlated", code: source });
                for (let i = 0; i < 20; i++) {
                    device.createShaderModule({ label: `ordinary-${i}`, code: source });
                }
            });
            while (!popStarted) await Promise.resolve();

            expect(globals.__gpuDiagnostics.artifacts).toHaveLength(16);
            expect(globals.__gpuDiagnostics.artifacts[0]).toMatchObject({
                label: "correlated",
                source,
                hash: shaderHash(source),
            });

            resolvePop(
                Object.assign(new Error("delayed pipeline failure"), {
                    name: "GPUValidationError",
                }) as unknown as GPUError,
            );
            await expect(validation).rejects.toMatchObject({
                label: "correlated",
                errorClass: "GPUValidationError",
            });
            expect(globals.__gpuDiagnostics.artifacts[0]).toMatchObject({
                label: "correlated",
                source,
                hash: shaderHash(source),
            });
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a delayed successful compilation result cannot unpin a scoped pipeline failure", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        let resolveInfo!: (info: GPUCompilationInfo) => void;
        const delayed = new Promise<GPUCompilationInfo>((resolve) => {
            resolveInfo = resolve;
        });
        const source = "@compute @workgroup_size(1) fn main() {}";
        const device = {
            ...fakeDevice(),
            createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({
                getCompilationInfo: () =>
                    descriptor.label === "delayed"
                        ? delayed
                        : Promise.resolve({ messages: [] } as unknown as GPUCompilationInfo),
            }),
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            const validation = validateGpu(device, "delayed", () => {
                device.createShaderModule({ label: "delayed", code: source });
                throw Object.assign(new Error("pipeline rejected"), { name: "OperationError" });
            });
            resolveInfo({ messages: [] } as unknown as GPUCompilationInfo);
            await expect(validation).rejects.toMatchObject({
                label: "delayed",
                errorClass: "OperationError",
            });

            await validateGpu(device, "later-context", () => {
                for (let i = 0; i < 20; i++) {
                    device.createShaderModule({ label: `later-${i}`, code: source });
                }
            });
            expect(globals.__gpuDiagnostics.artifacts).toHaveLength(16);
            expect(globals.__gpuDiagnostics.artifacts[0]).toMatchObject({
                label: "delayed",
                source,
            });
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a compilation-info rejection before settlement is retained by its raw scope", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        let rejectInfo!: (reason?: unknown) => void;
        const info = new Promise<GPUCompilationInfo>((_, reject) => {
            rejectInfo = reject;
        });
        const device = {
            ...fakeDevice(),
            createShaderModule: () => ({ getCompilationInfo: () => info }),
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            const validation = validateGpu(device, "raw-before", async () => {
                device.createShaderModule({
                    label: "shader-info-before",
                    code: "@compute fn main() {}",
                });
                rejectInfo(
                    Object.assign(new Error("compiler unavailable"), { name: "OperationError" }),
                );
                await Promise.resolve();
                await Promise.resolve();
            });

            await expect(validation).rejects.toMatchObject({
                name: "GpuDiagnosticError",
                label: "shader-info-before",
                errorClass: "OperationError",
            });
            expect(globals.__gpuDiagnostics.artifacts[0].compilationError).toMatchObject({
                errorClass: "OperationError",
            });
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a compilation-info rejection during settlement rejects its raw validation scope", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        let rejectInfo!: (reason?: unknown) => void;
        const info = new Promise<GPUCompilationInfo>((_, reject) => {
            rejectInfo = reject;
        });
        const device = {
            ...fakeDevice(),
            createShaderModule: () => ({ getCompilationInfo: () => info }),
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = { artifacts: [] };
            await requestGPU(device);
            const validating = validateGpu(device, "raw-during", () => {
                device.createShaderModule({
                    label: "shader-info-during",
                    code: "@compute fn main() {}",
                });
            });
            const outcome = validating.then(
                () => null,
                (error: unknown) => error,
            );
            await Promise.resolve();
            rejectInfo(
                Object.assign(new Error("compiler disconnected"), { name: "OperationError" }),
            );
            expect(await outcome).toMatchObject({
                name: "GpuDiagnosticError",
                label: "shader-info-during",
                errorClass: "OperationError",
            });
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });

    test("a retired active scope cannot populate its replacement session", async () => {
        const globals = globalThis as unknown as {
            __gpuDiagnostics?: { artifacts: ShaderArtifact[] };
        };
        const prior = globals.__gpuDiagnostics;
        const saved = { ...Compute };
        const oldCapture = { artifacts: [] as ShaderArtifact[] };
        let resolveOld!: (info: GPUCompilationInfo) => void;
        const oldInfo = new Promise<GPUCompilationInfo>((resolve) => {
            resolveOld = resolve;
        });
        const oldDevice = {
            ...fakeDevice(),
            createShaderModule: () => ({
                getCompilationInfo: () => oldInfo,
            }),
        } as unknown as GPUDevice;
        try {
            globals.__gpuDiagnostics = oldCapture;
            await requestGPU(oldDevice);
            const validation = validateGpu(oldDevice, "old", () => {
                oldDevice.createShaderModule({ label: "old", code: "@compute fn old() {}" });
            });

            const replacement = { artifacts: [] as ShaderArtifact[] };
            globals.__gpuDiagnostics = replacement;
            await requestGPU({
                ...fakeDevice(),
                createShaderModule: () => ({
                    getCompilationInfo: async () => ({ messages: [] }),
                }),
            } as unknown as GPUDevice);
            resolveOld({
                messages: [
                    {
                        type: "error",
                        message: "stale compiler result",
                    },
                ],
            } as unknown as GPUCompilationInfo);
            await validation;

            expect(oldCapture.artifacts).toEqual([]);
            expect(replacement.artifacts).toEqual([]);
        } finally {
            Object.assign(Compute, saved);
            if (prior) globals.__gpuDiagnostics = prior;
            else delete globals.__gpuDiagnostics;
        }
    });
});

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

    // the other failure checkTgsl guards: a second typegpu copy loaded after the engine's. The key is
    // redefined as a write-counting accessor (gpu.ts), so mutating it at check time exercises the real
    // production path; restore both the version and the counter in `finally` (the counter, not just
    // the version, since a later test would otherwise inherit an elevated count from this one).
    test("checkTgsl throws when a second typegpu copy's version differs from the engine's", () => {
        const globals = globalThis as unknown as Record<string, unknown>;
        const priorVersion = globals.__TYPEGPU_VERSION__;
        const priorWrites = globals.__SHALLOT_TYPEGPU_WRITES__;
        globals.__TYPEGPU_VERSION__ = `${priorVersion ?? "0.11.9"}-other-copy`;
        try {
            expect(() => checkTgsl()).toThrow(/Two copies of typegpu are loaded/);
        } finally {
            globals.__TYPEGPU_VERSION__ = priorVersion;
            globals.__SHALLOT_TYPEGPU_WRITES__ = priorWrites;
        }
    });

    // the gap a value comparison structurally can't close: two copies of a minor-pinned typegpu stamp
    // identical text, so `version !== snapshot` never trips. Write the *same* value back — typegpu's
    // own module top level would do exactly this on a same-version duplicate — and the write-count
    // still crosses 1, because the accessor counts writes, not distinct values.
    test("checkTgsl throws on a same-version duplicate (the write-count, not value, is what trips)", () => {
        const globals = globalThis as unknown as Record<string, unknown>;
        const priorWrites = globals.__SHALLOT_TYPEGPU_WRITES__;
        const sameVersion = globals.__TYPEGPU_VERSION__;
        globals.__TYPEGPU_VERSION__ = sameVersion;
        try {
            expect(() => checkTgsl()).toThrow(/Two copies of typegpu are loaded/);
        } finally {
            globals.__SHALLOT_TYPEGPU_WRITES__ = priorWrites;
        }
    });
});

// a device stand-in: `initFromDevice` only stores the handle and nothing here submits work, so root
// identity and the build boundary are testable with no adapter (testing.md — never bind a device in
// `bun test`). `Compute` is a process singleton every other test file also writes, so each test that
// drives `requestGPU` restores what it found.
const fakeDevice = () =>
    ({
        queue: { onSubmittedWorkDone: async () => {} },
        features: new Set(),
        limits: {},
        lost: new Promise(() => {}),
        pushErrorScope: () => {},
        popErrorScope: async () => null,
    }) as unknown as GPUDevice;

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

// The queue is module state whose "before the drain" half exists only until `precompileAll`.
// Each test opens a fresh build with `requestGPU`, the same boundary production uses.
describe("precompile", () => {
    test("the normal drain validates and fences each forcer without ProfilePlugin", async () => {
        const saved = { ...Compute };
        const events: string[] = [];
        const device = fakeDevice();
        device.pushErrorScope = (filter) => {
            events.push(`push:${filter}`);
        };
        device.popErrorScope = async () => {
            events.push("pop");
            return null;
        };
        try {
            await requestGPU(device);
            precompile("forward", () => {
                events.push("force");
                return {
                    initAsync: async () => {
                        events.push("fence");
                    },
                };
            });
            await precompileAll();
            expect(events).toEqual(["push:validation", "force", "fence", "pop"]);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("dependencies drain in stable topological order and reject invalid graphs", async () => {
        const saved = { ...Compute };
        try {
            await requestGPU(fakeDevice());
            const order: string[] = [];
            precompile("first", () => order.push("first"));
            precompile("dependent-a", () => order.push("dependent-a"), { after: ["publish"] });
            precompile("independent", () => order.push("independent"));
            precompile("dependent-b", () => order.push("dependent-b"), { after: ["publish"] });
            precompile("publish", () => order.push("publish"));
            precompile("optional", () => order.push("optional"), { after: ["missing"] });

            await precompileAll();
            expect(order).toEqual([
                "first",
                "independent",
                "publish",
                "dependent-a",
                "dependent-b",
                "optional",
            ]);

            await requestGPU(fakeDevice());
            precompile("duplicate", () => true);
            expect(() => precompile("duplicate", () => true)).toThrow(
                /duplicate precompile label "duplicate"/,
            );

            await requestGPU(fakeDevice());
            precompile("cycle-a", () => true, { after: ["cycle-b"] });
            precompile("cycle-b", () => true, { after: ["cycle-a"] });
            await expect(precompileAll()).rejects.toThrow(/precompile cycle.*cycle-a.*cycle-b/);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a scope numbers repeat instances, and resets with the label set it keeps unique", async () => {
        const saved = { ...Compute };
        try {
            await requestGPU(fakeDevice());
            // the single-instance app is the common one — its labels (and profiler rows) stay bare
            expect(precompileScope("radix")).toBe("radix");
            expect(precompileScope("radix")).toBe("radix-2");
            expect(precompileScope("bounds")).toBe("bounds");
            // the counts belong to the label set, so a rebuild names its stages the same way again
            await requestGPU(fakeDevice());
            expect(precompileScope("radix")).toBe("radix");
        } finally {
            Object.assign(Compute, saved);
        }
    });

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
        await expect(precompileAll()).rejects.toThrow(/precompile "broken" failed/);
        // "c" was never shifted off, so the throw costs one pipeline, not the rest of the queue
        expect(order).toEqual(["a", "b"]);
        await precompileAll();
        expect(order).toEqual(["a", "b", "c"]);
        await precompileAll();
        expect(order).toEqual(["a", "b", "c"]);

        // past the drain a lazily-built pipeline compiles on arrival: late beats silently dropped,
        // which would surface as a multi-second first-frame stall with nothing pointing at it
        await precompile("late", () => order.push("late"));
        expect(order).toEqual(["a", "b", "c", "late"]);
        await expect(
            precompile("late-broken", () => {
                throw new Error("createComputePipeline failed");
            }),
        ).rejects.toThrow(/precompile "late-broken" failed/);

        // a forcer that binds nothing dispatches nothing, so its pipeline silently falls through to the
        // first frame — the multi-second stall the queue exists to prevent. The drain refuses it
        await expect(precompile("unbound", () => null)).rejects.toThrow(
            /precompile "unbound" bound nothing/,
        );
        Object.assign(Compute, saved);
    });
});
