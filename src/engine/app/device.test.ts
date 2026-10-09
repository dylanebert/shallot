import { expect, test } from "bun:test";
import { RenderingPlugin } from "../../core/rendering";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderingPlugin } from "../../standard/rendering";
import { stampAdapter } from "../index";
import { createApp, type Plugin } from "./index";

const fallbackAdapter = {
    info: {
        vendor: "google",
        architecture: "swiftshader",
        device: "fallback",
        description: "SwiftShader",
        isFallbackAdapter: true,
    },
} as unknown as GPUAdapter;

test("GPU acquisition accepts a fallback adapter without stamping its verdict, so an app can look like it has real hardware", () => {
    const verdict = stampAdapter(fallbackAdapter);
    expect(verdict.class).toBe("fallback");
    expect(verdict.identity).toContain("SwiftShader");
});

test("an externally supplied GPU device without its adapter can be mistaken for a real adapter", () => {
    const verdict = stampAdapter();
    expect(verdict.class).toBe("unidentified");
    expect(verdict.identity).toBe("unidentified");
});

function withGpu(gpu: GPU | undefined): () => void {
    const previous = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });
    return () => {
        if (previous) Object.defineProperty(navigator, "gpu", previous);
        else Reflect.deleteProperty(navigator, "gpu");
    };
}

function fakeAdapter(
    features: readonly GPUFeatureName[],
    onRequest: (descriptor: GPUDeviceDescriptor) => void,
    limits: Partial<GPUSupportedLimits> = {},
): GPUAdapter {
    return {
        features: new Set(features),
        limits: { maxStorageBuffersPerShaderStage: 16, ...limits },
        requestDevice: async (descriptor: GPUDeviceDescriptor) => {
            onRequest(descriptor);
            throw new Error("stop after capability request");
        },
    } as unknown as GPUAdapter;
}

async function buildFailure(config: Parameters<typeof createApp>[0]): Promise<string> {
    try {
        const app = await createApp(config);
        app.dispose();
        return "build unexpectedly succeeded";
    } catch (error) {
        return error instanceof Error ? error.message : String(error);
    }
}

test("GPU needs live on their consumers: rendering needs HDR, standard rendering needs indirect draws and ten storage buffers", () => {
    expect(RenderingPlugin.gpu).toEqual({ features: ["rg11b10ufloat-renderable"] });
    expect(StandardRenderingPlugin.gpu).toEqual({
        features: ["indirect-first-instance"],
        limits: { maxStorageBuffersPerShaderStage: 10 },
    });
});

test("acquisition requests only enabled plugin requirements before warm-up", async () => {
    const requests: GPUDeviceDescriptor[] = [];
    const adapter = fakeAdapter(
        ["indirect-first-instance", "rg11b10ufloat-renderable"],
        (descriptor) => requests.push(descriptor),
    );
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    let warmed = false;
    try {
        await buildFailure({
            defaults: false,
            plugins: [
                ...DEFAULT_PLUGINS,
                {
                    name: "WarmProbe",
                    warm() {
                        warmed = true;
                    },
                },
            ],
        });
        expect(requests).toHaveLength(1);
        expect([...requests[0].requiredFeatures!].sort()).toEqual([
            "indirect-first-instance",
            "rg11b10ufloat-renderable",
        ]);
        expect(requests[0].requiredLimits).toEqual({
            maxStorageBuffersPerShaderStage: 10,
        });
        expect(warmed).toBe(false);
    } finally {
        restore();
    }
});

test("a compute-only empty declaration requests no features or non-default limits", async () => {
    let request: GPUDeviceDescriptor | undefined;
    const adapter = fakeAdapter([], (descriptor) => (request = descriptor));
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        await buildFailure({
            defaults: false,
            plugins: [{ name: "ComputeOnly", gpu: {} } as Plugin],
        });
        expect(request?.requiredFeatures).toEqual([]);
        expect(request?.requiredLimits).toEqual({});
    } finally {
        restore();
    }
});

test("split-stage zero limits stay explicit beside declared plugin limits", async () => {
    let request: GPUDeviceDescriptor | undefined;
    const adapter = fakeAdapter([], (descriptor) => (request = descriptor), {
        maxStorageBuffersInVertexStage: 0,
        maxStorageBuffersInFragmentStage: 0,
        maxStorageTexturesInVertexStage: 0,
        maxStorageTexturesInFragmentStage: 0,
    });
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        await buildFailure({
            defaults: false,
            plugins: [
                {
                    name: "StorageOwner",
                    gpu: { limits: { maxStorageBuffersPerShaderStage: 10 } },
                } as Plugin,
            ],
        });
        expect(request?.requiredLimits).toEqual({
            maxStorageBuffersPerShaderStage: 10,
            maxStorageBuffersInVertexStage: 0,
            maxStorageBuffersInFragmentStage: 0,
            maxStorageTexturesInVertexStage: 0,
            maxStorageTexturesInFragmentStage: 0,
        });
    } finally {
        restore();
    }
});

test("minimum alignment requirements merge to the tighter value", async () => {
    let request: GPUDeviceDescriptor | undefined;
    const adapter = fakeAdapter([], (descriptor) => (request = descriptor), {
        minUniformBufferOffsetAlignment: 64,
    });
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        await buildFailure({
            defaults: false,
            plugins: [
                {
                    name: "WideUniformAlignment",
                    gpu: { limits: { minUniformBufferOffsetAlignment: 512 } },
                } as Plugin,
                {
                    name: "TightUniformAlignment",
                    gpu: { limits: { minUniformBufferOffsetAlignment: 128 } },
                } as Plugin,
            ],
        });
        expect(request?.requiredLimits).toMatchObject({
            minUniformBufferOffsetAlignment: 128,
        });
    } finally {
        restore();
    }
});

test("an acquired device with worse alignment than required refuses before device creation", async () => {
    let requestedDevice = false;
    const adapter = fakeAdapter([], () => (requestedDevice = true), {
        minUniformBufferOffsetAlignment: 256,
    });
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        const message = await buildFailure({
            defaults: false,
            plugins: [
                {
                    name: "UniformAlignmentOwner",
                    gpu: { limits: { minUniformBufferOffsetAlignment: 128 } },
                } as Plugin,
            ],
        });
        expect(message).toContain('Plugin "UniformAlignmentOwner"');
        expect(message).toContain("minUniformBufferOffsetAlignment");
        expect(requestedDevice).toBe(false);
    } finally {
        restore();
    }
});

test("an acquired adapter missing a declared feature refuses with its plugin and cause", async () => {
    let requestedDevice = false;
    let initialized = false;
    const adapter = fakeAdapter([], () => (requestedDevice = true));
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        const message = await buildFailure({
            defaults: false,
            plugins: [
                {
                    name: "FeatureOwner",
                    gpu: { features: ["timestamp-query"] },
                    initialize() {
                        initialized = true;
                    },
                } as Plugin,
            ],
        });
        expect(message).toContain('Plugin "FeatureOwner"');
        expect(message).toContain('feature "timestamp-query"');
        expect(requestedDevice).toBe(false);
        expect(initialized).toBe(false);
    } finally {
        restore();
    }
});

test("an adapter below a declared limit refuses with its plugin before device creation", async () => {
    let requestedDevice = false;
    const adapter = fakeAdapter([], () => (requestedDevice = true), {
        maxStorageBuffersPerShaderStage: 8,
    });
    const restore = withGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        const message = await buildFailure({
            defaults: false,
            plugins: [
                {
                    name: "LimitOwner",
                    gpu: { limits: { maxStorageBuffersPerShaderStage: 10 } },
                } as Plugin,
            ],
        });
        expect(message).toContain('Plugin "LimitOwner"');
        expect(message).toContain("maxStorageBuffersPerShaderStage 10");
        expect(requestedDevice).toBe(false);
    } finally {
        restore();
    }
});

test("a supplied device below a declared limit refuses before initialization", async () => {
    let initialized = false;
    const device = {
        features: new Set<GPUFeatureName>(),
        limits: { maxStorageBuffersPerShaderStage: 8 },
    } as unknown as GPUDevice;
    const message = await buildFailure({
        defaults: false,
        device,
        plugins: [
            {
                name: "SuppliedLimitOwner",
                gpu: { limits: { maxStorageBuffersPerShaderStage: 10 } },
                initialize() {
                    initialized = true;
                },
            } as Plugin,
        ],
    });
    expect(message).toContain('Plugin "SuppliedLimitOwner"');
    expect(message).toContain("maxStorageBuffersPerShaderStage 10");
    expect(initialized).toBe(false);
});

test("a supplied device missing a declared feature refuses before initialization", async () => {
    let initialized = false;
    const device = {
        features: new Set<GPUFeatureName>(),
        limits: { maxStorageBuffersPerShaderStage: 16 },
    } as unknown as GPUDevice;
    const message = await buildFailure({
        defaults: false,
        device,
        plugins: [
            {
                name: "SuppliedFeatureOwner",
                gpu: {
                    features: ["timestamp-query"],
                },
                initialize() {
                    initialized = true;
                },
            } as Plugin,
        ],
    });
    expect(message).toContain('Plugin "SuppliedFeatureOwner"');
    expect(message).toContain('feature "timestamp-query"');
    expect(initialized).toBe(false);
});
