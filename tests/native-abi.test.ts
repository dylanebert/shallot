import { ptr, toArrayBuffer } from "bun:ffi";
import { expect, test } from "bun:test";
import { loadNative } from "../bin/bun-native";
import { BASE_FEATURES, deviceLimits } from "../src/engine/runtime/gpu";

const { createGPUInstance } = await loadNative();

test("native loader refuses real Node before attempting the Bun FFI import", () => {
    const loader = new URL("../bin/bun-native.ts", import.meta.url).href;
    const child = Bun.spawnSync([
        "node",
        "--input-type=module",
        "-e",
        `import { loadNative } from ${JSON.stringify(loader)}; await loadNative();`,
    ]);
    expect(child.exitCode).toBe(1);
    expect(child.stderr.toString()).toContain("Shallot native setup requires Bun");
    expect(child.stderr.toString()).not.toContain("Unsupported URL scheme");
});

test("native get-limits writes its 152-byte structure, not the carrier's 168-byte allocation", async () => {
    const gpu = createGPUInstance() as any;
    try {
        const adapter = await gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        const bytes = new Uint8Array(256).fill(0xa5);
        new DataView(bytes.buffer).setBigUint64(0, 0n, true);
        expect(adapter.lib.wgpuAdapterGetLimits(adapter.adapterPtr, ptr(bytes))).toBe(1);
        const native = new DataView(bytes.buffer);
        expect(native.getUint32(52, true)).toBeGreaterThanOrEqual(10);
        expect(native.getBigUint64(72, true)).toBe(
            BigInt(adapter.limits.maxStorageBufferBindingSize),
        );
        expect(native.getBigUint64(96, true)).toBe(BigInt(adapter.limits.maxBufferSize));
        expect([...bytes.slice(152)]).toEqual(Array(104).fill(0xa5));
        expect(native.getUint32(148, true)).not.toBe(0xa5a5a5a5);
        const device = await adapter.requestDevice({
            requiredFeatures: [...BASE_FEATURES],
            requiredLimits: deviceLimits(adapter.limits),
        });
        for (const feature of BASE_FEATURES) expect(device.features.has(feature)).toBe(true);
        expect(device.limits.maxStorageBuffersPerShaderStage).toBe(10);
        expect(device.limits.maxStorageBufferBindingSize).toBe(
            adapter.limits.maxStorageBufferBindingSize,
        );
        expect(device.limits.maxBufferSize).toBe(adapter.limits.maxBufferSize);
        device.destroy();
        await device.lost;
        adapter.destroy();
    } finally {
        gpu.destroy();
    }
});

test("actual native uncaptured error survives request completion until loss", async () => {
    const gpu = createGPUInstance() as any;
    try {
        const adapter = await gpu.requestAdapter();
        const device = await adapter.requestDevice({
            requiredFeatures: [...BASE_FEATURES],
            requiredLimits: deviceLimits(adapter.limits),
        });
        let errors = 0;
        device.addEventListener("uncapturederror", () => {
            errors++;
        });
        const buffer = device.createBuffer({ size: 4, usage: 0 });
        gpu._ticker.processEvents();
        expect(errors).toBe(1);
        expect(device._acquisition.handles.has("error")).toBe(true);
        buffer.destroy();
        device.destroy();
        await device.lost;
        await Promise.resolve();
        expect(device._acquisition.handles.size).toBe(0);
        adapter.destroy();
    } finally {
        gpu.destroy();
    }
});

test("actual native rejects an owned device graph with a post-validation excessive limit", async () => {
    const gpu = createGPUInstance() as any;
    const adapter = await gpu.requestAdapter();
    await Promise.resolve();
    const limit = adapter.limits.maxStorageBuffersPerShaderStage;
    const lib = adapter.lib;
    let entries = 0;
    let returns = 0;
    let attempt: any;
    adapter.lib = {
        ...lib,
        wgpuAdapterRequestDevice: (pointer: number, descriptor: number, info: number) => {
            attempt = [...gpu._ticker.acquisitions].at(-1);
            const view = new DataView(toArrayBuffer(descriptor, 0, 144));
            const address = Number(view.getBigUint64(40, true));
            expect(
                [...attempt.arena.owners].some(
                    (buffer: any) => buffer instanceof ArrayBuffer && ptr(buffer) === address,
                ),
            ).toBe(true);
            const limits = new DataView(toArrayBuffer(address, 0, 168));
            expect(limits.getUint32(52, true)).toBe(10);
            limits.setUint32(52, limit + 1, true);
            entries++;
            const result = lib.wgpuAdapterRequestDevice(pointer, descriptor, info);
            returns++;
            return result;
        },
    };
    try {
        await expect(
            adapter.requestDevice({
                requiredFeatures: [...BASE_FEATURES],
                requiredLimits: deviceLimits(adapter.limits),
            }),
        ).rejects.toThrow("WGPU Error (Error)");
        gpu._ticker.processEvents();
        await Promise.resolve();
        await Promise.resolve();
        expect(entries).toBe(1);
        expect(returns).toBe(1);
        expect(attempt.requestDone).toBe(true);
        expect(attempt.lossDone).toBe(true);
        expect(attempt.handles.size).toBe(0);
        expect(gpu._ticker.acquisitions.size).toBe(0);
        expect(gpu._ticker._waiting).toBe(0);
    } finally {
        adapter.destroy();
        gpu.destroy();
    }
});

test("actual native shutdown cancels a pending adapter request", async () => {
    const gpu = createGPUInstance() as any;
    const completion = gpu.requestAdapter().then(
        (adapter: any) => {
            adapter?.destroy();
            return "success";
        },
        () => "cancelled",
    );
    const attempt = [...gpu._ticker.acquisitions][0] as any;
    expect(attempt.handles.size).toBe(1);
    gpu.destroy();
    expect(await completion).toBe("cancelled");
    await Promise.resolve();
    expect(attempt.handles.size).toBe(0);
    expect(gpu._ticker.acquisitions.size).toBe(0);
    expect(gpu._ticker._waiting).toBe(0);
});
