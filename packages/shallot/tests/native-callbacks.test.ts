import { CFunction, toArrayBuffer } from "bun:ffi";
import { expect, test } from "bun:test";
import { BASE_FEATURES, deviceLimits } from "../../shallot-runtime/src/engine/runtime/gpu";
import { loadNative } from "../../shallot-tooling/bin/bun-native";

const { createGPUInstance } = await loadNative();
const descriptor = (adapter: GPUAdapter) => ({
    label: "callbacks label",
    defaultQueue: { label: "callbacks queue" },
    requiredFeatures: [...BASE_FEATURES],
    requiredLimits: deviceLimits(adapter.limits),
});
const turn = async () => {
    await Promise.resolve();
    await Promise.resolve();
};

for (const mode of [
    "adapter-decode",
    "device-decode",
    "queued-error",
    "loss-decode",
    "error-decode",
    "loss-delivery",
    "live-error",
]) {
    test(`actual callback delivery ${mode}`, () => {
        const child = Bun.spawnSync(
            [process.execPath, `${import.meta.dir}/native-callbacks-delivery.fixture.ts`],
            {
                env: { ...process.env, CALLBACK_MODE: mode },
                timeout: 4000,
            },
        );
        const diagnostic = ["loss-decode", "error-decode", "loss-delivery", "live-error"].includes(
            mode,
        );
        expect(child.stderr.toString()).toBe(diagnostic ? `controlled ${mode} failure\n` : "");
        expect(child.exitCode).toBe(diagnostic ? 1 : 0);
        const result = JSON.parse(child.stdout.toString());
        expect(result.returns).toBe(1);
        expect(result.handles).toBe(0);
        expect(result.diagnostic).toBe(diagnostic);
    });
}

for (const status of [1, 2, 3, 4, 5]) {
    test(`adapter callback status ${status} with null handle`, async () => {
        const gpu = createGPUInstance() as any;
        let attempt: any;
        let closes = 0;
        gpu.lib = {
            ...gpu.lib,
            wgpuInstanceRequestAdapter: (_instance: number, _options: number, info: number) => {
                attempt = [...gpu._ticker.acquisitions].at(-1);
                const handle = attempt.handles.get("request");
                const close = handle.close.bind(handle);
                handle.close = () => {
                    closes++;
                    close();
                };
                const view = new DataView(toArrayBuffer(info, 0, 40));
                const callback = CFunction({
                    ptr: Number(view.getBigUint64(16, true)) as any,
                    args: ["u32", "ptr", "ptr", "u64", "ptr", "ptr"],
                    returns: "void",
                });
                callback(status, null, null, 0n, null, null);
                expect(closes).toBe(0);
                callback.close();
                return 0n;
            },
        };
        try {
            if (status === 3) expect(await gpu.requestAdapter()).toBeNull();
            else
                await expect(gpu.requestAdapter()).rejects.toThrow(
                    `WGPU Error (${status}): [empty message]`,
                );
            await turn();
            expect(closes).toBe(1);
            expect(attempt.handles.size).toBe(0);
            expect(gpu._ticker.acquisitions.size).toBe(0);
            expect(gpu._ticker._waiting).toBe(0);
        } finally {
            gpu.destroy();
        }
    });
}

test("reentrant instance shutdown inside acquisition entry defers callback closes", async () => {
    const gpu = createGPUInstance() as any;
    const lib = gpu.lib;
    let attempt: any;
    let closes = 0;
    let eventsAfterRelease = 0;
    const processEvents = gpu._ticker.lib.wgpuInstanceProcessEvents;
    gpu._ticker.lib = {
        ...lib,
        wgpuInstanceProcessEvents: (pointer: number) => {
            if (gpu._destroyed) eventsAfterRelease++;
            return processEvents(pointer);
        },
    };
    gpu.lib = {
        ...lib,
        wgpuInstanceRequestAdapter: (...args: any[]) => {
            attempt = [...gpu._ticker.acquisitions].at(-1);
            const handle = attempt.handles.get("request");
            const close = handle.close.bind(handle);
            handle.close = () => {
                closes++;
                close();
            };
            const result = lib.wgpuInstanceRequestAdapter(...args);
            gpu.destroy();
            expect(closes).toBe(0);
            return result;
        },
    };
    await expect(gpu.requestAdapter()).rejects.toThrow("WGPU Error (2)");
    await turn();
    gpu._ticker.processEvents();
    expect(closes).toBe(1);
    expect(eventsAfterRelease).toBe(0);
    expect(attempt.handles.size).toBe(0);
    expect(gpu._ticker.acquisitions.size).toBe(0);
    expect(gpu._ticker._waiting).toBe(0);
});

for (const mode of ["before", "null"]) {
    for (const site of ["adapter", "device"]) {
        for (let target = 1; target <= (site === "adapter" ? 1 : 7); target++) {
            test(`callback constructor ${site} ${target} ${mode}`, () => {
                const child = Bun.spawnSync(
                    [process.execPath, `${import.meta.dir}/native-callbacks.fixture.ts`],
                    {
                        env: {
                            ...process.env,
                            CALLBACK_MODE: mode,
                            CALLBACK_SITE: site,
                            CALLBACK_TARGET: String(target),
                        },
                        timeout: 4000,
                    },
                );
                expect(child.stderr.toString()).toBe("");
                expect(child.exitCode).toBe(0);
                expect(JSON.parse(child.stdout.toString()).target).toBe(target);
            });
        }
    }
}

for (const status of [1, 2, 3, 4]) {
    for (const order of ["loss-first", "request-first"]) {
        test(`native callback trampolines: request ${status}, ${order}`, async () => {
            const gpu = createGPUInstance() as any;
            try {
                const adapter = await gpu.requestAdapter();
                await turn();
                let attempt: any;
                let callbacks: any;
                adapter.lib = {
                    ...adapter.lib,
                    wgpuAdapterRequestDevice: (_adapter: number, desc: number, info: number) => {
                        attempt = [...gpu._ticker.acquisitions].at(-1);
                        const d = new DataView(toArrayBuffer(desc, 0, 144));
                        const r = new DataView(toArrayBuffer(info, 0, 40));
                        callbacks = {
                            request: CFunction({
                                ptr: Number(r.getBigUint64(16, true)) as any,
                                args: ["u32", "ptr", "ptr", "u64", "ptr", "ptr"],
                                returns: "void",
                            }),
                            loss: CFunction({
                                ptr: Number(d.getBigUint64(88, true)) as any,
                                args: ["ptr", "u32", "ptr", "u64", "ptr", "ptr"],
                                returns: "void",
                            }),
                            error: CFunction({
                                ptr: Number(d.getBigUint64(120, true)) as any,
                                args: ["ptr", "u32", "ptr", "u64", "ptr", "ptr"],
                                returns: "void",
                            }),
                        };
                        return 0n;
                    },
                };
                const rejected = adapter.requestDevice(descriptor(adapter)).then(
                    () => null,
                    (error: Error) => error,
                );
                expect(attempt.handles.size).toBe(3);
                const handles = [...attempt.handles.values()] as any[];
                const counts = new Map(handles.map((handle) => [handle, 0]));
                for (const handle of handles) {
                    const close = handle.close.bind(handle);
                    handle.close = () => {
                        counts.set(handle, counts.get(handle)! + 1);
                        close();
                    };
                }
                callbacks.error(null, 1, null, 0n, null, null);
                expect(attempt.handles.size).toBe(3);
                if (order === "loss-first") callbacks.loss(null, 4, null, 0n, null, null);
                callbacks.request(status, null, null, 0n, null, null);
                // No handle can close on the callback/native stack itself.
                expect([...counts.values()]).toEqual([0, 0, 0]);
                if (order === "request-first") {
                    await turn();
                    expect(attempt.handles.has("loss")).toBe(true);
                    expect(attempt.handles.has("error")).toBe(true);
                    callbacks.loss(null, 4, null, 0n, null, null);
                }
                const failure = await rejected;
                expect(failure).toBeInstanceOf(Error);
                expect(failure?.name).toBe("OperationError");
                expect(failure?.message).toBe(
                    `WGPU Error (${{ 1: "Success", 2: "CallbackCancelled", 3: "Error", 4: "Unknown" }[status]}): [empty message]`,
                );
                await turn();
                expect([...counts.values()]).toEqual([1, 1, 1]);
                expect(attempt.handles.size).toBe(0);
                expect(gpu._ticker.acquisitions.size).toBe(0);
                expect(gpu._ticker._waiting).toBe(0);
                for (const callback of Object.values(callbacks) as any[]) callback.close();
                adapter.destroy();
            } finally {
                gpu.destroy();
            }
        });
    }
}

for (let allocation = 1; allocation <= 11; allocation++) {
    test(`pre-entry allocation ${allocation} unwinds callbacks and pending registration`, async () => {
        const gpu = createGPUInstance() as any;
        try {
            const adapter = await gpu.requestAdapter();
            await turn();
            const input = descriptor(adapter);
            const register = gpu._ticker.register.bind(gpu._ticker);
            let attempt: any;
            gpu._ticker.register = () => {
                attempt = [...gpu._ticker.acquisitions].at(-1);
                const hold = attempt.arena.hold.bind(attempt.arena);
                let count = 0;
                attempt.arena.hold = (value: any) => {
                    hold(value);
                    if (++count === allocation) throw new Error("packing allocation failure");
                    return value;
                };
                register();
            };
            await expect(adapter.requestDevice(input)).rejects.toThrow(
                "packing allocation failure",
            );
            expect(attempt.entered).toBe(false);
            expect(attempt.handles.size).toBe(0);
            expect(attempt.arena.owners.size).toBe(0);
            expect(gpu._ticker.acquisitions.size).toBe(0);
            expect(gpu._ticker._waiting).toBe(0);
            adapter.destroy();
        } finally {
            gpu.destroy();
        }
    });
}

test("failed attempt delayed events cannot reach overlapping replacement devices on one instance", async () => {
    const gpu = createGPUInstance() as any;
    const adapter = await gpu.requestAdapter();
    const other = await gpu.requestAdapter();
    const lib = adapter.lib;
    let stale: any;
    let failedRequest: any;
    let loss: any;
    let error: any;
    adapter.lib = {
        ...lib,
        wgpuAdapterRequestDevice: (_pointer: number, descriptor: number, info: number) => {
            stale = [...gpu._ticker.acquisitions].at(-1);
            const d = new DataView(toArrayBuffer(descriptor, 0, 144));
            const r = new DataView(toArrayBuffer(info, 0, 40));
            const args = ["ptr", "u32", "ptr", "u64", "ptr", "ptr"] as const;
            loss = CFunction({
                ptr: Number(d.getBigUint64(88, true)) as any,
                args: [...args],
                returns: "void",
            });
            error = CFunction({
                ptr: Number(d.getBigUint64(120, true)) as any,
                args: [...args],
                returns: "void",
            });
            failedRequest = CFunction({
                ptr: Number(r.getBigUint64(16, true)) as any,
                args: ["u32", "ptr", "ptr", "u64", "ptr", "ptr"],
                returns: "void",
            });
            return 0n;
        },
    };
    try {
        const pending = adapter.requestDevice(descriptor(adapter));
        await expect(adapter.requestDevice(descriptor(adapter))).rejects.toThrow(
            "Adapter already consumed",
        );
        expect(stale.requestDone).toBe(false);
        failedRequest(3, null, null, 0n, null, null);
        failedRequest.close();
        await expect(pending).rejects.toThrow("WGPU Error (Error)");
        await turn();
        expect(stale.handles.size).toBe(2);
        adapter.lib = lib;
        const [replacement, distinct] = await Promise.all([
            adapter.requestDevice(descriptor(adapter)),
            other.requestDevice(descriptor(other)),
        ]);
        await turn();
        let replacementErrors = 0;
        let distinctErrors = 0;
        replacement.onuncapturederror = () => {
            replacementErrors++;
        };
        distinct.onuncapturederror = () => {
            distinctErrors++;
        };
        let replacementLost = false;
        replacement.lost.then(() => {
            replacementLost = true;
        });
        error(null, 1, null, 0n, null, null);
        loss(null, 4, null, 0n, null, null);
        error.close();
        loss.close();
        await turn();
        expect(stale.handles.size).toBe(0);
        expect(replacementErrors).toBe(0);
        expect(distinctErrors).toBe(0);
        expect(replacementLost).toBe(false);
        const buffer = distinct.createBuffer({ size: 4, usage: 0 });
        gpu._ticker.processEvents();
        expect(distinctErrors).toBe(1);
        expect(replacementErrors).toBe(0);
        buffer.destroy();
        distinct.destroy();
        await distinct.lost;
        await turn();
        expect(replacementLost).toBe(false);
        expect(replacement._acquisition.handles.size).toBe(2);
        replacement.destroy();
        await replacement.lost;
        await turn();
        expect(gpu._ticker.acquisitions.size).toBe(0);
        expect(gpu._ticker._waiting).toBe(0);
    } finally {
        adapter.destroy();
        other.destroy();
        gpu.destroy();
    }
});

test("actual successful native device is rolled back when queue construction throws", async () => {
    const gpu = createGPUInstance() as any;
    const adapter = await gpu.requestAdapter();
    await turn();
    const lib = adapter.lib;
    let destroys = 0;
    let releases = 0;
    let attempt: any;
    adapter.lib = {
        ...lib,
        wgpuDeviceGetQueue: () => {
            attempt = [...gpu._ticker.acquisitions].at(-1);
            throw new Error("queue construction failed");
        },
        wgpuDeviceDestroy: (pointer: number) => {
            destroys++;
            return lib.wgpuDeviceDestroy(pointer);
        },
        wgpuDeviceRelease: (pointer: number) => {
            releases++;
            return lib.wgpuDeviceRelease(pointer);
        },
    };
    try {
        await expect(adapter.requestDevice(descriptor(adapter))).rejects.toThrow(
            "queue construction failed",
        );
        await turn();
        gpu._ticker.processEvents();
        await turn();
        expect(destroys).toBe(1);
        expect(releases).toBe(1);
        expect(attempt.requestDone).toBe(true);
        expect(attempt.lossDone).toBe(true);
        expect(attempt.handles.size).toBe(0);
        expect(gpu._ticker.acquisitions.size).toBe(0);
    } finally {
        adapter.destroy();
        gpu.destroy();
    }
});

test("actual loader refuses explicit native library overrides", async () => {
    const native = await loadNative();
    expect(() => native.createGPUInstance("/not-the-peer/library.dylib")).toThrow(
        "library override",
    );
    await expect(native.setupGlobals({ libPath: "/not-the-peer/library.dylib" })).rejects.toThrow(
        "library override",
    );
});

test("actual adapter success owns a valid handle and drains its request once", async () => {
    const gpu = createGPUInstance() as any;
    const pending = gpu.requestAdapter();
    const attempt = [...gpu._ticker.acquisitions][0] as any;
    const handle = attempt.handles.get("request");
    const close = handle.close.bind(handle);
    let closes = 0;
    handle.close = () => {
        closes++;
        close();
    };
    const adapter = await pending;
    try {
        expect(adapter.adapterPtr).toBeGreaterThan(0);
        await turn();
        expect(closes).toBe(1);
        expect(attempt.handles.size).toBe(0);
        expect(gpu._ticker.acquisitions.size).toBe(0);
        expect(gpu._ticker._waiting).toBe(0);
    } finally {
        adapter.destroy();
        gpu.destroy();
    }
});

test("actual native destroy delivers loss and drains its independent handles", async () => {
    const gpu = createGPUInstance() as any;
    try {
        const adapter = await gpu.requestAdapter();
        const device = await adapter.requestDevice(descriptor(adapter));
        await turn();
        const attempt = device._acquisition;
        expect(attempt.handles.has("request")).toBe(false);
        expect(attempt.handles.size).toBe(2);
        device.destroy();
        await device.lost;
        await turn();
        expect(attempt.lossDone).toBe(true);
        expect(attempt.handles.size).toBe(0);
        expect(gpu._ticker.acquisitions.size).toBe(0);
        adapter.destroy();
    } finally {
        gpu.destroy();
    }
});
