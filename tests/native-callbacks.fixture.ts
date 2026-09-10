import * as ffi from "bun:ffi";
import { mock } from "bun:test";
import assert from "node:assert/strict";

// Isolated process: the exported constructor still creates real Bun callbacks.
// Failure before construction, or a constructed but unusable handle, must unwind
// every earlier handle without ever forwarding a null callback to Dawn.
const mode = process.env.CALLBACK_MODE!;
const target = Number(process.env.CALLBACK_TARGET ?? 0);
const real = ffi.JSCallback;
const handles: { handle: InstanceType<typeof real>; closes: number }[] = [];
let constructed = 0;
let armed = false;
let active = 0;
mock.module("bun:ffi", () => ({
    ...ffi,
    JSCallback: new Proxy(real, {
        construct(_target, [fn, options]) {
            const index = armed ? ++constructed : 0;
            if (index === target && mode === "before")
                throw new Error("callback construction failure");
            const handle = new real((...args: any[]) => {
                active++;
                try {
                    return fn(...args);
                } finally {
                    active--;
                }
            }, options);
            if (!armed) return handle;
            const record = { handle, closes: 0 };
            handles.push(record);
            return new Proxy(handle, {
                get(object, key) {
                    if (key === "ptr" && index === target && mode === "null") return null;
                    if (key === "close")
                        return () => {
                            assert.equal(active, 0, "close outside native/callback stack");
                            record.closes++;
                            handle.close();
                        };
                    return Reflect.get(object, key);
                },
            });
        },
    }),
}));
const { loadNative } = await import("../bin/bun-native");
const { BASE_FEATURES, deviceLimits } = await import("../src/engine/runtime/gpu");
const native = await loadNative();
const gpu = native.createGPUInstance() as any;
const turn = async () => {
    await Promise.resolve();
    await Promise.resolve();
};
const adapterSite = process.env.CALLBACK_SITE === "adapter";
let adapter: any;
let entries = 0;
let returns = 0;
let destroys = 0;
let releases = 0;
let queueReleases = 0;
let attempt: any;
try {
    if (!adapterSite) adapter = await gpu.requestAdapter();
    await turn();
    const owner = adapterSite ? gpu : adapter;
    const lib = owner.lib;
    const name = adapterSite ? "wgpuInstanceRequestAdapter" : "wgpuAdapterRequestDevice";
    owner.lib = {
        ...lib,
        [name]: (...args: any[]) => {
            entries++;
            active++;
            try {
                const result = lib[name](...args);
                returns++;
                return result;
            } finally {
                active--;
            }
        },
        wgpuDeviceDestroy: (pointer: number) => {
            destroys++;
            return lib.wgpuDeviceDestroy(pointer);
        },
        wgpuDeviceRelease: (pointer: number) => {
            releases++;
            return lib.wgpuDeviceRelease(pointer);
        },
        wgpuQueueRelease: (pointer: number) => {
            queueReleases++;
            return lib.wgpuQueueRelease(pointer);
        },
    };
    const register = gpu._ticker.register.bind(gpu._ticker);
    gpu._ticker.register = () => {
        attempt = [...gpu._ticker.acquisitions].at(-1);
        register();
    };
    const descriptor = adapterSite
        ? undefined
        : {
              requiredFeatures: [...BASE_FEATURES],
              requiredLimits: deviceLimits(adapter.limits),
              label: "callback constructor",
              defaultQueue: { label: "callback queue" },
          };
    armed = true;
    const outcome = await (adapterSite
        ? gpu.requestAdapter()
        : adapter.requestDevice(descriptor)
    ).then(
        () => null,
        (error: Error) => error,
    );
    assert(outcome instanceof Error, "constructor failure rejects request");
    await turn();
    gpu._ticker.processEvents();
    await turn();
    assert.equal(constructed, target, "target constructor reached");
    assert.equal(entries, adapterSite || target <= 3 ? 0 : 1, "native entry population");
    assert.equal(returns, entries, "actual native returns");
    assert.equal(destroys, entries, "unpublished device destroyed");
    assert.equal(releases, entries, "unpublished device released");
    assert.equal(queueReleases, entries, "partial constructor queue released");
    assert(
        handles.every((record) => record.closes === 1),
        "every constructed handle closes exactly once",
    );
    assert.equal(attempt.arena.owners.size, 0);
    assert.equal(attempt.handles.size, 0);
    assert.equal(gpu._ticker.acquisitions.size, 0);
    assert.equal(gpu._ticker._waiting, 0);
    console.log(
        JSON.stringify({
            mode,
            target,
            site: adapterSite ? "adapter" : "device",
            entries,
            returns,
            destroys,
            releases,
            queueReleases,
            handles: handles.length,
            closes: handles.map((r) => r.closes),
        }),
    );
} finally {
    adapter?.destroy();
    gpu.destroy();
}
