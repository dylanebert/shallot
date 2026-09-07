import * as ffi from "bun:ffi";
import { mock } from "bun:test";
import assert from "node:assert/strict";

const mode = process.env.CALLBACK_MODE!;
const failure = new Error(`controlled ${mode} failure`);
const message = Buffer.from("valid native callback message");
const stringify = Buffer.prototype.toString;
let decoding = false;
Buffer.prototype.toString = function (...args: any[]) {
    if (decoding) throw failure;
    return stringify.apply(this, args as any);
};
const real = ffi.JSCallback;
let armed = false;
let count = 0;
let active = 0;
const records: { closes: number }[] = [];
mock.module("bun:ffi", () => ({
    ...ffi,
    JSCallback: new Proxy(real, {
        construct(_target, [fn, options]) {
            const index = armed ? ++count : 0;
            const record = { closes: 0 };
            if (armed) records.push(record);
            const handle = new real((...args: any[]) => {
                const inject =
                    (mode === "adapter-decode" && index === 1) ||
                    (mode === "device-decode" && index === 3) ||
                    (mode === "loss-decode" && index === 2) ||
                    (mode === "error-decode" && index === 1);
                if (inject) {
                    args[2] = ffi.ptr(message);
                    args[3] = BigInt(message.length);
                }
                active++;
                decoding = inject;
                try {
                    return fn(...args);
                } finally {
                    active--;
                    decoding = false;
                }
            }, options);
            return new Proxy(handle, {
                get(object, key) {
                    if (key === "close")
                        return () => {
                            assert.equal(active, 0, "no close on native callback stack");
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
let diagnostic: unknown;
process.on("uncaughtException", (error) => {
    if (error !== failure) {
        console.error(error);
        process.exit(2);
    }
    diagnostic = error;
});
let adapter: any;
let device: any;
let attempt: any;
let releases = 0;
let destroys = 0;
let entries = 0;
let returns = 0;
try {
    if (mode !== "adapter-decode") adapter = await gpu.requestAdapter();
    await turn();
    const owner = adapter ?? gpu;
    const lib = owner.lib;
    const request = adapter ? "wgpuAdapterRequestDevice" : "wgpuInstanceRequestAdapter";
    const release = adapter ? "wgpuDeviceRelease" : "wgpuAdapterRelease";
    owner.lib = {
        ...lib,
        [request]: (...args: any[]) => {
            entries++;
            if (mode === "queued-error") {
                const view = new DataView(ffi.toArrayBuffer(args[1], 0, 144));
                const error = ffi.CFunction({
                    ptr: Number(view.getBigUint64(120, true)) as any,
                    args: ["ptr", "u32", "ptr", "u64", "ptr", "ptr"],
                    returns: "void",
                });
                error(null, 1, ffi.ptr(message), BigInt(message.length), null, null);
                error.close();
            }
            active++;
            try {
                const value = lib[request](...args);
                returns++;
                return value;
            } finally {
                active--;
            }
        },
        [release]: (pointer: number) => {
            releases++;
            return lib[release](pointer);
        },
        wgpuDeviceDestroy: (pointer: number) => {
            destroys++;
            return lib.wgpuDeviceDestroy(pointer);
        },
    };
    const register = gpu._ticker.register.bind(gpu._ticker);
    gpu._ticker.register = () => {
        attempt = [...gpu._ticker.acquisitions].at(-1);
        register();
    };
    if (mode === "queued-error" || mode === "live-error") {
        (native.globalConstructors.GPUDevice as any).prototype.handleUncapturedError = () => {
            throw failure;
        };
    }
    if (mode === "loss-delivery") {
        (native.globalConstructors.GPUDevice as any).prototype.handleDeviceLost = () => {
            throw failure;
        };
    }
    armed = true;
    const result = await (adapter
        ? adapter.requestDevice({
              requiredFeatures: [...BASE_FEATURES],
              requiredLimits: deviceLimits(adapter.limits),
          })
        : gpu.requestAdapter()
    ).then(
        (value: any) => ({ value, error: null }),
        (error: Error) => ({ value: null, error }),
    );
    const rejected = ["adapter-decode", "device-decode", "queued-error"].includes(mode);
    if (rejected) assert.equal(result.error, failure, "original request failure preserved");
    else {
        assert.equal(result.error, null);
        device = result.value;
        if (mode === "live-error" || mode === "error-decode") {
            let buffer: any;
            try {
                buffer = device.createBuffer({ size: 4, usage: 0 });
                gpu._ticker.processEvents();
            } catch (error) {
                assert.equal(error, failure);
            }
            buffer?.destroy();
            await turn();
            assert.equal(
                attempt.handles.size,
                2,
                "live error does not terminate loss/error lifetime",
            );
        }
        try {
            device.destroy();
        } catch (error) {
            assert.equal(error, failure, "only controlled callback failure can interrupt teardown");
        }
        let lost: any;
        device.lost.then((value: any) => {
            lost = value;
        });
        await turn();
        assert(lost, "terminal lost delivered despite callback throw");
        assert.equal(typeof lost.reason, "string", "terminal lost reason");
        if (mode === "loss-decode") assert.match(lost.message, /decoding failed/);
    }
    await turn();
    gpu._ticker.processEvents();
    await turn();
    assert.equal(entries, 1);
    assert.equal(returns, 1);
    assert.equal(attempt.handles.size, 0);
    assert.equal(gpu._ticker.acquisitions.size, 0);
    assert.equal(gpu._ticker._waiting, 0);
    if (rejected) {
        assert.equal(releases, 1, "unpublished successful native handle released");
        assert.equal(destroys, adapter ? 1 : 0);
        assert(
            records.every((record) => record.closes === 1),
            "all unpublished callback resources close once",
        );
    } else {
        assert.equal(diagnostic, failure, "original callback exception reported after unwind");
        assert(
            records.slice(0, 3).every((record) => record.closes === 1),
            "terminal acquisition callbacks close once",
        );
    }
    console.log(
        JSON.stringify({
            mode,
            entries,
            returns,
            releases,
            destroys,
            handles: attempt.handles.size,
            pending: gpu._ticker._waiting,
            diagnostic: diagnostic === failure,
            closes: records.map((record) => record.closes),
        }),
    );
    if (!rejected) {
        console.error(failure.message);
        process.exitCode = 1;
    }
} finally {
    adapter?.destroy();
    gpu.destroy();
}
