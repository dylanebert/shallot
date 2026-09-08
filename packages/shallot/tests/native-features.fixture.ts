import { ptr, toArrayBuffer } from "bun:ffi";
import assert from "node:assert/strict";
import { BASE_FEATURES, deviceLimits } from "../../shallot-runtime/src/engine/runtime/gpu";
import { loadNative } from "../../shallot-tooling/bin/bun-native";

const site = process.argv[2] ?? "device";
const mode = process.argv[3] ?? "ownership";
const { createGPUInstance } = await loadNative();
const gpu = createGPUInstance() as any;
const adapter = await gpu.requestAdapter();
assert.ok(adapter);
const device = await adapter.requestDevice({
    requiredFeatures: BASE_FEATURES,
    requiredLimits: deviceLimits(adapter.limits),
});
const subject = site === "device" ? device : site === "adapter" ? adapter : gpu;
const key = site === "instance" ? "wgslLanguageFeatures" : "features";
// Acquisition capability admission may already have read the adapter cache.
subject[site === "instance" ? "_wgslLanguageFeatures" : "_features"] = null;
const name =
    site === "instance"
        ? "wgpuInstanceGetWGSLLanguageFeatures"
        : site === "adapter"
          ? "wgpuAdapterGetFeatures"
          : "wgpuDeviceGetFeatures";
const original = subject.lib[name];
let active = false;
let guard = false;
let forwarded = 0;
let reads = 0;
let releases = 0;
let pressures = 0;
let violation = "";
const records: {
    address: number;
    width: number;
    value: WeakRef<ArrayBuffer>;
    owners?: Set<unknown>;
}[] = [];
function check() {
    if (!active || guard || mode === "gc") return;
    for (const row of records) {
        if (!row.owners?.has(row.value.deref())) {
            violation ||= `${site}.missing-output-owner`;
            if (mode !== "delete-assertion") assert.fail(violation);
        }
    }
}
function pressure() {
    if (mode === "gc") {
        Bun.gc(true);
        pressures++;
    }
}
const Original = ArrayBuffer;
globalThis.ArrayBuffer = new Proxy(Original, {
    construct(ctor, args) {
        check();
        if (active) pressure();
        const value = Reflect.construct(ctor, args);
        if (active && mode !== "gc")
            records.push({
                address: ptr(value),
                width: value.byteLength,
                value: new WeakRef(value),
            });
        return value;
    },
});
const add = Set.prototype.add;
Set.prototype.add = function (value: unknown) {
    if (active && mode !== "gc") {
        const row = records.find((r) => r.value.deref() === value);
        if (row) {
            row.owners = this;
            if ((mode === "omit" || mode === "delete-assertion") && records.indexOf(row) === 1)
                return this;
        }
    }
    return add.call(this, value);
};
const clear = Set.prototype.clear;
Set.prototype.clear = function () {
    if (active && records.some((r) => r.owners === this)) releases++;
    return clear.call(this);
};
for (const method of ["getUint32", "getBigUint64"] as const) {
    const get = DataView.prototype[method];
    (DataView.prototype as any)[method] = function (...args: any[]) {
        if (active && !guard) {
            check();
            pressure();
            if (forwarded) {
                reads++;
                if (mode === "decode-failure") throw Error("injected output decode failure");
            }
        }
        return (get as any).apply(this, args);
    };
}
subject.lib = {
    ...subject.lib,
    [name]: (handle: number, output: number) => {
        check();
        // This guard is independent of the assertion: assertion deletion must not forward bad memory.
        if (violation) throw Error(`nonforwarding safety stop: ${violation}`);
        if (mode !== "gc") {
            assert.deepEqual(
                records.map((r) => r.width),
                [16, site === "instance" ? 128 : 512],
            );
            assert.equal(output, records[0]!.address);
            guard = true;
            try {
                const view = new DataView(toArrayBuffer(output, 0, 16));
                assert.equal(view.getBigUint64(8, true), BigInt(records[1]!.address));
            } finally {
                guard = false;
            }
        }
        pressure();
        if (mode === "native-failure") throw Error("injected native failure");
        if (mode === "status-failure") return 2;
        const result = original(handle, output);
        forwarded++;
        check();
        pressure();
        return result;
    },
};
let failure: unknown;
try {
    active = true;
    const features = subject[key];
    active = false;
    if (violation) throw Error(violation);
    if (mode !== "gc") {
        assert.equal(records.length, 2, "complete output graph");
        assert.equal(releases, 1, "output arena released once");
        assert.ok(
            records.every((r) => r.owners?.size === 0),
            "output unwind",
        );
    }
    if (mode.endsWith("failure")) {
        assert.equal(features.size, 0);
        assert.equal(forwarded, mode === "decode-failure" ? 1 : 0);
    } else {
        assert.equal(forwarded, 1);
        assert.ok(reads > 0, "completed real unpack");
        if (site !== "instance")
            for (const feature of BASE_FEATURES) assert.ok(features.has(feature), feature);
        active = true;
        assert.equal(subject[key], features, "cached identity");
        active = false;
        assert.equal(forwarded, 1, "cached second read does not query");
    }
} catch (error) {
    failure = error;
} finally {
    active = false;
    device.destroy();
    await device.lost;
    adapter.destroy();
    gpu.destroy();
}
console.log({ site, mode, forwarded, reads, releases, pressures, violation });
if (failure) throw failure;
console.log("FEATURE_OUTPUT_PASS");
