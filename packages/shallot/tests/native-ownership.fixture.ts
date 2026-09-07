import { ptr, toArrayBuffer } from "bun:ffi";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { loadNative } from "../bin/bun-native";
import { BASE_FEATURES, deviceLimits } from "../src/engine/runtime/gpu";

const carrier = resolve(import.meta.dir, "../dist/native.js");
const { createGPUInstance } = await loadNative();
const mode = process.argv[2] ?? "ownership";
const target = process.argv[3] ?? "device";
const shape = process.env.ADAPTER_SHAPE ?? "nonempty";
const omit = process.env.OMIT;
const mutation = process.env.MUTATE ?? "count-low";
const gpu = createGPUInstance() as any;
const copy = (p: number, n: number) => Buffer.from(new Uint8Array(toArrayBuffer(p, 0, n)));
const u64 = (b: Buffer, o: number) => b.readBigUInt64LE(o);
const pointer = (b: Buffer, o: number) => {
    const value = u64(b, o);
    assert.ok(value <= BigInt(Number.MAX_SAFE_INTEGER), "pointer range");
    return Number(value);
};
// Pinned from the intact pack trace, correlated to the independent native ABI at entry.
const deviceInventory: [string, number, string][] = [
    ["outer", 144, "buffer"],
    ["label-view", 16, "buffer"],
    ["label", 18, "utf8"],
    ["features", 12, "buffer"],
    ["limits", 168, "buffer"],
    ["queue", 24, "buffer"],
    ["queue-label-view", 16, "buffer"],
    ["inline-child", 16, "utf8"],
    ["loss-info", 40, "buffer"],
    ["error-info", 32, "buffer"],
    ["request-info", 40, "buffer"],
];
const adapterInventory: [string, number, string][] =
    shape === "nonempty"
        ? [
              ["outer", 32, "buffer"],
              ["request-info", 40, "buffer"],
          ]
        : [["request-info", 40, "buffer"]];
let calls = 0,
    returns = 0,
    allocationCount = 0,
    pressures = 0;
let active: any = null,
    site = "",
    guard = false,
    violation: unknown;
let records: { field: string; p: number; n: number; owner: WeakRef<object> }[] = [];
const gcRecords: { p: number; n: number }[] = [];
function checkOwners() {
    if (!active || guard) return;
    guard = true;
    try {
        for (const record of records)
            assert.ok(
                active.arena.owners.has(record.owner.deref()),
                `${site}.${record.field} ownership`,
            );
    } catch (error) {
        violation ??= error;
        throw error;
    } finally {
        guard = false;
    }
}
function observe(value: ArrayBuffer | Uint8Array, kind: string) {
    if (!active || guard) return value;
    guard = true;
    try {
        const inventory = site === "adapter" ? adapterInventory : deviceInventory;
        const row = inventory[records.length];
        assert.ok(row, `${site} unknown allocation ${records.length}`);
        const [field, n, expectedKind] = row;
        assert.equal(kind, expectedKind, `${site}.${field} allocation kind`);
        assert.equal(value.byteLength, n, `${site}.${field} allocation width`);
        const p = ptr(value);
        assert.ok(!records.some((r) => r.p === p), `${site} duplicate allocation`);
        records.push({ field, p, n, owner: new WeakRef(value) });
        allocationCount++;
    } catch (error) {
        violation ??= error;
        throw error;
    } finally {
        guard = false;
    }
    return value;
}
function pressure() {
    const snapshots = gcRecords.map(({ p, n }) => ({ p, bytes: copy(p, n) }));
    Bun.gc(true);
    pressures++;
    for (const { p, bytes } of snapshots)
        assert.deepEqual(copy(p, bytes.length), bytes, "GC bytes changed during packing");
}
assert.ok(gpu._ticker.acquisitions instanceof Set, "instance acquisition observer prerequisite");
const register = gpu._ticker.register.bind(gpu._ticker);
gpu._ticker.register = () => {
    const attempt = [...gpu._ticker.acquisitions].at(-1) as any;
    assert.ok(attempt?.arena?.owners instanceof Set, "real arena prerequisite");
    site = attempt.lossDone ? "adapter" : "device";
    records = [];
    gcRecords.length = 0;
    const hold = attempt.arena.hold.bind(attempt.arena);
    if (mode === "gc") {
        attempt.arena.hold = (value: any) => {
            pressure();
            hold(value);
            gcRecords.push({ p: ptr(value), n: value.byteLength });
            allocationCount++;
            return value;
        };
    } else {
        active = attempt;
        attempt.arena.hold = (value: any) => {
            const record = records.find((r) => r.owner.deref() === value);
            if (!(site === target && record?.field === omit)) hold(value);
            return value;
        };
    }
    return register();
};
if (mode !== "gc") {
    const Original = ArrayBuffer;
    globalThis.ArrayBuffer = new Proxy(Original, {
        construct(ctor, args) {
            checkOwners();
            return observe(Reflect.construct(ctor, args), "buffer");
        },
    });
    const encode = TextEncoder.prototype.encode;
    TextEncoder.prototype.encode = function (text) {
        checkOwners();
        return observe(encode.call(this, text), "utf8") as Uint8Array<ArrayBuffer>;
    };
    for (const key of Object.getOwnPropertyNames(DataView.prototype).filter((key) =>
        key.startsWith("set"),
    )) {
        const original = (DataView.prototype as any)[key];
        (DataView.prototype as any)[key] = function (...args: any[]) {
            checkOwners();
            return original.apply(this, args);
        };
    }
}
function address(field: string) {
    const record = records.find((r) => r.field === field);
    assert.ok(record, `${site}.${field} allocation missing`);
    return BigInt(record.p);
}
function instrument(lib: any, name: string, acquisitionSite: string) {
    const original = lib[name];
    return {
        ...lib,
        [name]: (handle: number, arg: number, cb: number) => {
            calls++;
            site = acquisitionSite;
            checkOwners();
            // Observer allocations and native readback are not pack allocations.
            guard = true;
            try {
                const inventory = site === "adapter" ? adapterInventory : deviceInventory;
                if (mode !== "gc") {
                    assert.equal(records.length, inventory.length, `${site} complete inventory`);
                    assert.equal(
                        BigInt(cb),
                        address("request-info"),
                        `${site} callback-info identity`,
                    );
                    assert.equal(
                        BigInt(arg ?? 0),
                        site === "adapter" && shape !== "nonempty" ? 0n : address("outer"),
                        `${site} outer identity`,
                    );
                }
                const outer = arg ? copy(arg, site === "adapter" ? 32 : 144) : null;
                const callback = copy(cb, 40);
                const graph: [string, number, Buffer][] = [["request-info", cb, callback]];
                if (outer) graph.push(["outer", arg, outer]);
                let fp = 0,
                    lp = 0;
                if (site === "device") {
                    assert.ok(outer);
                    if (mode !== "gc") {
                        assert.equal(
                            u64(outer, 32),
                            address("features"),
                            "feature pointer identity",
                        );
                        assert.equal(u64(outer, 40), address("limits"), "limits pointer identity");
                    }
                    fp = pointer(outer, 32);
                    lp = pointer(outer, 40);
                    graph.push(["features", fp, copy(fp, 12)], ["limits", lp, copy(lp, 168)]);
                    for (const [offset, field, text] of [
                        [8, "label", "ownership boundary"],
                        [56, "inline-child", "production queue"],
                    ] as const) {
                        if (mode !== "gc")
                            assert.equal(
                                u64(outer, offset),
                                address(field),
                                `${field} pointer identity`,
                            );
                        assert.equal(
                            u64(outer, offset + 8),
                            BigInt(text.length),
                            `${field} string size`,
                        );
                        const p = pointer(outer, offset);
                        assert.equal(
                            copy(p, text.length).toString(),
                            text,
                            `${field} string value`,
                        );
                        graph.push([field, p, copy(p, text.length)]);
                    }
                }
                if (mode === "layout" && site === target) {
                    const [field, bit] = mutation.split("-");
                    const view = new DataView(
                        toArrayBuffer(
                            field === "feature" ? fp : field === "size" ? lp : arg,
                            0,
                            field === "feature" ? 12 : field === "size" ? 168 : 144,
                        ),
                    );
                    const offset =
                        field === "pointer"
                            ? 32
                            : field === "count"
                              ? 24
                              : field === "size"
                                ? 96
                                : 0;
                    if (field === "feature")
                        view.setUint32(
                            offset,
                            view.getUint32(offset, true) ^ (bit === "high" ? 0x10000 : 1),
                            true,
                        );
                    else
                        view.setBigUint64(
                            offset,
                            view.getBigUint64(offset, true) ^ (bit === "high" ? 1n << 40n : 1n),
                            true,
                        );
                }
                // Read malformed producer output before any native forwarding. Never dereference a mutated pointer.
                const current = outer ? copy(arg, outer.length) : null;
                if (current) assert.equal(u64(current, 0), 0n, `${site} outer chain`);
                assert.equal(u64(callback, 0), 0n, `${site} request chain`);
                assert.equal(callback.readUInt32LE(8), 2, `${site} request mode`);
                for (const offset of [24, 32])
                    assert.equal(u64(callback, offset), 0n, `${site} request userdata ${offset}`);
                if (mode !== "gc")
                    assert.equal(
                        u64(callback, 16),
                        BigInt(active.handles.get("request").ptr),
                        `${site} request callback identity`,
                    );
                else assert.notEqual(u64(callback, 16), 0n, `${site} request callback`);
                if (site === "adapter" && current) {
                    assert.equal(current.readUInt32LE(8), 2, "core feature level");
                    assert.equal(current.readUInt32LE(12), 2, "high-performance");
                    assert.equal(current.readUInt32LE(16), 0, "not fallback");
                    assert.equal(u64(current, 24), 0n, "surface");
                } else if (site === "device") {
                    assert.ok(current);
                    assert.equal(u64(current, 24), 3n, "feature count");
                    assert.equal(u64(current, 32), BigInt(fp), "feature pointer identity");
                    assert.equal(u64(current, 40), BigInt(lp), "limits pointer identity");
                    const features = copy(fp, 12),
                        limits = copy(lp, 168);
                    assert.deepEqual(
                        [0, 4, 8].map((o) => features.readUInt32LE(o)),
                        [9, 12, 11],
                        "u32 feature values",
                    );
                    assert.equal(u64(limits, 0), 0n, "limits chain");
                    assert.equal(limits.readUInt32LE(52), 10, "storage binding floor");
                    assert.equal(
                        u64(limits, 72),
                        BigInt(expectedLimits.maxStorageBufferBindingSize),
                        "u64 storage binding size",
                    );
                    assert.equal(
                        u64(limits, 96),
                        BigInt(expectedLimits.maxBufferSize),
                        "u64 buffer size",
                    );
                    for (const o of [48, 72, 112])
                        assert.equal(u64(current, o), 0n, `nested chain ${o}`);
                    assert.equal(current.readUInt32LE(80), 2, "loss mode");
                    for (const [o, key] of [
                        [88, "loss"],
                        [120, "error"],
                    ] as const) {
                        if (mode !== "gc")
                            assert.equal(
                                u64(current, o),
                                BigInt(active.handles.get(key).ptr),
                                `${key} callback identity`,
                            );
                        else assert.notEqual(u64(current, o), 0n, `${key} callback`);
                    }
                    for (const o of [96, 104, 128, 136])
                        assert.equal(u64(current, o), 0n, `userdata ${o}`);
                    if (mode !== "gc") {
                        for (const [field, offset, n] of [
                            ["label-view", 8, 16],
                            ["queue", 48, 24],
                            ["queue-label-view", 56, 16],
                            ["loss-info", 72, 40],
                            ["error-info", 112, 32],
                        ] as const)
                            assert.deepEqual(
                                copy(Number(address(field)), n),
                                current.subarray(offset, offset + n),
                                `${field} inline copy`,
                            );
                    }
                }
                if (mode === "layout" && site === target) {
                    console.log({
                        diagnostic: "semantic assertion absent",
                        mutation,
                        calls,
                        returns,
                    });
                    process.exit(0); // Non-forwarding assertion-deletion control.
                }
                if (mode === "gc" && site === target) pressure();
                if (mode === "gc")
                    for (const [field, p, bytes] of graph)
                        assert.deepEqual(
                            copy(p, bytes.length),
                            bytes,
                            `${site}.${field} changed before native consumption`,
                        );
                guard = false;
                checkOwners();
                guard = true;
                if (omit === "premature-release" && site === target) active.arena.release();
                guard = false;
                checkOwners();
                guard = true;
                const result = original(handle, arg, cb);
                returns++;
                if (omit === "release-on-return" && site === target) active.arena.release();
                guard = false;
                checkOwners();
                guard = true;
                if (mode === "gc")
                    for (const [field, p, bytes] of graph)
                        assert.deepEqual(
                            copy(p, bytes.length),
                            bytes,
                            `${site}.${field} changed through native return`,
                        );
                console.log({
                    site,
                    inventory:
                        mode === "gc" ? "GC numeric snapshots only" : records.map((r) => r.field),
                    nativeReturned: true,
                });
                return result;
            } finally {
                active = null;
                guard = false;
            }
        },
    };
}
let expectedLimits: Record<string, number>;
try {
    gpu.lib = instrument(gpu.lib, "wgpuInstanceRequestAdapter", "adapter");
    const adapter = await gpu.requestAdapter(
        shape === "nonempty"
            ? { featureLevel: "core", powerPreference: "high-performance" }
            : shape === "null"
              ? null
              : undefined,
    );
    if (violation) throw violation;
    assert.ok(adapter, "actual adapter");
    active = null;
    expectedLimits = deviceLimits(adapter.limits);
    void adapter.features;
    adapter.lib = instrument(adapter.lib, "wgpuAdapterRequestDevice", "device");
    const device = await adapter.requestDevice({
        label: "ownership boundary",
        requiredFeatures: [...BASE_FEATURES],
        requiredLimits: expectedLimits,
        defaultQueue: { label: "production queue" },
    });
    if (violation) throw violation;
    device.destroy();
    adapter.destroy();
    await Promise.resolve();
    assert.equal(calls, 2);
    assert.equal(returns, 2);
    assert.equal(gpu._ticker.acquisitions.size, 0, "attempts drained after actual destroy/loss");
    assert.equal(gpu._ticker._waiting, 0, "pending requests drained");
    assert.equal(
        allocationCount,
        deviceInventory.length + adapterInventory.length,
        "exact allocation population",
    );
    if (mode === "gc") assert.ok(pressures > 10, "GC pressure population");
    console.log({
        mode,
        target,
        shape,
        omit,
        mutation,
        carrier,
        sha256: createHash("sha256")
            .update(await Bun.file(carrier).bytes())
            .digest("hex"),
        calls,
        returns,
        allocationCount,
        pressures,
        pass: true,
    });
} catch (error) {
    console.error(violation ?? error);
    console.log({
        mode,
        target,
        shape,
        omit,
        mutation,
        calls,
        returns,
        allocationCount,
        pass: false,
    });
    process.exitCode = 1;
} finally {
    active = null;
    gpu.destroy();
}
process.exit(process.exitCode ?? 0);
