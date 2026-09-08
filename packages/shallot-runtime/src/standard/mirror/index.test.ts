import { afterEach, expect, spyOn, test } from "bun:test";
import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Compute, type State } from "../../engine";
import { observeDevice } from "../../engine/runtime/gpu";
import { Mirror, mirror } from ".";

afterEach(() => {
    Mirror.reset();
    Object.assign(Compute, { root: undefined });
});

test("Mirror retains the schema-carrying source while copying from its raw twin", () => {
    const raw = { size: 16 } as GPUBuffer;
    const source = {
        resourceType: "buffer",
        dataType: d.arrayOf(d.u32, 4),
    } as TgpuBuffer<d.WgslArray<d.U32>>;
    Object.assign(Compute, {
        root: { unwrap: (value: unknown) => (value === source ? raw : value) },
    });

    const m = mirror(source);
    expect(m.source).toBe(source);
    expect(m.size).toBe(16);
    m.dispose();
});

// `Mirror.flush`'s own ring-growth branch is the site that carries `lazy: true` (index.ts:119) — this
// drives that branch for real (an empty ring, real backpressure) and reads the descriptor
// `device.createBuffer` actually received, rather than restating the literal against itself the way
// `profile/index.test.ts`'s synthetic `lazy: true` call does for the profiler side of the seam.
test("Mirror's ring growth marks its staging slot lazy", () => {
    const raw = { size: 16 } as GPUBuffer;
    const created: (GPUBufferDescriptor & { lazy?: boolean })[] = [];
    const stagingSlot = {
        mapAsync: () => new Promise<void>(() => {}), // never resolves — the test only needs the encode-time descriptor
        destroy: () => {},
    } as unknown as GPUBuffer;
    const device = {
        createCommandEncoder: () => ({ copyBufferToBuffer: () => {}, finish: () => ({}) }),
        createBuffer: (desc: GPUBufferDescriptor) => {
            created.push(desc as GPUBufferDescriptor & { lazy?: boolean });
            return stagingSlot;
        },
        queue: { submit: () => {} },
    } as unknown as GPUDevice;
    const prevDevice = Compute.device;
    Object.assign(Compute, { device, frame: 0 });

    const m = mirror(raw);
    try {
        // the ring starts empty, so this first flush is real backpressure growth, not a warm-ring hit
        Mirror.flush({ time: { fixedTick: 0 } } as unknown as State);
    } finally {
        m.dispose();
        Object.assign(Compute, { device: prevDevice });
    }

    expect(created.length).toBe(1);
    expect(created[0].label).toBe("mirror-staging");
    expect(created[0].lazy).toBe(true);
});

function readbackDevice() {
    const loss = Promise.withResolvers<GPUDeviceLostInfo>();
    const maps: ReturnType<typeof Promise.withResolvers<void>>[] = [];
    const slots: { destroyed: number; reads: number; bytes: ArrayBuffer }[] = [];
    const calls = { encoders: 0, submits: 0, copies: 0 };
    const device = {
        lost: loss.promise,
        createCommandEncoder: () => {
            calls.encoders++;
            return {
                copyBufferToBuffer: () => calls.copies++,
                finish: () => ({}),
            };
        },
        createBuffer: () => {
            const slot = { destroyed: 0, reads: 0, bytes: new ArrayBuffer(16) };
            slots.push(slot);
            return {
                mapAsync: () => {
                    const map = Promise.withResolvers<void>();
                    maps.push(map);
                    return map.promise;
                },
                getMappedRange: () => {
                    slot.reads++;
                    return slot.bytes;
                },
                unmap: () => {},
                destroy: () => slot.destroyed++,
            };
        },
        queue: { submit: () => calls.submits++ },
    } as unknown as GPUDevice;
    const diagnostics: string[] = [];
    observeDevice(device, (message) => diagnostics.push(message));
    return { device, loss, maps, slots, calls, diagnostics };
}

test("loss terminates multiple mirrors without publishing stale maps or touching a replacement", async () => {
    const old = readbackDevice();
    const next = readbackDevice();
    const state = { time: { fixedTick: 7 } } as unknown as State;
    const raw = { size: 16 } as GPUBuffer;
    Object.assign(Compute, { device: old.device, frame: 1 });
    const a = mirror(raw);
    const b = mirror(raw);
    Mirror.flush(state);
    expect(old.maps).toHaveLength(2);
    old.loss.resolve({ reason: "unknown", message: "test loss" } as GPUDeviceLostInfo);
    await old.loss.promise;
    old.maps[0].resolve();
    old.maps[1].reject(new Error("lost map"));
    await Promise.allSettled(old.maps.map((map) => map.promise));
    expect(a.snapshot).toBeNull();
    expect(b.snapshot).toBeNull();
    const calls = { ...old.calls };
    for (let i = 0; i < 4; i++) Mirror.flush(state);
    expect(old.calls).toEqual(calls);
    expect(old.maps).toHaveLength(2);
    expect(old.slots.every((slot) => slot.reads === 0)).toBe(true);
    expect(old.diagnostics).toHaveLength(1);

    Object.assign(Compute, { device: next.device, frame: 2 });
    const replacement = mirror(raw);
    Mirror.flush(state);
    expect(next.maps).toHaveLength(1);
    next.maps[0].resolve();
    await next.maps[0].promise;
    expect(replacement.snapshot?.frame).toBe(2);
    expect(a.snapshot).toBeNull();
    expect(b.snapshot).toBeNull();
});

test("loss before first flush allocates nothing while another device remains live", async () => {
    const owner = readbackDevice();
    Object.assign(Compute, { device: owner.device, frame: 0 });
    mirror({ size: 16 } as GPUBuffer);
    owner.loss.resolve({ reason: "destroyed", message: "before flush" } as GPUDeviceLostInfo);
    await owner.loss.promise;
    Mirror.flush({ time: { fixedTick: 0 } } as unknown as State);
    expect(owner.slots).toHaveLength(0);
    expect(owner.calls).toEqual({ encoders: 0, copies: 0, submits: 0 });
});

test("ordinary rejection is loud and the same ring recovers without allocating", async () => {
    const owner = readbackDevice();
    const state = { time: { fixedTick: 0 } } as unknown as State;
    Object.assign(Compute, { device: owner.device, frame: 0 });
    const m = mirror({ size: 16, label: "recoverable" } as GPUBuffer, { ring: 1 });
    const errors = spyOn(console, "error").mockImplementation(() => {});
    try {
        for (let i = 0; i < 2; i++) {
            Mirror.flush(state);
            owner.maps[i].reject(new Error("ordinary map refusal"));
            await Promise.allSettled([owner.maps[i].promise]);
            expect(errors).toHaveBeenCalledTimes(i + 1);
            expect(m.snapshot).toBeNull();
        }
        Object.assign(Compute, { frame: 3 });
        Mirror.flush(state);
        new Uint8Array(owner.slots[0].bytes)[0] = 73;
        owner.maps[2].resolve();
        await owner.maps[2].promise;
        expect(owner.slots).toHaveLength(1);
        expect(m.snapshot?.frame).toBe(3);
        expect(new Uint8Array(m.snapshot!.bytes)[0]).toBe(73);
        expect(owner.slots[0].destroyed).toBe(0);
        expect(errors.mock.calls[0][0]).toContain("recoverable");
    } finally {
        errors.mockRestore();
    }
});

test("out-of-order success preserves the newest bytes and recycles the older slot", async () => {
    const owner = readbackDevice();
    const state = { time: { fixedTick: 1 } } as unknown as State;
    Object.assign(Compute, { device: owner.device, frame: 1 });
    const m = mirror({ size: 16 } as GPUBuffer);
    Mirror.flush(state);
    Object.assign(Compute, { frame: 2 });
    Mirror.flush(state);
    new Uint8Array(owner.slots[0].bytes)[0] = 11;
    new Uint8Array(owner.slots[1].bytes)[0] = 22;
    owner.maps[1].resolve();
    await owner.maps[1].promise;
    owner.maps[0].resolve();
    await owner.maps[0].promise;
    expect(m.snapshot?.frame).toBe(2);
    expect(new Uint8Array(m.snapshot!.bytes)[0]).toBe(22);
    Object.assign(Compute, { frame: 3 });
    Mirror.flush(state);
    expect(owner.slots).toHaveLength(2);
    expect(owner.maps).toHaveLength(3);
});

for (const retirement of ["dispose", "reset", "replacement", "loss"] as const) {
    test(`${retirement} makes success and rejection harmless across multiple mirrors`, async () => {
        const old = readbackDevice();
        const next = retirement === "replacement" ? readbackDevice() : old;
        const state = { time: { fixedTick: 1 } } as unknown as State;
        Object.assign(Compute, { device: old.device, frame: 1 });
        const a = mirror({ size: 16 } as GPUBuffer);
        const b = mirror({ size: 16 } as GPUBuffer);
        Mirror.flush(state);
        const errors = spyOn(console, "error").mockImplementation(() => {});
        try {
            if (retirement === "dispose") {
                a.dispose();
                b.dispose();
            }
            if (retirement === "reset") Mirror.reset();
            if (retirement === "loss") {
                old.loss.resolve({ reason: "unknown", message: "retired" } as GPUDeviceLostInfo);
                await old.loss.promise;
            }
            Object.assign(Compute, { device: next.device, frame: 2 });
            old.maps[0].resolve();
            old.maps[1].reject(new Error("retired map"));
            await Promise.allSettled(old.maps.map((map) => map.promise));
            expect(a.snapshot).toBeNull();
            expect(b.snapshot).toBeNull();
            expect(old.slots.every((slot) => slot.reads === 0 && slot.destroyed === 1)).toBe(true);
            expect(errors).not.toHaveBeenCalled();
            if (retirement !== "loss") {
                const live = mirror({ size: 16 } as GPUBuffer);
                Mirror.flush(state);
                expect(next.maps.length).toBe(next === old ? 3 : 1);
                next.maps.at(-1)!.resolve();
                await next.maps.at(-1)!.promise;
                expect(live.snapshot?.frame).toBe(2);
            }
        } finally {
            errors.mockRestore();
        }
    });
}
