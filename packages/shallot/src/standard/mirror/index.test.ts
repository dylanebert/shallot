import { afterEach, expect, test } from "bun:test";
import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Compute, type State } from "../../engine";
import { Mirror, mirror } from ".";

afterEach(() => {
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
