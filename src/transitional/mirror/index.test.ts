import { expect, test } from "bun:test";
import { State, Time } from "../../engine";
import { Compute, withCompute, withComputeAsync } from "../../engine/runtime/gpu";
import { Mirror, mirror as makeMirror } from "./index";

interface PendingMap {
    resolve(bytes: readonly number[]): void;
}

interface ControlledBuffer {
    data: ArrayBuffer;
    pending: PendingMap[];
    mapAsync(): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

function restoreProperty(
    target: object,
    key: PropertyKey,
    descriptor: PropertyDescriptor | undefined,
): void {
    if (descriptor === undefined) Reflect.deleteProperty(target, key);
    else Object.defineProperty(target, key, descriptor);
}

async function withControlledDevice(
    body: (state: State, slots: ControlledBuffer[], subject: Mirror) => Promise<void>,
): Promise<void> {
    const usage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
    const mapMode = Object.getOwnPropertyDescriptor(globalThis, "GPUMapMode");
    const slots: ControlledBuffer[] = [];
    const device = {
        createCommandEncoder() {
            return {
                copyBufferToBuffer() {},
                finish() {
                    return {};
                },
            };
        },
        createBuffer() {
            const buffer: ControlledBuffer = {
                data: new ArrayBuffer(4),
                pending: [],
                mapAsync() {
                    let resolvePromise!: () => void;
                    const promise = new Promise<void>((resolve) => {
                        resolvePromise = resolve;
                    });
                    buffer.pending.push({
                        resolve(bytes) {
                            buffer.data = Uint8Array.from(bytes).buffer;
                            resolvePromise();
                        },
                    });
                    return promise;
                },
                getMappedRange() {
                    return buffer.data;
                },
                unmap() {},
                destroy() {},
            };
            slots.push(buffer);
            return buffer as unknown as GPUBuffer;
        },
        queue: { submit() {} },
    } as unknown as GPUDevice;

    Object.defineProperty(globalThis, "GPUBufferUsage", {
        configurable: true,
        value: { MAP_READ: 1, COPY_DST: 2 },
    });
    Object.defineProperty(globalThis, "GPUMapMode", {
        configurable: true,
        value: { READ: 1 },
    });
    const state = new State();
    const source = { size: 4, label: "controlled-source" } as GPUBuffer;
    const compute = {
        device,
        frame: 0,
        adapter: { class: "test", identity: "test" },
        root: { unwrap: (buffer: GPUBuffer) => buffer },
        pending: () => 0,
        sync: async () => {},
        buffers: new Map<string, GPUBuffer>(),
        textures: new Map<string, GPUTexture>(),
        samplers: new Map<string, GPUSampler>(),
        typed: new Map<string, unknown>(),
    };
    state.attachGpu(compute as unknown as Parameters<State["attachGpu"]>[0], (callback) =>
        withCompute(compute, callback),
    );
    Mirror.reset(state);
    const subject = withCompute(compute, () => makeMirror(state, source, { ring: 2 }));
    try {
        await withComputeAsync(compute, () => body(state, slots, subject));
    } finally {
        subject.dispose();
        state.dispose();
        Mirror.reset(state);
        restoreProperty(globalThis, "GPUBufferUsage", usage);
        restoreProperty(globalThis, "GPUMapMode", mapMode);
    }
}

test("an older Mirror map that resolves after a newer one leaves the newer snapshot bytes and tick/frame intact", async () => {
    await withControlledDevice(async (state, slots, subject) => {
        state.step(Time.FIXED_DT);
        Compute.frame = 31;
        Mirror.flush(state);
        state.step(Time.FIXED_DT);
        Compute.frame = 32;
        Mirror.flush(state);

        expect(slots).toHaveLength(2);
        slots[1].pending[0].resolve([2, 2, 2, 2]);
        await Promise.resolve();
        const newer = subject.snapshot;
        expect(newer).not.toBeNull();
        if (!newer) throw new Error("newer Mirror map did not populate a snapshot");
        expect(newer.fixedTick).toBe(2);
        expect(newer.frame).toBe(32);
        expect([...new Uint8Array(newer.bytes)]).toEqual([2, 2, 2, 2]);

        slots[0].pending[0].resolve([1, 1, 1, 1]);
        await Promise.resolve();
        expect(subject.snapshot).toBe(newer);
        expect(newer.fixedTick).toBe(2);
        expect(newer.frame).toBe(32);
        expect([...new Uint8Array(newer.bytes)]).toEqual([2, 2, 2, 2]);
    });
});

test("a Mirror callback resolving after dispose leaves no snapshot or allocated staging buffers", async () => {
    await withControlledDevice(async (state, slots, subject) => {
        state.step(Time.FIXED_DT);
        Mirror.flush(state);
        expect(slots).toHaveLength(1);
        subject.dispose();
        slots[0].pending[0].resolve([4, 3, 2, 1]);
        await Promise.resolve();
        expect(subject.snapshot).toBeNull();
        expect(subject.allocated).toBe(0);
    });
});
