import { expect, test } from "bun:test";
import { State, Time } from "../../engine";
import { Compute, observeDevice } from "../../engine/runtime/gpu";
import { Mirror, mirror as makeMirror } from "./index";

function restoreProperty(
    target: object,
    key: PropertyKey,
    descriptor: PropertyDescriptor | undefined,
): void {
    if (descriptor === undefined) Reflect.deleteProperty(target, key);
    else Object.defineProperty(target, key, descriptor);
}

async function withGpuMirror(
    body: (state: State, device: GPUDevice, source: GPUBuffer, mirror: Mirror) => Promise<void>,
): Promise<void> {
    const peerModule = "bun-webgpu";
    const peer = (await import(peerModule)) as { setupGlobals(): Promise<void> };
    await peer.setupGlobals();
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("GPU check has no WebGPU adapter");
    const device = await adapter.requestDevice();
    const deviceDescriptor = Object.getOwnPropertyDescriptor(Compute, "device");
    const frameDescriptor = Object.getOwnPropertyDescriptor(Compute, "frame");
    Object.assign(Compute, { device, frame: 0 });
    observeDevice(device, () => {});
    Mirror.reset();
    const state = new State();
    const source = device.createBuffer({
        label: "mirror-contract-source",
        size: 16,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const subject = makeMirror(source, { ring: 1 });
    try {
        await body(state, device, source, subject);
    } finally {
        subject.dispose();
        source.destroy();
        Mirror.reset();
        state.dispose();
        device.destroy();
        restoreProperty(Compute, "device", deviceDescriptor);
        restoreProperty(Compute, "frame", frameDescriptor);
    }
}

function interceptBufferCreation(
    device: GPUDevice,
    afterCreate: (buffer: GPUBuffer, descriptor: GPUBufferDescriptor) => void,
): () => void {
    const previous = Object.getOwnPropertyDescriptor(device, "createBuffer");
    const createBuffer = device.createBuffer.bind(device);
    Object.defineProperty(device, "createBuffer", {
        configurable: true,
        value(descriptor: GPUBufferDescriptor) {
            const buffer = createBuffer(descriptor);
            afterCreate(buffer, descriptor);
            return buffer;
        },
    });
    return () => restoreProperty(device, "createBuffer", previous);
}

function trackMirrorMaps(device: GPUDevice): { promises: Promise<void>[]; restore(): void } {
    const promises: Promise<void>[] = [];
    const restores: (() => void)[] = [];
    const restoreCreation = interceptBufferCreation(device, (buffer, descriptor) => {
        if (descriptor.label !== "mirror-staging") return;
        const originalMap = buffer.mapAsync.bind(buffer);
        const previous = Object.getOwnPropertyDescriptor(buffer, "mapAsync");
        Object.defineProperty(buffer, "mapAsync", {
            configurable: true,
            value: (...args: Parameters<GPUBuffer["mapAsync"]>) => {
                const promise = originalMap(...args);
                promises.push(promise);
                return promise;
            },
        });
        restores.push(() => restoreProperty(buffer, "mapAsync", previous));
    });
    return {
        promises,
        restore() {
            restoreCreation();
            for (const restore of restores) restore();
        },
    };
}

test("a Mirror snapshot carries the fixed tick and frame captured when its readback was encoded", async () => {
    await withGpuMirror(async (state, device, source, subject) => {
        const maps = trackMirrorMaps(device);
        try {
            const marker = new Uint32Array([0x12345678]);
            device.queue.writeBuffer(source, 0, marker);
            state.step(Time.FIXED_DT);
            const fixedTick = state.time.fixedTick;
            Compute.frame = 47;
            Mirror.flush(state);
            expect(maps.promises).toHaveLength(1);
            await maps.promises[0];
            expect(subject.snapshot?.fixedTick).toBe(fixedTick);
            expect(subject.snapshot?.frame).toBe(47);
            expect(new Uint32Array(subject.snapshot!.bytes)[0]).toBe(marker[0]);
        } finally {
            maps.restore();
        }
    });
}, 20_000);

test("Mirror.flush creates no staging work for a registered Mirror after its device is lost", async () => {
    await withGpuMirror(async (state, device, _source, subject) => {
        device.destroy();
        await device.lost;
        Mirror.flush(state);
        expect(subject.allocated).toBe(0);
        expect(subject.snapshot).toBeNull();
    });
}, 20_000);

test("a pending Mirror callback releases its staging buffers and leaves no snapshot after device loss", async () => {
    await withGpuMirror(async (state, device, _source, subject) => {
        const maps = trackMirrorMaps(device);
        try {
            Mirror.flush(state);
            expect(maps.promises).toHaveLength(1);
            device.destroy();
            await device.lost;
            await maps.promises[0].catch(() => {});
            expect(subject.allocated).toBe(0);
            expect(subject.snapshot).toBeNull();
        } finally {
            maps.restore();
        }
    });
}, 20_000);

test("a rejected Mirror map recycles its slot so a later readback recovers at ring depth one", async () => {
    await withGpuMirror(async (state, device, source, subject) => {
        const errors: string[] = [];
        const originalError = console.error;
        const slots: {
            buffer: GPUBuffer;
            calls: number;
            maps: Promise<void>[];
            restoreMap?: () => void;
        }[] = [];
        console.error = (...args: Parameters<typeof console.error>) => {
            errors.push(args.map(String).join(" "));
        };
        const restoreCreate = interceptBufferCreation(device, (buffer, descriptor) => {
            if (descriptor.label !== "mirror-staging") return;
            const originalMap = buffer.mapAsync.bind(buffer);
            const previous = Object.getOwnPropertyDescriptor(buffer, "mapAsync");
            const tracked: {
                buffer: GPUBuffer;
                calls: number;
                maps: Promise<void>[];
                restoreMap?: () => void;
            } = { buffer, calls: 0, maps: [] };
            slots.push(tracked);
            Object.defineProperty(buffer, "mapAsync", {
                configurable: true,
                value: (...args: Parameters<GPUBuffer["mapAsync"]>) => {
                    tracked.calls++;
                    if (tracked.calls === 1) {
                        const rejected = Promise.reject(new Error("deliberate map rejection"));
                        tracked.maps.push(rejected);
                        return rejected;
                    }
                    const mapped = originalMap(...args);
                    tracked.maps.push(mapped);
                    return mapped;
                },
            });
            tracked.restoreMap = () => restoreProperty(buffer, "mapAsync", previous);
        });
        try {
            device.queue.writeBuffer(source, 0, new Uint32Array([7]));
            state.step(Time.FIXED_DT);
            Compute.frame = 1;
            Mirror.flush(state);
            await slots[0].maps[0].catch(() => {});
            expect(errors).toHaveLength(1);
            expect(subject.allocated).toBe(1);
            expect(slots).toHaveLength(1);
            expect(slots[0].calls).toBe(1);

            device.queue.writeBuffer(source, 0, new Uint32Array([9]));
            state.step(Time.FIXED_DT);
            Compute.frame = 2;
            Mirror.flush(state);
            await slots[0].maps[1];
            expect(slots).toHaveLength(1);
            expect(slots[0].calls).toBe(2);
            expect(subject.snapshot?.fixedTick).toBe(state.time.fixedTick);
            expect(subject.snapshot?.frame).toBe(2);
            expect(new Uint32Array(subject.snapshot!.bytes)[0]).toBe(9);
        } finally {
            restoreCreate();
            console.error = originalError;
            for (const slot of slots) slot.restoreMap?.();
        }
    });
}, 20_000);
