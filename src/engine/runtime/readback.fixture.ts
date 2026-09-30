import { State } from "../ecs";

export function countStaging(device: GPUDevice) {
    const original = device.createBuffer.bind(device);
    const counts = { created: 0, live: 0 };
    device.createBuffer = (descriptor) => {
        const buffer = original(descriptor);
        if ((descriptor.usage & GPUBufferUsage.MAP_READ) !== 0) {
            counts.created++;
            counts.live++;
            const destroy = buffer.destroy.bind(buffer);
            let destroyed = false;
            buffer.destroy = () => {
                if (!destroyed) {
                    destroyed = true;
                    counts.live--;
                }
                destroy();
            };
        }
        return buffer;
    };
    return {
        counts,
        restore: () => {
            device.createBuffer = original;
        },
    };
}

export interface ControlledSlot {
    buffer: GPUBuffer;
    resolve(bytes: number[]): void;
    reject(error: Error): void;
    destroyed: boolean;
}

/** A controllable mapping boundary for cheap pool lifecycle proofs, not GPU-content proofs. */
export async function controlledReadback(
    body: (state: State, slots: ControlledSlot[], errors: EventTarget) => Promise<void>,
) {
    const descriptors = ["GPUBufferUsage", "GPUMapMode"].map(
        (name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const,
    );
    Object.defineProperty(globalThis, "GPUBufferUsage", {
        configurable: true,
        value: { MAP_READ: 1, COPY_DST: 2 },
    });
    Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: { READ: 1 } });
    const slots: ControlledSlot[] = [];
    const events = new EventTarget();
    const state = new State();
    const device = {
        limits: { maxBufferSize: 1 << 20 },
        addEventListener: events.addEventListener.bind(events),
        removeEventListener: events.removeEventListener.bind(events),
        createCommandEncoder: () => ({ finish: () => ({}) }),
        queue: { submit() {} },
        createBuffer(descriptor: GPUBufferDescriptor) {
            let data = new ArrayBuffer(Number(descriptor.size));
            let resolve!: () => void;
            let reject!: (error: Error) => void;
            const slot: ControlledSlot = {
                buffer: undefined as unknown as GPUBuffer,
                resolve(bytes) {
                    data = Uint8Array.from(bytes).buffer;
                    buffer.mapState = "mapped";
                    resolve();
                },
                reject(error) {
                    reject(error);
                },
                destroyed: false,
            };
            const buffer = {
                size: Number(descriptor.size),
                mapState: "unmapped",
                mapAsync: () =>
                    new Promise<void>((yes, no) => {
                        resolve = yes;
                        reject = no;
                    }),
                getMappedRange: () => data,
                unmap() {
                    buffer.mapState = "unmapped";
                },
                destroy() {
                    slot.destroyed = true;
                    buffer.mapState = "unmapped";
                },
            };
            slot.buffer = buffer as unknown as GPUBuffer;
            state.own(slot.buffer);
            slots.push(slot);
            return slot.buffer;
        },
    } as unknown as GPUDevice;
    state.attachGpu(
        {
            device,
            frame: 0,
            buffers: new Map(),
            textures: new Map(),
            samplers: new Map(),
            typed: new Map(),
            root: {},
            pending: () => 0,
            sync: async () => {},
            adapter: { class: "test", identity: "controlled" },
        },
        (run) => run(),
    );
    try {
        await body(state, slots, events);
    } finally {
        state.dispose();
        for (const [name, descriptor] of descriptors) {
            if (descriptor) Object.defineProperty(globalThis, name, descriptor);
            else Reflect.deleteProperty(globalThis, name);
        }
    }
}
