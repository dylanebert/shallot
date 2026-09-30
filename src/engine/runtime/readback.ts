import type { State } from "../ecs";

interface Slot {
    buffer: GPUBuffer;
    busy: boolean;
    lastFrame: number;
    reject?: (error: unknown) => void;
}

/** Copy-time identity of a one-shot GPU result. */
export interface ReadbackStamp {
    readonly frame: number;
    readonly fixedTick: number;
}

/** World-owned one-shot staging pool. No request means no mapping. */
export class ReadbackPool {
    private readonly _slots: Slot[] = [];
    private _disposed = false;
    private _unusedFrames = 10;
    private readonly _onError = (event: GPUUncapturedErrorEvent) => {
        for (const slot of this._slots) slot.reject?.(event.error);
    };

    private readonly _device: GPUDevice;
    private readonly _state?: State;

    constructor(device: GPUDevice, state?: State) {
        this._device = device;
        this._state = state;
        device.addEventListener("uncapturederror", this._onError);
    }

    /** Release available staging after this many frames without use. Defaults to 10. */
    get maxUnusedFrames(): number {
        return this._unusedFrames;
    }
    set maxUnusedFrames(value: number) {
        if (!Number.isSafeInteger(value) || value < 1)
            throw new RangeError("readback maxUnusedFrames must be a positive integer");
        this._unusedFrames = value;
    }

    /** Number of staging buffers, including requests in flight. */
    get allocated(): number {
        return this._slots.length;
    }

    /** @internal Called after each world frame, without creating a pool for an unused world. */
    advance(frame: number): void {
        for (let i = this._slots.length - 1; i >= 0; i--) {
            const slot = this._slots[i];
            if (!slot.busy && frame - slot.lastFrame >= this._unusedFrames) {
                slot.buffer.destroy();
                this._slots.splice(i, 1);
            }
        }
    }

    /** @internal Encode and submit one copy, then return independent owned bytes. */
    async request(
        size: number,
        label: string,
        encode: (encoder: GPUCommandEncoder, staging: GPUBuffer) => void,
    ): Promise<ReadbackStamp & { bytes: ArrayBuffer }> {
        if (this._disposed) throw new Error(`${label}: readback world is disposed`);
        this._state?.assertReadbackAllowed();
        const frame = this._state?.gpu.frame ?? 0;
        const fixedTick = this._state?.time.fixedTick ?? 0;
        let slot = this._slots.find((entry) => !entry.busy && entry.buffer.size === size);
        if (!slot) {
            slot = {
                buffer: this._device.createBuffer({
                    label: "shallot-readback-staging",
                    size,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
                busy: false,
                lastFrame: frame,
            };
            this._slots.push(slot);
        }
        slot.busy = true;
        slot.lastFrame = frame;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const failure = new Promise<never>((_, reject) => {
            slot.reject = reject;
            timer = setTimeout(
                () =>
                    reject(
                        new Error(
                            `${label}: frame ${frame} tick ${fixedTick} readback map exceeded 750 ms`,
                        ),
                    ),
                750,
            );
        });
        try {
            const encoder = this._device.createCommandEncoder({ label });
            encode(encoder, slot.buffer);
            this._device.queue.submit([encoder.finish()]);
            await Promise.race([slot.buffer.mapAsync(GPUMapMode.READ, 0, size), failure]);
            if (this._disposed) throw new Error(`${label}: readback world is disposed`);
            const bytes = slot.buffer.getMappedRange(0, size).slice(0);
            const state = this._state;
            return {
                get bytes() {
                    state?.assertReadbackAllowed();
                    return bytes;
                },
                frame,
                fixedTick,
            };
        } catch (error) {
            slot.buffer.destroy();
            const index = this._slots.indexOf(slot);
            if (index >= 0) this._slots.splice(index, 1);
            throw error;
        } finally {
            clearTimeout(timer);
            slot.reject = undefined;
            if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
            slot.busy = false;
            slot.lastFrame = this._state?.gpu.frame ?? frame;
        }
    }

    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        this._device.removeEventListener("uncapturederror", this._onError);
        for (const slot of this._slots) {
            slot.reject?.(new Error("readback world disposed during request"));
            slot.buffer.destroy();
        }
        this._slots.length = 0;
    }
}
