import type { World } from "../ecs";
import { deviceLost, type LazyAlloc } from "./gpu";

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

/** World-owned one-shot staging pool. No request means no mapping or readback allocation.
 * Staging is reused, but each request allocates mapping promises and independent result bytes.
 * Results carry copy-time stamps; arrival timing is not deterministic simulation input. */
export class ReadbackPool {
    private readonly _slots: Slot[] = [];
    private _disposed = false;
    private _unusedFrames = 10;
    private readonly _onError = (event: GPUUncapturedErrorEvent) => {
        for (const slot of this._slots) slot.reject?.(event.error);
    };

    private readonly _device: GPUDevice;
    private readonly _world: World;

    constructor(world: World) {
        this._device = world.gpu.device;
        this._world = world;
        this._device.addEventListener("uncapturederror", this._onError);
    }

    /** Release available staging after this many idle frames. Defaults to 10; zero releases at completion. */
    get maxUnusedFrames(): number {
        return this._unusedFrames;
    }
    set maxUnusedFrames(value: number) {
        if (!Number.isSafeInteger(value) || value < 0)
            throw new RangeError("readback maxUnusedFrames must be a non-negative integer");
        this._unusedFrames = value;
    }

    /** Number of staging buffers, including requests in flight. */
    get allocated(): number {
        return this._slots.length;
    }

    /** @internal Called after each world frame, without creating a pool for an unused world. */
    advance(frame: number): void {
        if (deviceLost(this._device)) {
            this.dispose();
            return;
        }
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
        if (deviceLost(this._device)) this.dispose();
        if (this._disposed) throw new Error(`${label}: readback world or device is disposed`);
        if (
            !Number.isSafeInteger(size) ||
            size < 4 ||
            size % 4 !== 0 ||
            size > this._device.limits.maxBufferSize
        ) {
            throw new RangeError(
                `${label}: readback size must be a positive multiple of 4 within device.limits.maxBufferSize`,
            );
        }
        const frame = this._world.gpu.frame;
        const fixedTick = this._world.time.fixedTick;
        let slot = this._slots.find((entry) => !entry.busy && entry.buffer.size === size);
        if (!slot) {
            const descriptor: GPUBufferDescriptor & LazyAlloc = {
                label: "shallot-readback-staging",
                size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                lazy: true,
            };
            slot = {
                buffer: this._device.createBuffer(descriptor),
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
            const copyBuffer = encoder.copyBufferToBuffer;
            const copyTexture = encoder.copyTextureToBuffer?.bind(encoder);
            if (copyBuffer)
                encoder.copyBufferToBuffer = (source: GPUBuffer, ...args: unknown[]) => {
                    if (!this._world.owns(source))
                        throw new Error(
                            `${label}: buffer "${source.label || "unlabeled buffer"}" is not owned by this world`,
                        );
                    Reflect.apply(copyBuffer, encoder, [source, ...args]);
                    return undefined;
                };
            if (copyTexture)
                encoder.copyTextureToBuffer = (source, ...args) => {
                    if (!this._world.owns(source.texture))
                        throw new Error(
                            `${label}: texture "${source.texture.label || "unlabeled texture"}" is not owned by this world`,
                        );
                    copyTexture(source, ...args);
                };
            encode(encoder, slot.buffer);
            this._device.queue.submit([encoder.finish()]);
            await Promise.race([slot.buffer.mapAsync(GPUMapMode.READ, 0, size), failure]);
            if (this._disposed) throw new Error(`${label}: readback world is disposed`);
            const bytes = slot.buffer.getMappedRange(0, size).slice(0);
            return { bytes, frame, fixedTick };
        } catch (error) {
            if (deviceLost(this._device)) this.dispose();
            slot.buffer.destroy();
            const index = this._slots.indexOf(slot);
            if (index >= 0) this._slots.splice(index, 1);
            throw error;
        } finally {
            clearTimeout(timer);
            slot.reject = undefined;
            if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
            slot.busy = false;
            slot.lastFrame = this._world.gpu.frame;
            if (this._unusedFrames === 0) this.advance(slot.lastFrame);
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
