import { capacity } from "../../engine";

export interface BufferView {
    buffer: GPUBuffer;
    offset: number;
    size: number;
}

export function bindView(binding: number, view: BufferView): GPUBindGroupEntry {
    return { binding, resource: { buffer: view.buffer, offset: view.offset, size: view.size } };
}

export interface GBuf {
    readonly buffer: GPUBuffer;
}

export function gbuf(
    device: GPUDevice,
    label: string,
    usage: number,
    size: (cap: number) => number,
): GBuf {
    let currentCap = capacity();
    let gpuBuffer: GPUBuffer = device.createBuffer({
        label,
        size: size(currentCap),
        usage,
    });

    return {
        get buffer(): GPUBuffer {
            const cap = capacity();
            if (cap !== currentCap) {
                gpuBuffer.destroy();
                gpuBuffer = device.createBuffer({
                    label,
                    size: size(cap),
                    usage,
                });
                currentCap = cap;
            }
            return gpuBuffer;
        },
    };
}

export interface Binding {
    readonly group: GPUBindGroup;
    invalidate(): void;
}

export function binding(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    entries: () => GPUBindGroupEntry[],
): Binding {
    let cachedCapacity = capacity();
    let cached: GPUBindGroup | null = null;

    return {
        get group() {
            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                cached = null;
            }
            if (!cached) {
                cached = device.createBindGroup({ layout, entries: entries() });
            }
            return cached;
        },
        invalidate() {
            cached = null;
        },
    };
}

export function view(
    ref: GBuf,
    offset: (cap: number) => number,
    size: (cap: number) => number,
): BufferView {
    return {
        get buffer() {
            return ref.buffer;
        },
        get offset() {
            return offset(capacity());
        },
        get size() {
            return size(capacity());
        },
    };
}
