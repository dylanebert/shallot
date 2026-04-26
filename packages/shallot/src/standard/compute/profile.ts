import { resource } from "../../engine";

export const GpuProfile = resource<Map<string, number>[]>("gpu-profile");

export interface ProfileState {
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    readBuffer: GPUBuffer;
    capacity: number;
    nextSlot: number;
    passes: string[];
    durations: Map<string, number>;
    pendingCount: number;
    pendingPasses: string[];
}

export function createProfileState(device: GPUDevice, capacity = 32): ProfileState {
    const querySet = device.createQuerySet({
        type: "timestamp",
        count: capacity * 2,
    });

    const byteSize = capacity * 2 * 8;

    const resolveBuffer = device.createBuffer({
        label: "profile-resolve",
        size: byteSize,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
        label: "profile-read",
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    return {
        querySet,
        resolveBuffer,
        readBuffer,
        capacity,
        nextSlot: 0,
        passes: [],
        durations: new Map(),
        pendingCount: 0,
        pendingPasses: [],
    };
}

const slotCache: {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex: number;
    endOfPassWriteIndex: number;
}[] = [];

export function allocSlot(
    state: ProfileState,
    name: string,
): GPUComputePassTimestampWrites | undefined {
    const slot = state.nextSlot;
    if (slot >= state.capacity) return undefined;
    state.passes[slot] = name;
    state.nextSlot = slot + 1;
    let cached = slotCache[slot];
    if (!cached || cached.querySet !== state.querySet) {
        cached = {
            querySet: state.querySet,
            beginningOfPassWriteIndex: slot * 2,
            endOfPassWriteIndex: slot * 2 + 1,
        };
        slotCache[slot] = cached;
    }
    return cached;
}

export function resetProfile(state: ProfileState): void {
    state.nextSlot = 0;
}

export function resolveProfile(encoder: GPUCommandEncoder, state: ProfileState): void {
    const count = state.nextSlot * 2;
    if (count === 0) return;
    encoder.resolveQuerySet(state.querySet, 0, count, state.resolveBuffer, 0);
    if (state.readBuffer.mapState === "unmapped") {
        encoder.copyBufferToBuffer(state.resolveBuffer, 0, state.readBuffer, 0, count * 8);
        state.pendingCount = state.nextSlot;
        state.pendingPasses = state.passes.slice(0, state.nextSlot);
    }
}

export function readProfile(state: ProfileState): void {
    if (state.readBuffer.mapState !== "unmapped") return;
    if (state.pendingCount === 0) return;
    state.readBuffer.mapAsync(GPUMapMode.READ).catch(() => {});
}

export function drainProfile(state: ProfileState): void {
    if (state.readBuffer.mapState !== "mapped") return;
    state.durations.clear();
    const range = state.readBuffer.getMappedRange();
    const data = new BigUint64Array(range);
    for (let i = 0; i < state.pendingCount; i++) {
        const name = state.pendingPasses[i];
        const begin = data[i * 2];
        const end = data[i * 2 + 1];
        state.durations.set(name, (state.durations.get(name) ?? 0) + Number(end - begin) / 1e6);
    }
    state.readBuffer.unmap();
}

export interface GpuAlloc {
    label: string;
    bytes: number;
    kind: "buffer" | "texture";
}

export interface CompileTiming {
    label: string;
    ms: number;
}

export interface GpuRegistry {
    buffers: Set<GPUBuffer>;
    textures: Set<GPUTexture>;
    sizes: Map<object, GpuAlloc>;
    pendingSubmits: number;
    bufferBytes: number;
    textureBytes: number;
    compileTimings: CompileTiming[];
    compileTotalMs: number;
    compileSpans: { label: string; start: number; end: number }[];
    count(): number;
    totalBytes(): number;
    assertEmpty(): void;
    finalizeCompile(): void;
}

export const GpuRegistryResource = resource<GpuRegistry>("gpu-registry");

export function trackDevice(device: GPUDevice): GpuRegistry {
    const registry = createGpuRegistry();

    device.createBuffer = wrapCreateBuffer(device.createBuffer.bind(device), registry);
    device.createTexture = wrapCreateTexture(device.createTexture.bind(device), registry);
    device.createComputePipelineAsync = wrapAsyncPipeline(
        device.createComputePipelineAsync.bind(device),
        registry,
    );
    device.createRenderPipelineAsync = wrapAsyncPipeline(
        device.createRenderPipelineAsync.bind(device),
        registry,
    );

    const origSubmit = device.queue.submit.bind(device.queue);
    device.queue.submit = (commandBuffers: Iterable<GPUCommandBuffer>) => {
        origSubmit(commandBuffers);
        registry.pendingSubmits++;
        if (registry.pendingSubmits > 12) {
            console.warn(`GPU back-pressure: ${registry.pendingSubmits} submissions pending`);
        }
        device.queue.onSubmittedWorkDone().then(() => {
            registry.pendingSubmits--;
        });
    };

    return registry;
}

function texelBytes(format: string): number {
    if (format.startsWith("rgba32")) return 16;
    if (format.startsWith("rgba16")) return 8;
    if (format.startsWith("rgba8") || format.startsWith("bgra8")) return 4;
    if (format.startsWith("rg32")) return 8;
    if (format.startsWith("rg16")) return 4;
    if (format.startsWith("rg8")) return 2;
    if (format.startsWith("r32")) return 4;
    if (format.startsWith("r16")) return 2;
    if (format.startsWith("r8")) return 1;
    if (format === "depth24plus" || format === "depth32float" || format === "depth24plus-stencil8")
        return 4;
    return 4;
}

function normalizeSize(size: GPUExtent3DStrict): [number, number, number] {
    if (Array.isArray(size)) return [size[0], size[1] ?? 1, size[2] ?? 1];
    const dict = size as GPUExtent3DDictStrict;
    return [dict.width, dict.height ?? 1, dict.depthOrArrayLayers ?? 1];
}

function textureByteSize(desc: GPUTextureDescriptor): number {
    const [w, h, d] = normalizeSize(desc.size);
    const bpp = texelBytes(desc.format);
    const mips = desc.mipLevelCount ?? 1;
    let total = 0;
    for (let i = 0; i < mips; i++) {
        total += Math.max(1, w >> i) * Math.max(1, h >> i) * d * bpp;
    }
    return total;
}

function wrapCreateBuffer(
    orig: (desc: GPUBufferDescriptor) => GPUBuffer,
    registry: GpuRegistry,
): (desc: GPUBufferDescriptor) => GPUBuffer {
    return (descriptor: GPUBufferDescriptor): GPUBuffer => {
        const buffer = orig(descriptor);
        const bytes = descriptor.size;
        registry.buffers.add(buffer);
        registry.bufferBytes += bytes;
        registry.sizes.set(buffer, { label: descriptor.label ?? "", bytes, kind: "buffer" });
        const origDestroy = buffer.destroy.bind(buffer);
        (buffer as any).destroy = () => {
            registry.buffers.delete(buffer);
            registry.bufferBytes -= bytes;
            registry.sizes.delete(buffer);
            origDestroy();
        };
        return buffer;
    };
}

function wrapCreateTexture(
    orig: (desc: GPUTextureDescriptor) => GPUTexture,
    registry: GpuRegistry,
): (desc: GPUTextureDescriptor) => GPUTexture {
    return (descriptor: GPUTextureDescriptor): GPUTexture => {
        const texture = orig(descriptor);
        const bytes = textureByteSize(descriptor);
        registry.textures.add(texture);
        registry.textureBytes += bytes;
        registry.sizes.set(texture, { label: descriptor.label ?? "", bytes, kind: "texture" });
        const origDestroy = texture.destroy.bind(texture);
        (texture as any).destroy = () => {
            registry.textures.delete(texture);
            registry.textureBytes -= bytes;
            registry.sizes.delete(texture);
            origDestroy();
        };
        return texture;
    };
}

function wrapAsyncPipeline<D, P>(
    orig: (desc: D) => Promise<P>,
    registry: GpuRegistry,
): (desc: D) => Promise<P> {
    return async (desc: D): Promise<P> => {
        const start = performance.now();
        const pipeline = await orig(desc);
        const label = (desc as any).label ?? "";
        registry.compileSpans.push({ label, start, end: performance.now() });
        return pipeline;
    };
}

function createGpuRegistry(): GpuRegistry {
    const buffers = new Set<GPUBuffer>();
    const textures = new Set<GPUTexture>();
    const sizes = new Map<object, GpuAlloc>();

    return {
        buffers,
        textures,
        sizes,
        pendingSubmits: 0,
        bufferBytes: 0,
        textureBytes: 0,
        compileTimings: [],
        compileTotalMs: 0,
        compileSpans: [],
        count() {
            return buffers.size + textures.size;
        },
        totalBytes() {
            return this.bufferBytes + this.textureBytes;
        },
        assertEmpty() {
            if (buffers.size === 0 && textures.size === 0) return;
            for (const buf of buffers) {
                console.warn(`leaked buffer: ${buf.label || "(unlabeled)"}`);
            }
            for (const tex of textures) {
                console.warn(`leaked texture: ${tex.label || "(unlabeled)"}`);
            }
        },
        finalizeCompile() {
            const ends = this.compileSpans;
            if (ends.length === 0) return;
            for (const e of ends) {
                this.compileTimings.push({ label: e.label, ms: e.end - e.start });
            }
            this.compileTimings.sort((a, b) => b.ms - a.ms);
            let earliest = Infinity;
            let latest = -Infinity;
            for (const e of ends) {
                if (e.start < earliest) earliest = e.start;
                if (e.end > latest) latest = e.end;
            }
            this.compileTotalMs = latest - earliest;
            this.compileSpans = [];
        },
    };
}
