/**
 * thrown when the device can't meet a required WebGPU feature or limit. `missing` names the absent
 * feature(s); {@link requestGPU} throws it before any plugin loads, so an unsupported device fails loud
 * with a named cause rather than an opaque validation error deep in a pipeline.
 */
export class UnsupportedError extends Error {
    readonly missing: readonly string[];
    constructor(message: string, missing: readonly string[] = []) {
        super(message);
        this.name = "UnsupportedError";
        this.missing = missing;
    }
}

const mb = (bytes: number): string => `${(bytes / (1 << 20)).toFixed(0)} MB`;

/**
 * pre-flight a large/fixed-cap storage buffer against the device's per-binding limit. A heavy scene
 * grows several of these (the physics contact store, the BVH node buffer); past `maxStorageBufferBindingSize`
 * the bare allocation OOMs silently or surfaces an opaque bind-group validation error, so this throws a
 * named {@link UnsupportedError} first: the buffer, the needed-vs-available MB, and a remedy. Pure
 * (bytes + limit), so a unit test exercises it with no device. `label` names the buffer (e.g.
 * `"[bvh] the node buffer"`); `remedy` says how to fit under the limit.
 */
export function checkStorageBinding(
    label: string,
    bytes: number,
    maxBinding: number,
    remedy: string,
): void {
    if (bytes > maxBinding) {
        throw new UnsupportedError(
            `${label} needs ${mb(bytes)}, but the device's maxStorageBufferBindingSize is ` +
                `${mb(maxBinding)}. ${remedy}`,
        );
    }
}

/**
 * pre-flight a texture (or texture array) against the device's dimension + array-layer limits. A texture
 * whose width/height exceeds `maxTextureDimension2D` or whose layer count exceeds `maxTextureArrayLayers`
 * (a VAT keyed by a huge vertex/frame count, a glTF baseColor / sprite array unioning many sources) fails
 * at an opaque `createTexture` validation error; this throws a named {@link UnsupportedError} first: the
 * extent, the needed-vs-available, and a remedy. Pure (extents + limits), so a unit test exercises it with
 * no device. `layers` defaults to 1 (a plain 2D texture).
 */
export function checkTextureLimits(
    label: string,
    size: { width: number; height: number; layers?: number },
    limits: Pick<GPUSupportedLimits, "maxTextureDimension2D" | "maxTextureArrayLayers">,
    remedy: string,
): void {
    const dim = Math.max(size.width, size.height);
    if (dim > limits.maxTextureDimension2D) {
        throw new UnsupportedError(
            `${label} needs a ${size.width}×${size.height} texture, but the device's ` +
                `maxTextureDimension2D is ${limits.maxTextureDimension2D}. ${remedy}`,
        );
    }
    const layers = size.layers ?? 1;
    if (layers > limits.maxTextureArrayLayers) {
        throw new UnsupportedError(
            `${label} needs ${layers} array layers, but the device's maxTextureArrayLayers is ` +
                `${limits.maxTextureArrayLayers}. ${remedy}`,
        );
    }
}

/**
 * active GPU device with per-frame fence sync
 * @expand
 */
export interface Compute {
    /** active GPU device */
    readonly device: GPUDevice;
    /** monotonically incremented per frame */
    frame: number;
    /** frames submitted but not yet retired by the GPU */
    pending(): number;
    /** register the just-submitted frame's completion fence; tracks {@link pending}, returns the fence */
    sync(): Promise<void>;
    /**
     * named GPU buffers published for cross-system lookup. Slabs with a name
     * self-register; producers register their static buffers (cube vertices,
     * transforms firehose, …). Consumers (renderers) resolve binding names
     * to buffers at bind-group build time
     */
    readonly buffers: Map<string, GPUBuffer>;
    /**
     * named GPU textures published for cross-system lookup, mirroring
     * {@link buffers}. Producers register loaded / rendered textures;
     * surfaces declaring a `texture-2d` / `texture-depth-2d` binding resolve
     * the name here at bind-group build time
     */
    readonly textures: Map<string, GPUTexture>;
    /**
     * named GPU samplers published for cross-system lookup, mirroring
     * {@link buffers}. Surfaces declaring a `sampler` / `sampler-comparison`
     * binding resolve the name here at bind-group build time
     */
    readonly samplers: Map<string, GPUSampler>;
    /** optional GPU timestamp slot allocator hook; returns writes for a pass descriptor */
    span?: (
        name: string,
    ) => GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined;
    /**
     * optional indirect-draw tally hook installed by `ProfilePlugin`. A pass reports the
     * `drawIndexedIndirect` commands it issues (the honest count, post the skip), and the profiler
     * derives Dawn's injected indirect-draw validation floor (`#draws × ~1µs`, untimed by
     * `timestampWrites` because it runs before the pass, gpu.md "WebGPU-specific traps"). A `?.`
     * no-op without the plugin, mirroring {@link span}. A bundle reports its *recorded* draw count;
     * the injected validation runs the same for a replay
     */
    indirect?: (name: string, count: number) => void;
}

/** active GPU compute singleton, populated by {@link requestGPU} */
export const Compute: Compute = {} as Compute;

// the base floor every shallot app needs (the default renderer + slab substrate). Optional
// capabilities declare their own on top via `Plugin.features` (required — a missing one throws) or
// `Plugin.preferredFeatures` (best-effort — requested only where the adapter has it, never throws).
// `subgroups` is the standing preferred case: the BVH builder (physics broadphase / accel structure)
// runs a faster subgroup arm where present and an LDS arm where absent (WebKit), so it's preferred,
// not required — a no-subgroup device still loads a physics app, on the LDS arm.
const BASE_FEATURES = [
    "shader-f16",
    "timestamp-query",
    "indirect-first-instance",
    // a fused postfx composite writes the swapchain from a compute pass; on Mac/Windows the
    // preferred canvas format is bgra8unorm, and a storage view of it needs this feature
    "bgra8unorm-storage",
    // the default HDR scene offscreen + sear's MSAA color target are rg11b10ufloat (render.md "Camera
    // passes"): grants it render-attachment + multisample + resolve. Half the bandwidth of rgba16float at
    // 4× MSAA, on the whole floor (desktop / Steam Deck / recent Android all support it)
    "rg11b10ufloat-renderable",
] as const;

const COMPRESSION_FAMILIES = [
    "texture-compression-bc",
    "texture-compression-etc2",
    "texture-compression-astc",
] as const;

/** shallot's per-stage storage buffer floor. 99.6% of WebGPU devices support 10. */
const REQUIRED_STORAGE_BUFFERS_PER_STAGE = 10;

/**
 * split requested features against what an adapter offers. `required` (the base floor ∪ the active
 * plugins' `Plugin.features`) that the adapter lacks land in `missing`; the caller throws. `preferred`
 * (the plugins' `Plugin.preferredFeatures`) are `granted` only where present, never gating the device:
 * a plugin asks for an arm it can run without (the BVH builder's `subgroups`). Pure over the adapter's
 * feature set, so a unit test exercises it with no device.
 */
export function resolveFeatures(
    available: { has(feature: GPUFeatureName): boolean },
    required: readonly GPUFeatureName[],
    preferred: readonly GPUFeatureName[],
): { granted: GPUFeatureName[]; missing: GPUFeatureName[] } {
    const missing = required.filter((f) => !available.has(f));
    const granted = preferred.filter((f) => available.has(f) && !required.includes(f));
    return { granted, missing };
}

/**
 * populate the {@link Compute} singleton. With no argument, acquires a device
 * via `navigator.gpu` and enforces shallot's feature floor (the base floor plus
 * any `features` the active plugins require), throwing {@link UnsupportedError}
 * otherwise. `preferred` features are requested only where the adapter has them
 * (never gating the device). Pass an external device to adopt it as-is; the caller
 * is responsible for feature support.
 */
export async function requestGPU(
    device?: GPUDevice,
    features: readonly GPUFeatureName[] = [],
    preferred: readonly GPUFeatureName[] = [],
): Promise<Compute> {
    const d = device ?? (await acquireDevice(features, preferred));
    let inFlight = 0;
    return Object.assign(Compute, {
        device: d,
        frame: 0,
        pending: () => inFlight,
        sync: () => {
            inFlight++;
            const fence = d.queue.onSubmittedWorkDone();
            fence.then(() => inFlight--);
            return fence;
        },
        buffers: new Map<string, GPUBuffer>(),
        textures: new Map<string, GPUTexture>(),
        samplers: new Map<string, GPUSampler>(),
    });
}

async function acquireDevice(
    extra: readonly GPUFeatureName[],
    preferred: readonly GPUFeatureName[],
): Promise<GPUDevice> {
    if (!navigator.gpu) throw new UnsupportedError("WebGPU not supported in this browser");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new UnsupportedError("No compatible GPU found");

    const required = [...new Set<GPUFeatureName>([...BASE_FEATURES, ...extra])];
    const { granted, missing } = resolveFeatures(adapter.features, required, preferred);
    if (missing.length > 0) throw new UnsupportedError("Missing required WebGPU features", missing);

    const compression = COMPRESSION_FAMILIES.find((f) => adapter.features.has(f));
    if (!compression)
        throw new UnsupportedError(
            "No supported texture compression format. Requires one of:",
            COMPRESSION_FAMILIES.slice(),
        );

    if (adapter.limits.maxStorageBuffersPerShaderStage < REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
        throw new UnsupportedError(
            `Only ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; ${REQUIRED_STORAGE_BUFFERS_PER_STAGE} required`,
        );
    }

    const device = await adapter.requestDevice({
        requiredFeatures: [...required, ...granted, compression] as GPUFeatureName[],
        requiredLimits: deviceLimits(adapter.limits),
    });

    device.lost.then((info) => console.error(`GPU device lost: ${info.reason}`, info.message));
    device.onuncapturederror = (event) => {
        const msg = event.error instanceof GPUValidationError ? event.error.message : event.error;
        console.error("GPU uncaptured error:", msg);
    };

    return device;
}

/**
 * the limits requested at device acquisition, read from the adapter. Requesting ≤ the adapter's
 * reported value is always granted, so this never rejects `requestDevice`; the storage-binding /
 * buffer sizes pass the adapter's full values through (physics' compacted contact store needs the
 * full size past the 128 MB / 256 MB spec defaults at high capacity, and a consumer overreaching
 * the true limit still fails loud at bind-group validation).
 *
 * The split-stage storage limits are a 2024 spec addition absent on older mobile WebGPU, where the
 * adapter reports `undefined`. Forwarding `undefined` makes WebIDL's `GPUSize64` conversion throw
 * "Value NaN is outside the range [0, 9007199254740991]" and reject the device, so drop any absent
 * limit and let the device apply its default (the unified `maxStorageBuffersPerShaderStage` still
 * governs there).
 */
export function deviceLimits(limits: GPUSupportedLimits): Record<string, number> {
    const wanted: Record<string, number | undefined> = {
        maxTextureDimension2D: limits.maxTextureDimension2D,
        maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        maxStorageBuffersInVertexStage: limits.maxStorageBuffersInVertexStage,
        maxStorageBuffersInFragmentStage: limits.maxStorageBuffersInFragmentStage,
        maxStorageTexturesInVertexStage: limits.maxStorageTexturesInVertexStage,
        maxStorageTexturesInFragmentStage: limits.maxStorageTexturesInFragmentStage,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxBufferSize: limits.maxBufferSize,
    };
    const required: Record<string, number> = {};
    for (const [key, value] of Object.entries(wanted)) {
        if (value !== undefined) required[key] = value;
    }
    return required;
}
