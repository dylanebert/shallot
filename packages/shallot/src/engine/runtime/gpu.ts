import tgpu, { type TgpuBuffer, type TgpuRoot } from "typegpu";
import { type AnyData, u32 } from "typegpu/data";
import { captureGpuLog } from "./log";
import { now } from "./platform";

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
    /**
     * TypeGPU root adopting {@link device} — the handle every typed buffer, bind group, and pipeline
     * is created through, and the reach-back out (`root.unwrap(...)`) to the raw WebGPU handle.
     * Created by {@link requestGPU} (never at import time — the module stays side-effect free) and
     * **memoized per device**: a rebuild on the same device keeps the same root, so a cross-build memo
     * may hold a typed resource exactly like a raw one. It has no teardown of its own — typed
     * resources die with the device, and an adopted device's lifetime is the host's
     */
    readonly root: TgpuRoot;
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
    /**
     * typed twins of {@link buffers}, keyed by the same names: a producer that owns a schema
     * publishes the {@link TgpuBuffer} here beside the raw handle, so a typed consumer binds it
     * without re-declaring the layout and a raw consumer keeps reading `buffers`. Wiped with the
     * raw maps on every {@link requestGPU}, so a producer re-publishes each build
     */
    readonly typed: Map<string, TgpuBuffer<AnyData>>;
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
    /**
     * optional pipeline-compile timing hook installed by `ProfilePlugin`, mirroring {@link span} /
     * {@link indirect}. typegpu pipelines are sync-created (`root.unwrap` calls the synchronous
     * `create*Pipeline`), and Dawn defers the real shader compile to the forced {@link precompile}
     * dispatch — so timing the creation call would report a number that reads like a compile time and
     * isn't one. {@link precompileAll} instead measures each forcer's dispatch through its own
     * completion fence and reports it here. A `?.` no-op without the plugin; when it IS installed the
     * drain serializes on that fence, which is what makes the per-pipeline split real rather than every
     * row reading the whole drain.
     */
    precompiled?: (label: string, start: number, end: number) => void;
}

/** active GPU compute singleton, populated by {@link requestGPU} */
export const Compute: Compute = {} as Compute;

// the base floor every shallot app needs (the default renderer + slab substrate). It gates device
// acquisition before any plugin loads, so **a floor entry earns its place only by being a
// `DEFAULT_PLUGINS` need** — anything an opt-in plugin uses belongs on that plugin, via
// `Plugin.features` (required — a missing one throws) or `Plugin.preferredFeatures` (best-effort —
// requested only where the adapter has it, never throws). Hence the deliberate absentees:
// `timestamp-query` is `ProfilePlugin.features`, the BC/ETC2/ASTC families are
// `GltfPlugin.preferredFeatures` (preferred, because `gltf/target.ts` runtime-branches on
// `device.features.has` — a family the device never *requested* reads false even where the hardware
// has it), and `shader-f16` gates the WGSL `f16` *type*, not `pack2x16float` / `unpack2x16float`, so
// the `f16x4` mirror binds `vec2<u32>` and unpacks. `subgroups` is the standing preferred case: the
// BVH builder (physics broadphase / accel structure) runs a faster subgroup arm where present and an
// LDS arm where absent (WebKit), so it's preferred, not required — a no-subgroup device still loads a
// physics app, on the LDS arm.
export const BASE_FEATURES = [
    "indirect-first-instance",
    // a fused postfx composite writes the swapchain from a compute pass; on Mac/Windows the
    // preferred canvas format is bgra8unorm, and a storage view of it needs this feature
    "bgra8unorm-storage",
    // the default HDR scene offscreen + sear's MSAA color target are rg11b10ufloat (render.md "Camera
    // passes"): grants it render-attachment + multisample + resolve. Half the bandwidth of rgba16float at
    // 4× MSAA, on the whole floor (desktop / Steam Deck / recent Android all support it)
    "rg11b10ufloat-renderable",
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
 * the TGSL build-metadata canary. A TGSL function body is transpiled at **build** time by
 * `unplugin-typegpu` (typegpu parses nothing at runtime), so a bundle built without the plugin carries
 * no metadata at all — resolution throws deep inside a pipeline and CPU-called kernels silently return
 * NaN. Resolve it to prove your build ran the transform; {@link checkTgsl} does exactly that.
 * @example
 * const wgsl = tgpu.resolve([tgslCanary]); // "fn canary(x: u32) -> u32 { return (x + 1u); }"
 */
export const tgslCanary = tgpu.fn(
    [u32],
    u32,
)((x) => {
    "use gpu";
    return x + 1;
});

// the version our copy of typegpu stamped when this module imported it. Two copies in one bundle share
// the `__TYPEGPU_META__` global AND delete from it, so they race; typegpu only console.warns. Snapshot
// it here and compare at check time — that catches a second copy loaded *after* the engine's, the
// direction module evaluation makes observable.
const _globals = globalThis as unknown as Record<string, string | undefined>;
const _typegpuVersion = _globals.__TYPEGPU_VERSION__;

/**
 * fail loud when the engine's own TGSL carries no build metadata, the one failure mode of the mandatory
 * build plugin and otherwise a silent wrong-answer class. {@link requestGPU} calls it before touching
 * the adapter.
 *
 * It proves exactly one thing: the engine's `.ts` modules went through the transform. It says nothing
 * about *your* TGSL — a config that reaches `node_modules` but skips your source (or a `.svelte` /
 * `.vue` block outside the transform's file filter) still passes here and fails at your own kernel.
 */
export function checkTgsl(): void {
    const version = _globals.__TYPEGPU_VERSION__;
    if (version !== _typegpuVersion) {
        throw new Error(
            `Two copies of typegpu are loaded (${_typegpuVersion} and ${version}). They share one ` +
                "metadata map and delete from it, so kernels resolve empty at random. Dedupe typegpu " +
                "to a single copy — it is a peerDependency of the engine for exactly this reason.",
        );
    }
    try {
        tgpu.resolve([tgslCanary]);
    } catch (cause) {
        throw new Error(
            "TGSL metadata is missing — this bundle was built without the typegpu transform, so every " +
                "engine shader would resolve wrong. Add `typegpu()` from `unplugin-typegpu/vite` to your " +
                "vite config (`shallot dev` / `shallot build` projects already have it), or register " +
                "`unplugin-typegpu/bun` in a bun preload.",
            { cause },
        );
    }
}

interface Forcer {
    label: string;
    force: () => unknown;
    after: readonly string[];
    order: number;
}

// pipelines queued for a forced compile at the end of warm, and whether that drain has already run for
// this build (see `precompile`)
const _precompile: Forcer[] = [];
const _precompileLabels = new Set<string>();
let _precompileOrder = 0;
let _draining = false;
let _drained = false;

function compile({ label, force }: Forcer): void {
    let forced: unknown;
    try {
        forced = force();
    } catch (cause) {
        throw new Error(`precompile "${label}" failed — its pipeline did not compile`, { cause });
    }
    // a forcer that binds nothing dispatches nothing, so the compile silently falls through to the
    // first frame — the exact stall the queue exists to prevent, and invisible without this
    if (!forced)
        throw new Error(
            `precompile "${label}" bound nothing — its pipeline would compile on the first frame instead`,
        );
}

/**
 * force a pipeline to compile before the first frame. typegpu has no async pipeline creation —
 * unwrapping a pipeline calls the synchronous `create*Pipeline`, and Dawn defers the real compile to
 * the first dispatch (measured ~3 s of first-frame drain at engine scale). A pipeline owner registers a
 * 0-workgroup dispatch from its `warm`; `build` drains the queue once every plugin has warmed, so the
 * compile is paid under the loading screen. Registered *after* that drain (a lazily-built pipeline, a
 * post-warm producer), the dispatch runs immediately instead — late is better than silently dropped.
 *
 * `force` **returns what it dispatched**: the drain throws on a nullish return, because a forcer whose
 * buffers aren't up yet no-ops and hands the compile back to frame one without a word. Allocate inside
 * the thunk if the buffers are late — the drain runs after every plugin's warm, which is the point.
 * `label` names the pipeline in either failure and must be unique within the build. `options.after`
 * names other queued labels that must drain first. Unknown labels are ignored because the plugin that
 * owns a predecessor may be absent.
 * @example
 * precompile("narrowphase", () => bind()?.dispatchWorkgroups(0) ?? null, {
 *     after: ["publish-inputs"],
 * });
 * @internal
 */
export function precompile(
    label: string,
    force: () => unknown,
    options: { after?: readonly string[] } = {},
): void {
    if (_precompileLabels.has(label)) {
        throw new Error(`duplicate precompile label "${label}"`);
    }
    _precompileLabels.add(label);
    const forcer = { label, force, after: options.after ?? [], order: _precompileOrder++ };
    if (_drained) compile(forcer);
    else _precompile.push(forcer);
}

// per-prefix instance counts for `precompileScope`, cleared with the label set they keep unique
const _precompileScopes = new Map<string, number>();

/**
 * a unique {@link precompile} label prefix for a factory an app can instantiate more than once (the
 * BVH stages: a scene builds one BVH, the physics broadphase another). The first instance keeps the
 * bare `prefix`, so a single-instance app's labels — and their profiler rows — read unchanged; every
 * later one gets `prefix-2`, `prefix-3`, … Counts clear with the label set, on `requestGPU`.
 *
 * A scoped label is therefore not a fixed string, so it can't be named by another forcer's `after`
 * (which would silently degrade to the missing-predecessor case). Scope only a factory nothing
 * orders against.
 * @example
 * const scope = precompileScope("radix"); // "radix", then "radix-2", …
 * precompile(`${scope}-init`, () => initBound.dispatchWorkgroups(0) ?? initBound);
 * @internal
 */
export function precompileScope(prefix: string): string {
    const n = (_precompileScopes.get(prefix) ?? 0) + 1;
    _precompileScopes.set(prefix, n);
    return n === 1 ? prefix : `${prefix}-${n}`;
}

function ordered(forcers: readonly Forcer[]): Forcer[] {
    const byLabel = new Map(forcers.map((forcer) => [forcer.label, forcer]));
    const outgoing = new Map(forcers.map((forcer) => [forcer.label, [] as Forcer[]]));
    const incoming = new Map(forcers.map((forcer) => [forcer.label, 0]));
    for (const forcer of forcers) {
        for (const label of new Set(forcer.after)) {
            if (!byLabel.has(label)) continue;
            outgoing.get(label)!.push(forcer);
            incoming.set(forcer.label, incoming.get(forcer.label)! + 1);
        }
    }

    const ready = forcers.filter((forcer) => incoming.get(forcer.label) === 0);
    ready.sort((a, b) => a.order - b.order);
    const result: Forcer[] = [];
    while (ready.length > 0) {
        const forcer = ready.shift()!;
        result.push(forcer);
        for (const dependent of outgoing.get(forcer.label)!) {
            const count = incoming.get(dependent.label)! - 1;
            incoming.set(dependent.label, count);
            if (count === 0) {
                const at = ready.findIndex((entry) => entry.order > dependent.order);
                ready.splice(at < 0 ? ready.length : at, 0, dependent);
            }
        }
    }
    if (result.length !== forcers.length) {
        const cycle = forcers
            .filter((forcer) => incoming.get(forcer.label)! > 0)
            .map((forcer) => forcer.label);
        throw new Error(`precompile cycle: ${cycle.join(" -> ")}`);
    }
    return result;
}

/**
 * drain the {@link precompile} queue in stable topological order. `build` calls it after every plugin
 * `warm`; a forcer registered afterwards runs on arrival. Entries leave the queue one at a time, so a
 * throwing forcer names itself and leaves the rest queued rather than taking them down with it.
 * @internal
 */
export async function precompileAll(): Promise<void> {
    if (_draining) return;
    _draining = true;
    try {
        while (_precompile.length > 0) {
            _precompile.splice(0, _precompile.length, ...ordered(_precompile));
            const forcer = _precompile.shift()!;
            const start = now();
            compile(forcer);
            // without the profiler there is nothing to attribute, so the drain stays one batched submit;
            // with it, each forcer settles behind its own fence, which is the only way the per-pipeline
            // numbers mean anything (back-to-back submits all resolve together and every row reads the same)
            if (Compute.precompiled) {
                await Compute.device.queue.onSubmittedWorkDone();
                Compute.precompiled(forcer.label, start, now());
            }
        }
        _drained = true;
    } finally {
        _draining = false;
    }
}

// the root is device-scoped, not build-scoped. A device outlives any one build (a host sharing one
// across builds is the cross-build memo case), and `initFromDevice`'s `destroy` frees nothing but the
// texture cache — so a per-build teardown would buy no cleanup and leave a window where `Compute.root`
// is undefined. One root per device, memoized; it dies with the device, exactly like a raw GPUBuffer.
let _rootDevice: GPUDevice | undefined;
let _root: TgpuRoot | undefined;

function adopt(device: GPUDevice): TgpuRoot {
    if (_root && _rootDevice === device) return _root;
    _precompile.length = 0;
    _rootDevice = device;
    _root = tgpu.initFromDevice({ device });
    return _root;
}

/**
 * populate the {@link Compute} singleton. With no argument, acquires a device
 * via `navigator.gpu` and enforces shallot's feature floor (the base floor plus
 * any `features` the active plugins require), throwing {@link UnsupportedError}
 * otherwise. `preferred` features are requested only where the adapter has them
 * (never gating the device). Pass an external device to adopt it as-is; the caller
 * is responsible for feature support. Either way the device is adopted by {@link Compute.root}, the
 * TypeGPU handle typed resources are created through — memoized per device, so only a *new* device
 * mints a new root.
 */
export async function requestGPU(
    device?: GPUDevice,
    features: readonly GPUFeatureName[] = [],
    preferred: readonly GPUFeatureName[] = [],
): Promise<Compute> {
    // before anything resolves: typegpu binds the console method a TGSL `console.log` calls at
    // shader-generation time, so a capture installed later never sees that kernel's lines.
    captureGpuLog();
    checkTgsl();
    const d = device ?? (await acquireDevice(features, preferred));
    _precompile.length = 0;
    _precompileLabels.clear();
    _precompileScopes.clear();
    _precompileOrder = 0;
    _draining = false;
    _drained = false;
    let inFlight = 0;
    return Object.assign(Compute, {
        device: d,
        root: adopt(d),
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
        typed: new Map<string, TgpuBuffer<AnyData>>(),
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

    if (adapter.limits.maxStorageBuffersPerShaderStage < REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
        throw new UnsupportedError(
            `Only ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; ${REQUIRED_STORAGE_BUFFERS_PER_STAGE} required`,
        );
    }

    const device = await adapter.requestDevice({
        requiredFeatures: [...required, ...granted],
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
