import tgpu, { type TgpuBuffer, type TgpuRoot } from "typegpu";
import { type AnyData, u32 } from "typegpu/data";
import type { Resource, World } from "../ecs";
import { type AdapterInfoFacts, type AdapterVerdict, classifyAdapter } from "./adapter";
import { captureGpuLog } from "./log";
import { now } from "./platform";

/**
 * thrown when WebGPU is unavailable or a device can't meet a required feature or limit. `missing` names
 * absent required feature(s); {@link requestGPU} throws before any plugin loads, with the host or device
 * failure named rather than surfacing as an opaque validation error deep in a pipeline.
 */
export class UnsupportedError extends Error {
    readonly missing: readonly string[];
    constructor(message: string, missing: readonly string[] = [], options?: ErrorOptions) {
        super(message, options);
        this.name = "UnsupportedError";
        this.missing = missing;
    }
}

/** one labeled WebGPU failure crossing a validation boundary. @internal */
export class GpuDiagnosticError extends Error {
    readonly label: string;
    readonly errorClass: string;
    constructor(label: string, cause: unknown) {
        const detail = failure(cause);
        super(`GPU "${label}" ${detail.errorClass}: ${detail.message}`, { cause });
        this.name = "GpuDiagnosticError";
        this.label = label;
        this.errorClass = detail.errorClass;
    }
}

const mb = (bytes: number): string => `${(bytes / (1 << 20)).toFixed(0)} MB`;

/**
 * extends a `device.createBuffer` / `createTexture` descriptor with the allocator's own declaration that
 * this specific call is a lazily-grown pool entry — a buffer or texture that appears on real GPU
 * backpressure (a readback ring's staging slot, a GPU table's staging buffer) rather than deterministically
 * for a fixed scenario at fixed params. The allocator is the one thing that knows this, so the mark
 * travels on the descriptor already crossing `ProfilePlugin`'s patched `createBuffer` / `createTexture`
 * seam — never inferred from the label string, which would silently miss the next such pool.
 * `Profile.lazyBytes` sums every allocation marked `lazy` separately
 * from the exact `bufferBytes` / `textureBytes` totals a byte-budget gate reads.
 */
export interface LazyAlloc {
    /** true when this allocation call is a lazily-grown pool entry (see the interface doc). Omitted or
     *  false for an eager, deterministic allocation — every other `createBuffer` / `createTexture` call
     *  site in the engine. */
    lazy?: boolean;
}

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
    if (!Number.isFinite(bytes)) {
        throw new UnsupportedError(`${label} received a non-finite byte count (${bytes})`);
    }
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
    if (!Number.isFinite(size.width) || !Number.isFinite(size.height)) {
        throw new UnsupportedError(
            `${label} received a non-finite extent (${size.width}×${size.height})`,
        );
    }
    const dim = Math.max(size.width, size.height);
    if (dim > limits.maxTextureDimension2D) {
        throw new UnsupportedError(
            `${label} needs a ${size.width}×${size.height} texture, but the device's ` +
                `maxTextureDimension2D is ${limits.maxTextureDimension2D}. ${remedy}`,
        );
    }
    const layers = size.layers ?? 1;
    if (!Number.isFinite(layers)) {
        throw new UnsupportedError(`${label} received a non-finite layer count (${layers})`);
    }
    if (layers > limits.maxTextureArrayLayers) {
        throw new UnsupportedError(
            `${label} needs ${layers} array layers, but the device's maxTextureArrayLayers is ` +
                `${limits.maxTextureArrayLayers}. ${remedy}`,
        );
    }
}

/**
 * active GPU device with per-frame fence sync
 */
export interface WorldGpu {
    readonly device: GPUDevice;
    /** classification of the adapter that supplied {@link device}; fallback and masked adapters remain visible */
    readonly adapter: AdapterVerdict;
    /**
     * TypeGPU root adopting {@link device}, through which typed buffers, bind groups and pipelines are
     * created; `root.unwrap(...)` returns the raw WebGPU handle. {@link requestGPU} creates one per
     * owning World, so two Worlds sharing a device own distinct typed handles. It has no teardown of
     * its own; the World releases the GPU resources created through it.
     */
    readonly root: TgpuRoot;
    /** monotonically incremented per frame */
    frame: number;
    /** frames submitted but not yet retired by the GPU */
    pending(): number;
    /** register the just-submitted frame's completion fence; tracks {@link pending}, returns the fence */
    sync(): Promise<void>;
    /**
     * named GPU buffers published for cross-system lookup. GPU tables
     * self-register; producers register their static buffers (cube vertices,
     * GlobalTransform rows, …). Consumers (renderers) resolve binding names
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
     * `timestampWrites` because it runs before the pass). A `?.`
     * no-op without the plugin, mirroring {@link span}. A bundle reports its *recorded* draw count;
     * the injected validation runs the same for a replay
     */
    indirect?: (name: string, count: number) => void;
    /**
     * optional pipeline-compile timing hook installed by `ProfilePlugin`, mirroring {@link span} /
     * {@link indirect}. typegpu pipelines are sync-created (`root.unwrap` calls the synchronous
     * `create*Pipeline`), and Dawn defers the real shader compile to the forced {@link precompile}
     * drain — so timing the creation call would report a number that reads like a compile time and
     * isn't one. {@link precompileAll} instead measures each forcer's own `initAsync()` await and
     * reports the resolved span here, called only when at least one element actually awaited an
     * `initAsync` — an array with none (the standard renderer's already-unwrapped raw pipelines, or `[]`)
     * never calls this, since reporting a span for a skip would read like a compile that never ran; a
     * `?.` no-op without the plugin means only timing attribution is conditional.
     */
    precompiled?: (label: string, start: number, end: number) => void;
}

type BoundMethod = (...args: never[]) => unknown;
type MethodCache = Map<PropertyKey, { source: BoundMethod; bound: BoundMethod }>;

function cachedMember(target: object, key: PropertyKey, methods: MethodCache): unknown {
    const source = Reflect.get(target, key, target);
    if (typeof source !== "function") return source;
    const cached = methods.get(key);
    if (cached && cached.source === source) return cached.bound;
    const bound = source.bind(target);
    methods.set(key, { source, bound });
    return bound;
}

/** a generated shader record, for debugging. @internal */
export interface ShaderArtifact {
    label: string;
    stage: string;
    source: string;
    hash: string;
    messages: Array<{
        type: string;
        message: string;
        lineNum: number;
        linePos: number;
        offset: number;
        length: number;
    }>;
    compilationError?: { errorClass: string; message: string };
}

/** the page-side artifact capture a browser driver opts into before application code runs. @internal */
export interface GpuDiagnostics {
    artifacts: ShaderArtifact[];
}

const ARTIFACT_LIMIT = 16;
const _instrumentedDevices = new WeakSet<GPUDevice>();

type ArtifactState = "pending" | "success" | "error";
interface ArtifactScope {
    owner: ArtifactSession;
    label: string;
    artifacts: Set<ShaderArtifact>;
    correlated: Set<ShaderArtifact>;
    pending: Set<Promise<void>>;
    failure?: GpuDiagnosticError;
    active: boolean;
}
interface ArtifactSession {
    device: GPUDevice;
    capture: GpuDiagnostics;
    states: Map<ShaderArtifact, ArtifactState>;
    owners: Map<ShaderArtifact, ArtifactScope>;
    modules: WeakMap<GPUShaderModule, ShaderArtifact>;
    scopes: ArtifactScope[];
}
let _artifactSession: ArtifactSession | undefined;

const artifactCapture = (): GpuDiagnostics | undefined =>
    (globalThis as { __gpuDiagnostics?: GpuDiagnostics }).__gpuDiagnostics;

/** stable 64-bit FNV-1a identity for generated WGSL. @internal */
export function shaderHash(source: string): string {
    let hash = 0xcbf29ce484222325n;
    for (const byte of new TextEncoder().encode(source)) {
        hash ^= BigInt(byte);
        hash = BigInt.asUintN(64, hash * 0x100000001b3n);
    }
    return hash.toString(16).padStart(16, "0");
}

function shaderStage(source: string): string {
    const stages = ["vertex", "fragment", "compute"].filter((stage) =>
        new RegExp(`@${stage}\\b`).test(source),
    );
    return stages.join("+") || "unknown";
}

function retireArtifactSession(): void {
    const session = _artifactSession;
    if (!session) return;
    // A replaced build/device must not keep exact WGSL alive behind an old hung compilation-info call.
    for (const artifact of session.capture.artifacts) artifact.source = "";
    session.capture.artifacts.length = 0;
    session.states.clear();
    session.owners.clear();
    for (const scope of session.scopes) {
        for (const artifact of scope.artifacts) artifact.source = "";
        scope.active = false;
        scope.artifacts.clear();
        scope.correlated.clear();
        scope.pending.clear();
    }
    session.scopes.length = 0;
    _artifactSession = undefined;
}

function beginArtifactSession(device: GPUDevice): void {
    retireArtifactSession();
    const capture = artifactCapture();
    if (!capture) return;
    capture.artifacts.length = 0;
    const session: ArtifactSession = {
        device,
        capture,
        states: new Map(),
        owners: new Map(),
        modules: new WeakMap(),
        scopes: [],
    };
    _artifactSession = session;

    if (_instrumentedDevices.has(device)) return;
    _instrumentedDevices.add(device);
    const original = device.createShaderModule.bind(device);
    device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
        const active = _artifactSession;
        if (!active || active.device !== device || active.capture !== artifactCapture()) {
            return original(descriptor);
        }
        const scope = active.scopes.at(-1);
        // Exact unresolved WGSL is retained only inside a bounded-lifetime validation scope.
        if (!scope) return original(descriptor);
        const source = descriptor.code;
        const artifact: ShaderArtifact = {
            label: descriptor.label || "<unlabeled shader module>",
            stage: shaderStage(source),
            source,
            hash: shaderHash(source),
            messages: [],
        };
        active.states.set(artifact, "pending");
        active.owners.set(artifact, scope);
        scope.artifacts.add(artifact);

        let module: GPUShaderModule;
        try {
            module = original(descriptor);
        } catch (error) {
            artifact.compilationError = failure(error);
            markArtifactError(active, artifact);
            promoteArtifact(active, artifact);
            throw error;
        }
        active.modules.set(module, artifact);
        let pending!: Promise<void>;
        pending = module
            .getCompilationInfo()
            .then((info) => {
                if (!ownsArtifactScope(active, scope)) return;
                artifact.messages = info.messages.map((message) => ({
                    type: message.type,
                    message: message.message,
                    lineNum: message.lineNum,
                    linePos: message.linePos,
                    offset: message.offset,
                    length: message.length,
                }));
                const errors = artifact.messages.filter((message) => message.type === "error");
                if (errors.length > 0) {
                    markArtifactError(active, artifact);
                    const error = Object.assign(
                        new Error(errors.map((message) => message.message).join(" | ")),
                        { name: "GPUCompilationError" },
                    );
                    scope.failure ??= new GpuDiagnosticError(artifact.label, error);
                } else if (active.states.has(artifact) && active.states.get(artifact) !== "error") {
                    active.states.set(artifact, "success");
                }
                promoteArtifact(active, artifact);
            })
            .catch((error) => {
                if (!ownsArtifactScope(active, scope)) return;
                const diagnostic = new GpuDiagnosticError(artifact.label, error);
                artifact.compilationError = failure(error);
                markArtifactError(active, artifact);
                scope.failure ??= diagnostic;
                promoteArtifact(active, artifact);
            })
            .finally(() => scope.pending.delete(pending));
        scope.pending.add(pending);
        return module;
    };

    const correlate = (modules: readonly GPUShaderModule[]) => {
        const active = _artifactSession;
        if (!active || active.device !== device || active.capture !== artifactCapture()) return;
        const scope = active.scopes.at(-1);
        if (!scope) return;
        for (const module of modules) {
            const artifact = active.modules.get(module);
            if (artifact) scope.correlated.add(artifact);
        }
    };
    if (typeof device.createComputePipeline === "function") {
        const create = device.createComputePipeline.bind(device);
        device.createComputePipeline = (descriptor) => {
            correlate([descriptor.compute.module]);
            return create(descriptor);
        };
    }
    if (typeof device.createComputePipelineAsync === "function") {
        const create = device.createComputePipelineAsync.bind(device);
        device.createComputePipelineAsync = (descriptor) => {
            correlate([descriptor.compute.module]);
            return create(descriptor);
        };
    }
    if (typeof device.createRenderPipeline === "function") {
        const create = device.createRenderPipeline.bind(device);
        device.createRenderPipeline = (descriptor) => {
            correlate(
                descriptor.fragment
                    ? [descriptor.vertex.module, descriptor.fragment.module]
                    : [descriptor.vertex.module],
            );
            return create(descriptor);
        };
    }
    if (typeof device.createRenderPipelineAsync === "function") {
        const create = device.createRenderPipelineAsync.bind(device);
        device.createRenderPipelineAsync = (descriptor) => {
            correlate(
                descriptor.fragment
                    ? [descriptor.vertex.module, descriptor.fragment.module]
                    : [descriptor.vertex.module],
            );
            return create(descriptor);
        };
    }
}

function promoteArtifact(session: ArtifactSession, artifact: ShaderArtifact): void {
    if (session.capture.artifacts.includes(artifact)) return;
    const state = session.states.get(artifact);
    if (!state || state === "pending") return;

    const prior = session.capture.artifacts.find((entry) => entry.label === artifact.label);
    if (prior) {
        if (session.states.get(prior) === "error" && state !== "error") {
            artifact.source = "";
            session.states.delete(artifact);
            session.owners.delete(artifact);
            return;
        }
        removeArtifact(session, prior);
    }
    if (session.capture.artifacts.length >= ARTIFACT_LIMIT) {
        const evict = session.capture.artifacts.find((entry) => !pinsArtifact(session, entry));
        if (!evict && !pinsArtifact(session, artifact)) {
            removeArtifact(session, artifact);
            return;
        }
        removeArtifact(session, evict ?? session.capture.artifacts[0]);
    }
    session.capture.artifacts.push(artifact);
}

function pinsArtifact(session: ArtifactSession, artifact: ShaderArtifact): boolean {
    if (session.states.get(artifact) === "error") return true;
    if (session.scopes.some((scope) => scope.active && scope.correlated.has(artifact))) return true;
    const owner = session.owners.get(artifact);
    return !!owner && ownsArtifactScope(session, owner) && artifact.label === owner.label;
}

function removeArtifact(session: ArtifactSession, artifact: ShaderArtifact): void {
    const at = session.capture.artifacts.indexOf(artifact);
    if (at >= 0) session.capture.artifacts.splice(at, 1);
    artifact.source = "";
    session.states.delete(artifact);
    session.owners.get(artifact)?.artifacts.delete(artifact);
    session.owners.delete(artifact);
}

function markArtifactError(session: ArtifactSession, artifact: ShaderArtifact): void {
    if (session.states.has(artifact)) session.states.set(artifact, "error");
}

function ownsArtifactScope(session: ArtifactSession, scope: ArtifactScope): boolean {
    return _artifactSession === session && scope.owner === session && scope.active;
}

async function settleArtifactScope(scope: ArtifactScope | undefined): Promise<void> {
    if (!scope || !ownsArtifactScope(scope.owner, scope)) return;
    await Promise.all([...scope.pending]);
    if (!ownsArtifactScope(scope.owner, scope)) return;
    if (scope.failure) throw scope.failure;
}

function beginArtifactScope(device: GPUDevice, label: string): ArtifactScope | undefined {
    const session = _artifactSession;
    if (!session || session.device !== device) return;
    const scope: ArtifactScope = {
        owner: session,
        label,
        artifacts: new Set(),
        correlated: new Set(),
        pending: new Set(),
        active: true,
    };
    session.scopes.push(scope);
    return scope;
}

function markFailingArtifact(scope: ArtifactScope | undefined): void {
    if (!scope || !ownsArtifactScope(scope.owner, scope)) return;
    for (const artifact of scope.artifacts) {
        if (artifact.label === scope.label) {
            markArtifactError(scope.owner, artifact);
            promoteArtifact(scope.owner, artifact);
        }
    }
    for (const artifact of scope.correlated) {
        markArtifactError(scope.owner, artifact);
        promoteArtifact(scope.owner, artifact);
    }
}

function endArtifactScope(scope: ArtifactScope | undefined): void {
    if (!scope || !ownsArtifactScope(scope.owner, scope)) return;
    const session = scope.owner;
    const at = session.scopes.lastIndexOf(scope);
    if (at >= 0) session.scopes.splice(at, 1);
    scope.active = false;
    scope.pending.clear();
    scope.correlated.clear();
    for (const artifact of scope.artifacts) {
        if (!session.capture.artifacts.includes(artifact)) {
            artifact.source = "";
            session.states.delete(artifact);
        }
        session.owners.delete(artifact);
    }
    scope.artifacts.clear();
}

const _observedDevices = new WeakSet<GPUDevice>();
const _lostDevices = new WeakSet<GPUDevice>();
const _rawDevices = new WeakMap<GPUDevice, GPUDevice>();

/** Native device identity for host APIs such as GPUCanvasContext.configure. Use the world's
 * device for resource creation so allocations remain world-owned. */
export function rawDevice(device: GPUDevice): GPUDevice {
    return _rawDevices.get(device) ?? device;
}

/** whether the original device's loss has been observed; never follows the active build. @internal */
export function deviceLost(device: GPUDevice): boolean {
    return _lostDevices.has(rawDevice(device));
}

function failure(error: unknown): { errorClass: string; message: string } {
    if (error instanceof Error) return { errorClass: error.name, message: error.message };
    if (typeof error === "object" && error !== null) {
        const value = error as { constructor?: { name?: string }; message?: unknown };
        return {
            errorClass: value.constructor?.name ?? "Error",
            message: String(value.message ?? error),
        };
    }
    return { errorClass: "Error", message: String(error) };
}

/**
 * run one labeled operation inside a balanced WebGPU validation scope.
 *
 * Error scopes are a per-device stack: two `validateGpu` operations awaited concurrently on one
 * device pop each other's scopes and lose attribution. Nest them (await fully inside), serialize
 * them (the late-`precompile` chain), or share one scope and re-drain serially on failure
 * (`precompileAll`'s batch-then-bisect).
 * @internal
 */
export async function validateGpu<T>(
    device: GPUDevice,
    label: string,
    operation: () => T | Promise<T>,
): Promise<T> {
    device = rawDevice(device);
    device.pushErrorScope("validation");
    const artifactScope = beginArtifactScope(device, label);
    let value: T | undefined;
    let thrown: unknown;
    let didThrow = false;
    try {
        value = await operation();
    } catch (error) {
        thrown = error;
        didThrow = true;
    }
    if (didThrow) markFailingArtifact(artifactScope);

    try {
        await settleArtifactScope(artifactScope);
    } catch (error) {
        if (!didThrow) {
            thrown = error;
            didThrow = true;
        }
    }

    let scoped: GPUError | null = null;
    try {
        scoped = await device.popErrorScope();
    } catch (error) {
        if (!didThrow) {
            thrown = error;
            didThrow = true;
        }
    }
    const failed = didThrow || scoped !== null;
    if (failed) markFailingArtifact(artifactScope);
    endArtifactScope(artifactScope);
    if (thrown instanceof GpuDiagnosticError) throw thrown;
    if (failed) throw new GpuDiagnosticError(label, didThrow ? thrown : scoped);
    return value as T;
}

/** install the device-wide failure channel without replacing a host's listener. @internal */
export function observeDevice(
    device: GPUDevice,
    report: (message: string) => void = (message) => console.error(message),
): void {
    if (_observedDevices.has(device)) return;
    _observedDevices.add(device);

    void device.lost?.then((info) => {
        _lostDevices.add(device);
        report(`GPU device lost ${info.reason}: ${info.message}`);
    });

    const uncaptured = (event: GPUUncapturedErrorEvent) => {
        const error = failure(event.error);
        report(`GPU uncaptured ${error.errorClass}: ${error.message}`);
    };
    if (typeof device.addEventListener === "function") {
        device.addEventListener("uncapturederror", uncaptured);
    } else {
        const host = device.onuncapturederror;
        device.onuncapturederror = (event) => {
            host?.call(device, event);
            uncaptured(event);
        };
    }
}

// the base floor every shallot app needs (the default renderer + GPU tables). It gates device
// acquisition before any plugin loads, so **a floor entry earns its place only by being a
// `DEFAULT_PLUGINS` need** — anything an opt-in plugin uses belongs on that plugin, via
// `Plugin.features` (required — a missing one throws) or `Plugin.preferredFeatures` (best-effort —
// requested only where the adapter has it, never throws). Hence the deliberate absentees:
// `timestamp-query` is an optional profiler capability. `shader-f16` gates the WGSL `f16` type,
// not `pack2x16float` / `unpack2x16float`. `subgroups` is the standing preferred case: the
// BVH builder (physics broadphase / accel structure) runs a faster subgroup arm where present and an
// LDS arm where absent (WebKit), so it's preferred, not required — a no-subgroup device still loads a
// physics app, on the LDS arm.
export const BASE_FEATURES = [
    "indirect-first-instance",
    // a fused postfx composite writes the swapchain from a compute pass; on Mac/Windows the
    // preferred canvas format is bgra8unorm, and a storage view of it needs this feature
    "bgra8unorm-storage",
    // the default HDR scene offscreen + the standard renderer's MSAA color target are rg11b10ufloat:
    // grants it render-attachment + multisample + resolve. Half the bandwidth of rgba16float at
    // 4× MSAA, on the whole floor (desktop / Steam Deck / recent Android all support it)
    "rg11b10ufloat-renderable",
] as const;

/** shallot's per-stage storage buffer floor, requested as the ceiling across all bind groups. 99.6% of WebGPU
 *  devices support 10. */
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
 */
export const tgslCanary = tgpu.fn(
    [u32],
    u32,
)((x) => {
    "use gpu";
    return x + 1;
});

// Two copies in one bundle share the `__TYPEGPU_META__` global and delete from it, so they race.
// typegpu's module top level writes `__TYPEGPU_VERSION__` on every evaluation, a same-version
// duplicate included, so comparing values cannot see two copies of one version. Counting writes can:
// the key becomes an accessor, and any later evaluation, whatever version it stamps, pushes the count
// past 1. A write that lands before this block runs is folded into the baseline and goes unseen.
const _globals = globalThis as unknown as Record<string, unknown>;
const _typegpuVersion = _globals.__TYPEGPU_VERSION__ as string | undefined;
if (_globals.__SHALLOT_TYPEGPU_WRITES__ === undefined) {
    let _liveVersion = _typegpuVersion;
    Object.defineProperty(_globals, "__TYPEGPU_VERSION__", {
        configurable: true,
        get: () => _liveVersion,
        set: (v: string | undefined) => {
            _globals.__SHALLOT_TYPEGPU_WRITES__ =
                (_globals.__SHALLOT_TYPEGPU_WRITES__ as number) + 1;
            _liveVersion = v;
        },
    });
    _globals.__SHALLOT_TYPEGPU_WRITES__ = 1;
}

/**
 * Throws when a second typegpu copy has written `__TYPEGPU_VERSION__` since this module loaded, or when
 * {@link tgslCanary} fails to resolve because the bundle skipped the typegpu transform.
 * {@link requestGPU} calls it before touching the adapter.
 *
 * It proves only that the engine's own modules went through the transform, not the caller's: source
 * outside the transform's file filter still fails at its own kernel.
 *
 * It does not refuse:
 *
 * - a duplicate typegpu that evaluates before this module, such as a dev-optimizer prebundled chunk;
 *   its write lands before the counter exists;
 * - a bundle without metadata whose canary still resolves. typegpu's runtime fallback derives WGSL
 *   from the canary's plain function body, and a plain `tgpu.fn` lacks the entry-point output-struct
 *   cast that the fallback gets wrong, so the failure surfaces only when a real pipeline compiles.
 *
 * The first is invisible to any counter this module installs, and the second needs a compiled
 * pipeline, which a check that runs before the adapter cannot create. Only booting an installed
 * project on a real device observes them.
 */
export function checkTgsl(): void {
    const writes = _globals.__SHALLOT_TYPEGPU_WRITES__ as number;
    if (writes > 1) {
        throw new Error(
            `Two copies of typegpu are loaded (first stamped ${_typegpuVersion}; the key has been ` +
                `written to ${writes} times since). They share one metadata map and delete from it, so ` +
                "kernels resolve empty at random. Dedupe typegpu to a single copy — it is a " +
                "peerDependency of the engine for exactly this reason.",
        );
    }
    try {
        tgpu.resolve([tgslCanary]);
    } catch (cause) {
        throw new Error(
            "TGSL metadata is missing — this bundle was built without the typegpu transform, so every " +
                "engine shader would resolve wrong. Add `shallot()` from `@dylanebert/shallot/vite` to " +
                "your vite config, or register `unplugin-typegpu/bun` in a bun preload.",
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
export const precompileState = {
    create: () => ({
        precompile: [] as Forcer[],
        labels: new Set<string>(),
        scopes: new Map<string, number>(),
        order: 0,
        draining: false,
        drained: false,
        late: Promise.resolve(),
    }),
};

function compile({ label, force }: Forcer): unknown {
    let forced: unknown;
    try {
        forced = force();
    } catch (cause) {
        throw new Error(`precompile "${label}" failed — its pipeline did not compile`, { cause });
    }
    // a forcer that binds nothing has no pipeline to init, so the compile silently falls through to
    // the first frame — the exact stall the queue exists to prevent, and invisible without this
    if (!forced)
        throw new Error(
            `precompile "${label}" bound nothing — its pipeline would compile on the first frame instead`,
        );
    return forced;
}

/**
 * force a pipeline to compile before the first frame. A typegpu pipeline is created synchronously
 * (`root.unwrap` calls the synchronous `create*Pipeline`), and Dawn can defer the real compile to the
 * first dispatch (measured ~3 s of first-frame drain at engine scale). A pipeline owner registers its
 * bound pipeline from `warm`; `createApp` drains the queue once every plugin has warmed, awaiting
 * `initAsync()` on each returned pipeline, so the compile is paid under the loading screen. Registered
 * *after* that drain (a lazily-built pipeline, a post-warm producer), the drain runs on arrival and the
 * returned promise must be awaited — late is better than silently dropped, but it still owes the same
 * validation.
 *
 * `force` **returns the bound pipeline** — never dispatch from the callback: a zero-workgroup dispatch
 * trips Dawn's `DispatchWorkgroups with a workgroup count of 0 is unusual` warning in your own code. The
 * drain classifies the return exhaustively: a typegpu pipeline (compute / render / guarded) is awaited
 * via its `initAsync`; an **array** is awaited element-wise — each entry exposing `initAsync` is
 * awaited, each that doesn't (the standard renderer's already-unwrapped raw pipelines) is skipped, and `[]`
 * (nothing specializes) awaits nothing; anything else truthy is a labelled throw. A nullish
 * return also throws, because a forcer whose buffers aren't up yet no-ops and hands the compile back to
 * frame one without a word. Allocate inside the thunk if the buffers are late — the drain runs after
 * every plugin's warm, which is the point. `label` names the pipeline in either failure and must be
 * unique within the build. `options.after` names other queued labels that must drain first. Unknown
 * labels are ignored because the plugin that owns a predecessor may be absent.
 */
export function precompile(
    world: World,
    label: string,
    force: () => unknown,
    options: { after?: readonly string[] } = {},
): Promise<void> {
    const _precompileState = world.resource(precompileState);

    if (_precompileState.labels.has(label)) {
        throw new Error(`duplicate precompile label "${label}"`);
    }
    _precompileState.labels.add(label);
    const forcer = { label, force, after: options.after ?? [], order: _precompileState.order++ };
    if (_precompileState.drained) {
        const completion = _precompileState.late.then(
            () => compileValidated(world, forcer),
            () => compileValidated(world, forcer),
        );
        // Device error scopes are a stack: serialize unrelated late factories so their pops stay LIFO.
        _precompileState.late = completion.catch(() => {});
        return completion;
    }
    _precompileState.precompile.push(forcer);
    // During warm, build owns the completion contract through precompileAll; registration itself is done.
    return Promise.resolve();
}

/**
 * a unique {@link precompile} label prefix for a factory an app can instantiate more than once (the
 * BVH stages: a scene builds one BVH, the physics broadphase another). The first instance keeps the
 * bare `prefix`, so a single-instance app's labels — and their profiler rows — read unchanged; every
 * later one gets `prefix-2`, `prefix-3`, … Counts and labels belong to the supplied World.
 *
 * A scoped label is therefore not a fixed string, so it can't be named by another forcer's `after`
 * (which would silently degrade to the missing-predecessor case). Scope only a factory nothing
 * orders against.
 */
export function precompileScope(world: World, prefix: string): string {
    const _precompileState = world.resource(precompileState);

    const n = (_precompileState.scopes.get(prefix) ?? 0) + 1;
    _precompileState.scopes.set(prefix, n);
    return n === 1 ? prefix : `${prefix}-${n}`;
}

/**
 * Partitions forcers into dependency levels (Kahn's algorithm): level 0 has no queued `after`
 * predecessor, level n depends only on earlier levels. {@link precompileAll} drains each level
 * concurrently under one shared validation scope. Within a level, registration `order` fixes the
 * order members start in and the label order the shared scope reports. Throws on a cycle.
 */
function ordered(forcers: readonly Forcer[]): Forcer[][] {
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

    const levels: Forcer[][] = [];
    let ready = forcers.filter((forcer) => incoming.get(forcer.label) === 0);
    ready.sort((a, b) => a.order - b.order);
    let drained = 0;
    while (ready.length > 0) {
        levels.push(ready);
        drained += ready.length;
        const next: Forcer[] = [];
        for (const forcer of ready) {
            for (const dependent of outgoing.get(forcer.label)!) {
                const count = incoming.get(dependent.label)! - 1;
                incoming.set(dependent.label, count);
                if (count === 0) next.push(dependent);
            }
        }
        next.sort((a, b) => a.order - b.order);
        ready = next;
    }
    if (drained !== forcers.length) {
        const cycle = forcers
            .filter((forcer) => incoming.get(forcer.label)! > 0)
            .map((forcer) => forcer.label);
        throw new Error(`precompile cycle: ${cycle.join(" -> ")}`);
    }
    return levels;
}

/**
 * Compiles one forcer with no error scope of its own, so a level's batch can run several inside one
 * shared scope. Timing brackets only this forcer's work, not the shared scope's push and pop.
 */
async function compileBody(
    forcer: Forcer,
): Promise<{ warmed: boolean; start: number; end: number }> {
    const start = now();
    let warmed = false;
    const pipeline = compile(forcer);
    // the drain classifies a forcer's return exhaustively:
    // (a) a typegpu pipeline (compute / render / guarded) exposes initAsync → await it
    if (typeof (pipeline as { initAsync?: unknown }).initAsync === "function") {
        await (pipeline as { initAsync(): Promise<void> }).initAsync();
        warmed = true;
    } else if (Array.isArray(pipeline)) {
        // (b) an array is awaited element-wise — the natural generalization of the standard renderer's
        // shape is an array of typegpu pipelines, and skipping the whole array unconditionally
        // would silently warm nothing for that caller. Each entry exposing initAsync is awaited;
        // the standard renderer's (standard/rendering/forward.ts) own already-unwrapped raw pipelines expose
        // none, so they're skipped element-wise too, same as `[]` when nothing specializes. `warmed`
        // tracks whether any entry actually did — an all-skip array must not report a compile span
        // below, since that's the one still-unwarmed path and the profiler must not call it warm
        const awaited = await Promise.all(
            pipeline.map((entry) =>
                typeof (entry as { initAsync?: unknown })?.initAsync === "function"
                    ? (entry as { initAsync(): Promise<void> }).initAsync().then(() => true)
                    : false,
            ),
        );
        warmed = awaited.some(Boolean);
    } else {
        // (c) anything else truthy is a forcer that returned the wrong shape — name the site
        throw new Error(
            `precompile "${forcer.label}" returned a value that is neither a pipeline with initAsync nor an array of raw pipelines`,
        );
    }
    const end = now();
    return { warmed, start, end };
}

/**
 * User Timing measure-name prefix for a compiled forcer's `pipeline_compile` vital
 * (`{@link PIPELINE_COMPILE_MEASURE_PREFIX}${forcer.label}`). A page-side RUM script can't reach
 * this bundle's world-owned GPU hooks — each demo bundles the engine itself, so `performance.measure`
 * is the cross-bundle wire; the constant is exported so a site-side script can filter on it without
 * a second engine copy. Stable string contract: the deployed demos run the published engine while
 * an importer builds from workspace source, so renaming this silently breaks that wire.
 *
 * Async-only coverage: a measure is emitted only when a forcer actually awaited `initAsync`
 * ({@link compileBody}'s `warmed`), the same gate `world.gpu.precompiled` reports through — a
 * non-forced sync pipeline records a near-zero stub (TypeGPU returns before the driver compiles),
 * so no vital is emitted for it.
 * @internal
 */
export const PIPELINE_COMPILE_MEASURE_PREFIX = "shallot:pipeline-compile:";

/**
 * Reports one forcer's compile span through `world.gpu.precompiled` and a paired
 * `performance.measure` entry. A forcer that awaited no `initAsync` compiled nothing here, so it
 * reports nothing.
 */
function reportCompile(
    world: World,
    forcer: Forcer,
    warmed: boolean,
    start: number,
    end: number,
): void {
    if (!warmed) return;
    world.gpu.precompiled?.(forcer.label, start, end);
    // telemetry must never throw into the validation path — a User Timing entry is a nice-to-have
    // for the site's RUM script, not a build-breaking dependency, so a missing or throwing
    // `performance.measure` (an older runtime, a locked-down embedder) is swallowed.
    try {
        if (typeof performance?.measure === "function") {
            performance.measure(`${PIPELINE_COMPILE_MEASURE_PREFIX}${forcer.label}`, {
                start,
                end,
            });
        }
    } catch {
        // never let telemetry break a build
    }
}

/** Compiles one forcer under its own validation scope, so a failure names it. Used for a
 *  single-member level, for a forcer registered after the drain, and to re-drain a failed batch. */
async function compileValidated(world: World, forcer: Forcer): Promise<void> {
    const { warmed, start, end } = await validateGpu(world.gpu.device, forcer.label, () =>
        compileBody(forcer),
    );
    reportCompile(world, forcer, warmed, start, end);
}

/**
 * Removes `drained` from the queue by identity. A {@link precompile} call during a batch's await
 * appends to the live queue and has already returned, so nothing would retry it; rebuilding the queue
 * from a snapshot taken before the await would drop it.
 */
function removeForcers(world: World, drained: readonly Forcer[]): void {
    const _precompileState = world.resource(precompileState);

    if (drained.length === 0) return;
    const set = new Set(drained);
    for (let i = _precompileState.precompile.length - 1; i >= 0; i--) {
        if (set.has(_precompileState.precompile[i])) _precompileState.precompile.splice(i, 1);
    }
}

/**
 * Drains the {@link precompile} queue level by level; `createApp` calls it after every plugin's
 * `warm`. A forcer registered during a drain joins the queue and drains in a later iteration.
 *
 * A multi-member level compiles concurrently under one shared validation scope. Error scopes are a
 * per-device stack, so a failed shared scope cannot say which member failed: the level then
 * re-drains serially from its start, so the throw names a forcer, and each member of a failing level
 * compiles twice. The serial re-drain stops at the first throw; the rest of the level and every
 * later level stay queued for a later call.
 * @internal
 */
export async function precompileAll(world: World): Promise<void> {
    const _precompileState = world.resource(precompileState);

    if (_precompileState.draining) return;
    _precompileState.draining = true;
    try {
        while (_precompileState.precompile.length > 0) {
            const levels = ordered(_precompileState.precompile);
            const level = levels[0];

            if (level.length === 1) {
                // splices before its one await — nothing can be appended between reading `levels`
                // and this splice, so a stale snapshot here is not reachable.
                const rest = levels.slice(1).flat();
                _precompileState.precompile.splice(0, _precompileState.precompile.length, ...rest);
                await compileValidated(world, level[0]);
                continue;
            }

            const scopeLabel = level.map((forcer) => forcer.label).join(", ");
            let results:
                | { forcer: Forcer; warmed: boolean; start: number; end: number }[]
                | undefined;
            try {
                results = await validateGpu(world.gpu.device, scopeLabel, () =>
                    Promise.all(
                        level.map(async (forcer) => ({ forcer, ...(await compileBody(forcer)) })),
                    ),
                );
            } catch {
                results = undefined;
            }

            if (results) {
                removeForcers(world, level);
                for (const { forcer, warmed, start, end } of results) {
                    reportCompile(world, forcer, warmed, start, end);
                }
                continue;
            }

            // batch-then-bisect: re-drain the level serially, from its start through the thrower.
            let ranThrough = 0;
            try {
                for (; ranThrough < level.length; ranThrough++) {
                    await compileValidated(world, level[ranThrough]);
                }
            } finally {
                // indices [0, ranThrough] actually ran (succeeded, or the thrower itself) — remove
                // exactly those, by identity, so anything appended during any of these awaits (the
                // batch attempt's, or a serial member's) survives into the next iteration.
                removeForcers(world, level.slice(0, ranThrough + 1));
            }
        }
        _precompileState.drained = true;
    } finally {
        _precompileState.draining = false;
    }
}

// TypeGPU's root and resource handles belong to the World using them, even when two Worlds share a device.
export const typegpuRoot: Resource<{ root?: TgpuRoot }> = { create: () => ({}) };

function adopt(
    device: GPUDevice,
    owner?: { resource?: <T>(declaration: Resource<T>) => T },
): TgpuRoot {
    const create = () => tgpu.initFromDevice({ device });
    if (!owner?.resource) return create();
    const value = owner.resource(typegpuRoot);
    return (value.root ??= create());
}

/** classify the adapter and warn once when the result is not real hardware. */
export function stampAdapter(
    adapter?: GPUAdapter,
    notice?: (verdict: AdapterVerdict) => void,
): AdapterVerdict {
    const verdict =
        adapter === undefined
            ? classifyAdapter({ present: true })
            : classifyAdapter({
                  present: true,
                  info: adapter.info as AdapterInfoFacts,
              });
    if (verdict.class !== "real") {
        console.warn(
            `[shallot] ${verdict.reason ?? `${verdict.class} adapter: ${verdict.identity}`}`,
        );
        notice?.(verdict);
    }
    return verdict;
}

/**
 * Creates a world GPU context. Without `device`, acquires one through `navigator.gpu` and enforces
 * the feature floor (the base floor plus the active plugins' `features`), throwing
 * {@link UnsupportedError} otherwise; `preferred` features are requested only where the adapter has
 * them. A supplied device is adopted as-is, and the caller owns its feature support. Either way the
 * context's {@link WorldGpu.root} adopts the device, one root per owning World even when two Worlds
 * share a device.
 */
export async function requestGPU(
    device?: GPUDevice,
    features: readonly GPUFeatureName[] = [],
    preferred: readonly GPUFeatureName[] = [],
    adapter?: GPUAdapter,
    owner?: {
        own(resource: { destroy(): void }): void;
        resource?: <T>(declaration: Resource<T>) => T;
    },
): Promise<WorldGpu> {
    // before anything resolves: typegpu binds the console method a TGSL `console.log` calls at
    // shader-generation time, so a capture installed later never sees that kernel's lines.
    captureGpuLog();
    checkTgsl();
    const acquired =
        device === undefined ? await acquireDevice(features, preferred) : { device, adapter };
    const d = rawDevice(acquired.device);
    const verdict = stampAdapter(acquired.adapter);
    observeDevice(d);
    beginArtifactSession(d);
    let inFlight = 0;
    // one settle reaction for every fence, resolved or rejected
    const settle = (): void => {
        inFlight--;
    };
    const trackedDevice = owner
        ? (() => {
              const deviceOverrides = new Map<PropertyKey, unknown>();
              // Capabilities are fixed at device acquisition; native getters need not rebuild wrappers in play.
              const limits = d.limits;
              const granted = d.features;
              const queue = d.queue;
              const queueOverrides = new Map<PropertyKey, unknown>();
              const queueMethods: MethodCache = new Map();
              const deviceMethods: MethodCache = new Map();
              const trackedQueue = new Proxy(queue, {
                  get(target, key) {
                      if (queueOverrides.has(key)) return queueOverrides.get(key);
                      return cachedMember(target, key, queueMethods);
                  },
                  set(_target, key, value) {
                      queueOverrides.set(key, value);
                      return true;
                  },
              });
              const createBuffer = d.createBuffer.bind(d);
              const createTexture = d.createTexture.bind(d);
              const ownedBuffer = (descriptor: GPUBufferDescriptor) => {
                  const buffer = createBuffer(descriptor);
                  if (descriptor.label !== undefined && buffer.label !== descriptor.label)
                      buffer.label = descriptor.label;
                  owner.own(buffer);
                  return buffer;
              };
              const ownedTexture = (descriptor: GPUTextureDescriptor) => {
                  const texture = createTexture(descriptor);
                  if (descriptor.label !== undefined && texture.label !== descriptor.label)
                      texture.label = descriptor.label;
                  owner.own(texture);
                  return texture;
              };
              return new Proxy(d, {
                  get(target, key) {
                      if (deviceOverrides.has(key)) return deviceOverrides.get(key);
                      if (key === "queue") return trackedQueue;
                      if (key === "limits") return limits;
                      if (key === "features") return granted;
                      if (key === "createBuffer") return ownedBuffer;
                      if (key === "createTexture") return ownedTexture;
                      return cachedMember(target, key, deviceMethods);
                  },
                  set(_target, key, value) {
                      deviceOverrides.set(key, value);
                      return true;
                  },
              });
          })()
        : d;
    _rawDevices.set(trackedDevice, d);
    const root = adopt(trackedDevice, owner);
    const rootMethods: MethodCache = new Map();
    const ownedRootBuffer = (...args: unknown[]) => {
        const resource = (root.createBuffer as (...args: unknown[]) => unknown)(...args);
        owner?.own(resource as { destroy(): void });
        return resource;
    };
    const ownedRootTexture = (...args: unknown[]) => {
        const resource = (root.createTexture as (...args: unknown[]) => unknown)(...args);
        owner?.own(resource as { destroy(): void });
        return resource;
    };
    const trackedRoot = owner
        ? new Proxy(root, {
              get(target, key) {
                  if (key === "createBuffer") return ownedRootBuffer;
                  if (key === "createTexture") return ownedRootTexture;
                  return cachedMember(target, key, rootMethods);
              },
          })
        : root;
    const compute: WorldGpu = {
        device: trackedDevice,
        adapter: verdict,
        root: trackedRoot,
        frame: 0,
        pending: () => inFlight,
        sync: () => {
            inFlight++;
            const fence = d.queue.onSubmittedWorkDone();
            fence.then(settle, settle);
            return fence;
        },
        buffers: new Map<string, GPUBuffer>(),
        textures: new Map<string, GPUTexture>(),
        samplers: new Map<string, GPUSampler>(),
        typed: new Map<string, TgpuBuffer<AnyData>>(),
    };

    return compute;
}

function gpuRuntimeName(): string {
    const versions = (
        globalThis as typeof globalThis & {
            process?: { versions?: Record<string, string | undefined> };
        }
    ).process?.versions;
    if (versions?.bun) return "Bun";
    if (versions?.node) return "Node.js";
    return "this runtime";
}

function failureMessage(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
}

async function acquireDevice(
    extra: readonly GPUFeatureName[],
    preferred: readonly GPUFeatureName[],
): Promise<{ device: GPUDevice; adapter: GPUAdapter }> {
    const gpu = typeof navigator === "undefined" ? undefined : navigator.gpu;
    if (!gpu) {
        const runtime = gpuRuntimeName();
        const fix =
            runtime === "Bun"
                ? " Install the optional bun-webgpu peer dependency to enable GPU builds."
                : "";
        throw new UnsupportedError(
            `WebGPU unavailable: navigator.gpu is missing in ${runtime}.${fix}`,
        );
    }

    const runtime = gpuRuntimeName();
    let adapter: GPUAdapter | null;
    try {
        adapter = await gpu.requestAdapter();
    } catch (cause) {
        throw new UnsupportedError(
            `WebGPU adapter request failed in ${runtime}: ${failureMessage(cause)}`,
            [],
            { cause },
        );
    }
    if (!adapter) throw new UnsupportedError(`No WebGPU adapter is available in ${runtime}.`);

    const required = [...new Set<GPUFeatureName>([...BASE_FEATURES, ...extra])];
    const { granted, missing } = resolveFeatures(adapter.features, required, preferred);
    if (missing.length > 0) throw new UnsupportedError("Missing required WebGPU features", missing);

    if (adapter.limits.maxStorageBuffersPerShaderStage < REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
        throw new UnsupportedError(
            `Only ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; ${REQUIRED_STORAGE_BUFFERS_PER_STAGE} required`,
        );
    }

    const requiredLimits: Record<string, number> = {
        maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    };
    // Older implementations expose the split-stage limits as zero even though the unified limit
    // governs them. Request zero explicitly so their requestDevice wrappers don't substitute the
    // newer spec defaults as impossible requirements.
    for (const limit of [
        "maxStorageBuffersInVertexStage",
        "maxStorageBuffersInFragmentStage",
        "maxStorageTexturesInVertexStage",
        "maxStorageTexturesInFragmentStage",
    ] as const) {
        if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
    }

    let device: GPUDevice;
    try {
        device = await adapter.requestDevice({
            requiredFeatures: [...required, ...granted],
            requiredLimits,
        });
    } catch (cause) {
        throw new UnsupportedError(
            `WebGPU device creation failed in ${runtime}: ${failureMessage(cause)}`,
            [],
            { cause },
        );
    }

    return { device, adapter };
}
