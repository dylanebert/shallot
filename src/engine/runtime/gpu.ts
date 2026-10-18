import tgpu, { type TgpuRoot } from "typegpu";
import { u32 } from "typegpu/data";
import type { Resource, World } from "../ecs";
import { type AdapterInfoFacts, type AdapterVerdict, classifyAdapter } from "./adapter";
import { continueSubmissionFence, submissionFence } from "./allocation-site";
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

export interface PipelineStorageBindingCounts {
    vertex: number;
    fragment: number;
    compute: number;
}

const PIPELINE_STAGES = [
    ["vertex", 1],
    ["fragment", 2],
    ["compute", 4],
] as const;

/** Count storage-buffer declarations across every group, once for each visible shader stage. */
export function countPipelineStorageBindings(
    groups: readonly (readonly GPUBindGroupLayoutEntry[])[],
): PipelineStorageBindingCounts {
    const counts: PipelineStorageBindingCounts = { vertex: 0, fragment: 0, compute: 0 };
    for (const group of groups) {
        for (const entry of group) {
            if (entry.buffer?.type !== "storage" && entry.buffer?.type !== "read-only-storage")
                continue;
            for (const [stage, mask] of PIPELINE_STAGES) {
                if ((entry.visibility & mask) !== 0) counts[stage]++;
            }
        }
    }
    return counts;
}

/** Refuse a pipeline before WebGPU creation when any bound group exceeds the granted stage limit. */
export function checkPipelineStorageBindings(
    plugin: string,
    pipeline: string,
    counts: PipelineStorageBindingCounts,
    limit: number,
): void {
    for (const [stage] of PIPELINE_STAGES) {
        const count = counts[stage];
        if (count > limit) {
            throw new UnsupportedError(
                `Plugin "${plugin}" pipeline "${pipeline}" uses ${count} ${stage} storage-buffer bindings; ` +
                    `maxStorageBuffersPerShaderStage is ${limit}`,
            );
        }
    }
}

function wgslFunctions(code: string): Map<string, string> {
    const functions = new Map<string, string>();
    const signature = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
    for (let match = signature.exec(code); match; match = signature.exec(code)) {
        const open = code.indexOf("{", signature.lastIndex);
        if (open < 0) break;
        let depth = 1;
        let close = open + 1;
        for (; close < code.length && depth > 0; close++) {
            if (code[close] === "{") depth++;
            else if (code[close] === "}") depth--;
        }
        if (depth !== 0) break;
        functions.set(match[1]!, code.slice(open + 1, close - 1));
        signature.lastIndex = close;
    }
    return functions;
}

/** Count statically-used storage globals for one entry point when WebGPU infers an `auto` layout. */
function wgslEntryStorageBindings(
    code: string,
    entryPoint: string | undefined,
    stage: "vertex" | "fragment" | "compute",
): number | undefined {
    const functions = wgslFunctions(code);
    const entry =
        entryPoint ??
        code.match(new RegExp(`@${stage}\\b(?:[^\\n]*\\n)*?\\s*fn\\s+([A-Za-z_]\\w*)\\s*\\(`))?.[1];
    if (!entry || !functions.has(entry)) return undefined;
    const storageNames = [
        ...code.matchAll(
            /@group\s*\(\s*\d+\s*\)\s*@binding\s*\(\s*\d+\s*\)\s*var\s*<\s*storage\b[^>]*>\s*([A-Za-z_]\w*)\s*:/g,
        ),
    ].map((match) => match[1]!);
    const visited = new Set<string>();
    const pending = [entry];
    const identifier = /\b([A-Za-z_]\w*)\s*\(/g;
    while (pending.length > 0) {
        const name = pending.pop()!;
        if (visited.has(name)) continue;
        visited.add(name);
        const body = functions.get(name)!;
        for (let match = identifier.exec(body); match; match = identifier.exec(body)) {
            if (functions.has(match[1]!) && !visited.has(match[1]!)) pending.push(match[1]!);
        }
    }
    const reachable = [...visited].map((name) => functions.get(name)!).join("\\n");
    return new Set(storageNames.filter((name) => new RegExp(`\\b${name}\\b`).test(reachable))).size;
}

function autoPipelineStorageBindings(
    shaderSources: WeakMap<GPUShaderModule, string>,
    descriptor: GPUComputePipelineDescriptor | GPURenderPipelineDescriptor,
): PipelineStorageBindingCounts | undefined {
    const counts: PipelineStorageBindingCounts = { vertex: 0, fragment: 0, compute: 0 };
    if ("compute" in descriptor) {
        const code = shaderSources.get(descriptor.compute.module);
        if (!code) return undefined;
        const count = wgslEntryStorageBindings(code, descriptor.compute.entryPoint, "compute");
        if (count === undefined) return undefined;
        counts.compute = count;
        return counts;
    }
    for (const [stage, state] of [
        ["vertex", descriptor.vertex],
        ["fragment", descriptor.fragment],
    ] as const) {
        if (!state) continue;
        const code = shaderSources.get(state.module);
        if (!code) return undefined;
        const count = wgslEntryStorageBindings(code, state.entryPoint, stage);
        if (count === undefined) return undefined;
        counts[stage] = count;
    }
    return counts;
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
 * Refuse bytes above `maxStorageBufferBindingSize` with an {@link UnsupportedError} naming
 * the buffer, required and available MB, and the caller-supplied remedy. Also refuses non-finite bytes.
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
 * Refuse width or height above `maxTextureDimension2D`, or layers above `maxTextureArrayLayers`,
 * with an {@link UnsupportedError} naming the extent, limits and caller-supplied remedy.
 * `layers` defaults to 1. Refuses non-finite extents or layer counts.
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
    /** Increments after each successful World.step, including steps that submit no commands. */
    frame: number;
    /** Issued sync fences not yet settled; either fulfillment or rejection retires one. */
    pending(): number;
    /** Register a completion fence after submitted queue work; tracks {@link pending}, returns the fence. */
    sync(): Promise<void>;
    /** Completion watermarks for the existing sync fences, not additional submissions.
     * A resource used before the next sync may recycle when that fence completes. */
    readonly fences: { readonly issued: number; readonly completed: number };
    /** optional GPU timestamp slot allocator hook; returns writes for a pass descriptor */
    span?: (
        name: string,
    ) => GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined;
    /**
     * Optional indirect-draw tally hook installed by `ProfilePlugin`. Report issued commands after
     * skips; a bundle reports its recorded draw count on replay.
     * Dawn's injected indirect-draw validation runs before the pass, outside `timestampWrites`.
     */
    indirect?: (name: string, count: number) => void;
    /**
     * Optional pipeline-compile timing hook. Reports a forcer's own span only when at least one
     * returned pipeline awaited `initAsync`; excludes shared validation-scope push and pop.
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
const _trackedDevices = new WeakSet<GPUDevice>();

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

    if (_trackedDevices.has(device)) return;
    _trackedDevices.add(device);
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
interface PipelineGuardState {
    plugin?: string;
    bindGroups: WeakMap<GPUBindGroupLayout, PipelineStorageBindingCounts>;
    layouts: WeakMap<GPUPipelineLayout, PipelineStorageBindingCounts | null>;
}
const _pipelineGuards = new WeakMap<GPUDevice, PipelineGuardState>();

/** Attribute synchronous pipeline creation inside a plugin lifecycle or precompile to its owning plugin. @internal */
export function withPipelineOwner<T>(
    device: GPUDevice | undefined,
    plugin: string | undefined,
    operation: () => T,
): T {
    const guard = device && _pipelineGuards.get(device);
    if (!guard || !plugin) return operation();
    const previous = guard.plugin;
    guard.plugin = plugin;
    try {
        return operation();
    } finally {
        guard.plugin = previous;
    }
}

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

/** GPU capabilities declared by a plugin that uses the device. Presence of this value, including `{}`, requires a device. */
export interface GpuRequirements {
    readonly features?: readonly GPUFeatureName[];
    readonly preferredFeatures?: readonly GPUFeatureName[];
    /** `max*` values are lower bounds; `min*Alignment` values are upper bounds. */
    readonly limits?: Readonly<Partial<GPUSupportedLimits>>;
}

/** A plugin's GPU declaration, kept with its owner through capability checks. */
export interface NamedGpuRequirements extends GpuRequirements {
    readonly plugin: string;
}

/**
 * split required and preferred features against what an adapter offers: return absent required features
 * and grant only available preferred features not already required. The caller decides how to report any
 * missing requirements.
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
 * TGSL resolution canary used by {@link checkTgsl}; resolution alone does not prove the build ran
 * the transform, because TypeGPU can fall back to parsing a plain function body.
 */
export const tgslCanary = tgpu.fn(
    [u32],
    u32,
)((x) => {
    "use gpu";
    return x + 1;
});

// TypeGPU copies share and delete from __TYPEGPU_META__, so duplicate evaluations race.
// Count version writes to detect same-version duplicates too; earlier writes remain invisible.
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
 * It does not check caller source outside the transform's file filter; that source still fails at its own kernel.
 *
 * It does not refuse:
 *
 * - a duplicate typegpu that evaluates before this module, such as a dev-optimizer prebundled chunk;
 *   its write lands before the counter exists;
 * - a bundle without metadata whose canary still resolves. typegpu's runtime fallback derives WGSL
 *   from the canary's plain function body, and a plain `tgpu.fn` lacks the entry-point output-struct
 *   cast that the fallback gets wrong, so the failure surfaces only when a real pipeline compiles.
 * These cases require an installed-project pipeline check on a real device.
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
    owner?: string;
}

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
    // Missing pipelines would silently defer compilation to the first frame.
    if (!forced)
        throw new Error(
            `precompile "${label}" returned nothing — its pipeline would compile on the first frame instead`,
        );
    return forced;
}

/**
 * Queue pipeline compilation during `warm`; `createApp` drains after all plugins warm.
 * After that drain, compilation is serialized on arrival and the returned promise must be awaited.
 * Dawn can defer synchronous pipeline compilation until first dispatch; `force` must return pipelines,
 * not dispatch them. A zero-workgroup dispatch trips Dawn's unusual-workgroup-count warning.
 * `initAsync` compiles the pipeline core shared by its `.with()` bindings. Already-unwrapped pipelines
 * were created synchronously and have nothing left to await.
 * A single return must expose `initAsync`. Arrays await each entry exposing `initAsync` and silently
 * skip other entries; an empty array awaits nothing. Falsy returns and other single values throw by label.
 * Labels must be unique within the build. `options.after` orders queued labels; absent labels are silently
 * ignored because their plugin may be absent. `options.owner` attributes pipeline-limit refusals after
 * an async warm boundary.
 */
export function precompile(
    world: World,
    label: string,
    force: () => unknown,
    options: { after?: readonly string[]; owner?: string } = {},
): Promise<void> {
    const _precompileState = world.resource(precompileState);

    if (_precompileState.labels.has(label)) {
        throw new Error(`duplicate precompile label "${label}"`);
    }
    _precompileState.labels.add(label);
    const forcer = {
        label,
        force,
        after: options.after ?? [],
        order: _precompileState.order++,
        owner: options.owner ?? _pipelineGuards.get(world.gpu.device)?.plugin,
    };
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
 * World-local {@link precompile} label prefix: first use returns `prefix`, then `prefix-2`, `prefix-3`, …
 * Use only for factories nothing orders against; another forcer's fixed `after` label would silently
 * miss dynamically scoped instances.
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
    world: World,
    forcer: Forcer,
): Promise<{ warmed: boolean; start: number; end: number }> {
    const start = now();
    let warmed = false;
    const pipeline = withPipelineOwner(world.gpu.device, forcer.owner, () => compile(forcer));
    if (typeof (pipeline as { initAsync?: unknown }).initAsync === "function") {
        await withPipelineOwner(world.gpu.device, forcer.owner, () =>
            (pipeline as { initAsync(): Promise<void> }).initAsync(),
        );
        warmed = true;
    } else if (Array.isArray(pipeline)) {
        const awaited = await Promise.all(
            pipeline.map((entry) =>
                typeof (entry as { initAsync?: unknown })?.initAsync === "function"
                    ? withPipelineOwner(world.gpu.device, forcer.owner, () =>
                          (entry as { initAsync(): Promise<void> }).initAsync(),
                      ).then(() => true)
                    : false,
            ),
        );
        warmed = awaited.some(Boolean);
    } else {
        throw new Error(
            `precompile "${forcer.label}" returned a value that is neither a pipeline with initAsync nor an array of raw pipelines`,
        );
    }
    const end = now();
    return { warmed, start, end };
}

/**
 * Stable User Timing measure-name prefix: `${PIPELINE_COMPILE_MEASURE_PREFIX}${forcer.label}`.
 * Cross-bundle observers filter this string without sharing World-owned GPU hooks.
 * Measures cover only forcers that awaited `initAsync`, matching `world.gpu.precompiled`;
 * synchronous pipeline creation emits no measure.
 * @internal
 */
export const PIPELINE_COMPILE_MEASURE_PREFIX = "shallot:pipeline-compile:";

/** User Timing mark and measure prefix for a plugin lifecycle hook. */
export const PLUGIN_HOOK_MEASURE_PREFIX = "shallot:plugin:";

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
        compileBody(world, forcer),
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
 * re-drains serially from its start to attribute the throw. Members reached in the serial retry
 * compile again. The serial re-drain stops at the first throw; the rest of the level and every
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
                        level.map(async (forcer) => ({
                            forcer,
                            ...(await compileBody(world, forcer)),
                        })),
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

export function adapterNotice(verdict: AdapterVerdict | undefined): string | undefined {
    if (!verdict || verdict.class === "real") return undefined;
    return `[shallot] ${verdict.reason ?? `${verdict.class} adapter: ${verdict.identity}`}`;
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
    const message = adapterNotice(verdict);
    if (message !== undefined) {
        console.warn(message);
        notice?.(verdict);
    }
    return verdict;
}

/** Acquire a world GPU context after validating each named requirement against the supplied or acquired device. */
export async function requestGPU(
    device?: GPUDevice,
    requirements: readonly NamedGpuRequirements[] = [],
    adapter?: GPUAdapter,
    owner?: {
        own(resource: { destroy(): void }): void;
        resource?: <T>(declaration: Resource<T>) => T;
        prepare?: (gpu: WorldGpu) => void;
    },
): Promise<WorldGpu> {
    // before anything resolves: typegpu binds the console method a TGSL `console.log` calls at
    // shader-generation time, so a capture installed later never sees that kernel's lines.
    captureGpuLog();
    checkTgsl();
    if (device !== undefined) checkRequirements(device.features, device.limits, requirements);
    const acquired = device === undefined ? await acquireDevice(requirements) : { device, adapter };
    const d = rawDevice(acquired.device);
    const verdict = stampAdapter(acquired.adapter);
    if (
        (globalThis as typeof globalThis & { readonly __shallotRequireHardwareGpu?: boolean })
            .__shallotRequireHardwareGpu === true &&
        verdict.class !== "real"
    ) {
        if (device === undefined) d.destroy();
        throw new Error(
            `Hardware WebGPU is required: ${verdict.reason ?? `${verdict.class} adapter: ${verdict.identity}`}. Startup stopped before plugin initialization.`,
        );
    }
    observeDevice(d);
    beginArtifactSession(d);
    let inFlight = 0;
    const fences = { issued: 0, completed: 0 };
    // Both reactions close over this World and are reused by every frame fence.
    const settle = (): void => {
        inFlight--;
    };
    // Queue fences complete in order; only successful completion makes staging reusable.
    const complete = (): void => {
        inFlight--;
        fences.completed++;
    };
    function sync(): Promise<void> {
        inFlight++;
        fences.issued++;
        const fence = submissionFence(d.queue);
        continueSubmissionFence(fence, complete, settle);
        return fence;
    }
    let pipelineGuardState: PipelineGuardState | undefined;
    const trackedDevice = owner
        ? (() => {
              const deviceOverrides = new Map<PropertyKey, unknown>();
              // Capabilities are fixed at device acquisition; native getters need not rebuild wrappers in play.
              const limits = d.limits;
              const granted = d.features;
              const queue = d.queue;
              const queueOverrides = new Map<PropertyKey, unknown>();
              const bindGroupLayouts = new WeakMap<
                  GPUBindGroupLayout,
                  PipelineStorageBindingCounts
              >();
              const pipelineLayouts = new WeakMap<
                  GPUPipelineLayout,
                  PipelineStorageBindingCounts | null
              >();
              const shaderSources = new WeakMap<GPUShaderModule, string>();
              pipelineGuardState = {
                  bindGroups: bindGroupLayouts,
                  layouts: pipelineLayouts,
              };
              const createShaderModule = d.createShaderModule.bind(d);
              const createBindGroupLayout = d.createBindGroupLayout.bind(d);
              const createPipelineLayout = d.createPipelineLayout.bind(d);
              const createComputePipeline = d.createComputePipeline.bind(d);
              const createComputePipelineAsync = d.createComputePipelineAsync.bind(d);
              const createRenderPipeline = d.createRenderPipeline.bind(d);
              const createRenderPipelineAsync = d.createRenderPipelineAsync.bind(d);
              const trackedShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
                  const module = createShaderModule(descriptor);
                  shaderSources.set(module, descriptor.code);
                  return module;
              };
              const trackedBindGroupLayout = (descriptor: GPUBindGroupLayoutDescriptor) => {
                  const layout = createBindGroupLayout(descriptor);
                  bindGroupLayouts.set(layout, countPipelineStorageBindings([descriptor.entries]));
                  return layout;
              };
              const trackedPipelineLayout = (descriptor: GPUPipelineLayoutDescriptor) => {
                  const counts: PipelineStorageBindingCounts = {
                      vertex: 0,
                      fragment: 0,
                      compute: 0,
                  };
                  let known = true;
                  for (const bindGroup of descriptor.bindGroupLayouts) {
                      if (bindGroup === null) continue;
                      const groupCounts = bindGroupLayouts.get(bindGroup);
                      if (!groupCounts) {
                          known = false;
                          break;
                      }
                      counts.vertex += groupCounts.vertex;
                      counts.fragment += groupCounts.fragment;
                      counts.compute += groupCounts.compute;
                  }
                  const layout = createPipelineLayout(descriptor);
                  pipelineLayouts.set(layout, known ? counts : null);
                  return layout;
              };
              const checkPipeline = (
                  descriptor: GPUComputePipelineDescriptor | GPURenderPipelineDescriptor,
              ) => {
                  const name = descriptor.label || "<unnamed>";
                  const plugin = pipelineGuardState?.plugin ?? "unattributed";
                  const counts =
                      descriptor.layout === "auto"
                          ? autoPipelineStorageBindings(shaderSources, descriptor)
                          : pipelineLayouts.get(descriptor.layout);
                  if (!counts) {
                      throw new UnsupportedError(
                          `Plugin "${plugin}" pipeline "${name}" cannot be checked against ` +
                              `maxStorageBuffersPerShaderStage because its layout is auto-generated or was not created through this World`,
                      );
                  }
                  checkPipelineStorageBindings(
                      plugin,
                      name,
                      counts,
                      limits.maxStorageBuffersPerShaderStage,
                  );
              };
              const trackedComputePipeline = (descriptor: GPUComputePipelineDescriptor) => {
                  checkPipeline(descriptor);
                  return createComputePipeline(descriptor);
              };
              const trackedComputePipelineAsync = (descriptor: GPUComputePipelineDescriptor) => {
                  checkPipeline(descriptor);
                  return createComputePipelineAsync(descriptor);
              };
              const trackedRenderPipeline = (descriptor: GPURenderPipelineDescriptor) => {
                  checkPipeline(descriptor);
                  return createRenderPipeline(descriptor);
              };
              const trackedRenderPipelineAsync = (descriptor: GPURenderPipelineDescriptor) => {
                  checkPipeline(descriptor);
                  return createRenderPipelineAsync(descriptor);
              };
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
                      if (key === "createShaderModule") return trackedShaderModule;
                      if (key === "createBindGroupLayout") return trackedBindGroupLayout;
                      if (key === "createPipelineLayout") return trackedPipelineLayout;
                      if (key === "createComputePipeline") return trackedComputePipeline;
                      if (key === "createComputePipelineAsync") return trackedComputePipelineAsync;
                      if (key === "createRenderPipeline") return trackedRenderPipeline;
                      if (key === "createRenderPipelineAsync") return trackedRenderPipelineAsync;
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
    if (pipelineGuardState) _pipelineGuards.set(trackedDevice, pipelineGuardState);
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
        fences,
        sync,
    };

    owner?.prepare?.(compute);
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
    if (typeof navigator !== "undefined") return "Browser";
    return "this runtime";
}

function failureMessage(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
}

const isAlignmentLimit = (name: string): boolean =>
    name.startsWith("min") && name.endsWith("Alignment");

function checkRequirements(
    features: { has(feature: GPUFeatureName): boolean },
    limits: GPUSupportedLimits,
    requirements: readonly NamedGpuRequirements[],
): void {
    for (const requirement of requirements) {
        for (const feature of requirement.features ?? []) {
            if (!features.has(feature)) {
                throw new UnsupportedError(
                    `Plugin "${requirement.plugin}" requires unsupported WebGPU feature "${feature}"`,
                    [feature],
                );
            }
        }
        for (const [name, needed] of Object.entries(requirement.limits ?? {})) {
            const available = Number(limits[name as keyof GPUSupportedLimits]);
            const required = Number(needed);
            const minAlignment = isAlignmentLimit(name);
            if (minAlignment ? available > required : available < required) {
                const condition = minAlignment ? `no greater than ${required}` : `${required}`;
                throw new UnsupportedError(
                    `Plugin "${requirement.plugin}" requires WebGPU limit ${name} ${condition}, but the device provides ${available}`,
                );
            }
        }
    }
}

function mergeLimits(requirements: readonly NamedGpuRequirements[]): Record<string, number> {
    const limits: Record<string, number> = {};
    for (const requirement of requirements) {
        for (const [name, value] of Object.entries(requirement.limits ?? {})) {
            const requested = Number(value);
            const minAlignment = isAlignmentLimit(name);
            const current = limits[name];
            limits[name] =
                current === undefined
                    ? requested
                    : minAlignment
                      ? Math.min(current, requested)
                      : Math.max(current, requested);
        }
    }
    return limits;
}

async function acquireDevice(
    requirements: readonly NamedGpuRequirements[],
): Promise<{ device: GPUDevice; adapter: GPUAdapter }> {
    const gpu = typeof navigator === "undefined" ? undefined : navigator.gpu;
    const runtime = gpuRuntimeName();
    if (!gpu) {
        const fix =
            runtime === "Bun"
                ? " Install the optional webgpu peer dependency to enable GPU builds."
                : "";
        throw new UnsupportedError(
            `WebGPU unavailable: navigator.gpu is missing in ${runtime}.${fix}`,
        );
    }

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
    if (!adapter) {
        const remedy =
            runtime === "Bun" || runtime === "Node.js"
                ? "Use a supported Dawn backend; on Linux without a GPU, install a software Vulkan driver such as mesa-vulkan-drivers."
                : "In a browser, enable hardware acceleration or use a browser with WebGPU support.";
        throw new UnsupportedError(`No WebGPU adapter is available in ${runtime}. ${remedy}`);
    }

    const required = [...new Set(requirements.flatMap((item) => item.features ?? []))];
    const preferred = [...new Set(requirements.flatMap((item) => item.preferredFeatures ?? []))];
    checkRequirements(adapter.features, adapter.limits, requirements);
    const { granted } = resolveFeatures(adapter.features, required, preferred);
    const requiredLimits = mergeLimits(requirements);
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
