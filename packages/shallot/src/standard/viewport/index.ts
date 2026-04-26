import { resource, type Plugin, type State, type System } from "../../engine";
import { createWebView, type View } from "./view";
import {
    Compute,
    ComputePlugin,
    type ComputeResources,
    type ExecutionContext,
    type ComputeNode,
    type ComputeGraph,
} from "../compute";
import {
    type ProfileState,
    GpuProfile,
    createProfileState,
    allocSlot,
    drainProfile,
    resetProfile,
    resolveProfile,
    readProfile,
} from "../compute/core";
export type { View } from "./view";

export const ActiveCamera = resource<{ eid: number }>("active-camera");
export const Views = resource<Map<number, View>>("views");

export const Canvas = {
    selector: {} as Record<number, string>,
};

export const ViewHooks =
    resource<((state: State, canvasEid: number, view: View) => void)[]>("view-hooks");

function executeNodes(
    nodes: readonly ComputeNode[],
    device: GPUDevice,
    ctx: ExecutionContext,
    profile: ProfileState | null,
    afterSubmitQueue: (() => void)[],
    reportCpu?: (name: string, ms: number) => void,
): void {
    let encoder = device.createCommandEncoder();
    (ctx as { encoder: GPUCommandEncoder }).encoder = encoder;

    const drainAfterSubmit = () => {
        for (let i = 0; i < afterSubmitQueue.length; i++) afterSubmitQueue[i]();
        afterSubmitQueue.length = 0;
    };

    for (const node of nodes) {
        if (reportCpu) {
            const t = performance.now();
            node.execute(ctx);
            reportCpu(`  ${node.name}`, performance.now() - t);
        } else {
            node.execute(ctx);
        }

        if (node.sync) {
            device.queue.submit([encoder.finish()]);
            drainAfterSubmit();
            encoder = device.createCommandEncoder();
            (ctx as { encoder: GPUCommandEncoder }).encoder = encoder;
        }
    }

    if (profile && profile.nextSlot > 0) {
        resolveProfile(encoder, profile);
    }

    if (reportCpu) {
        const t = performance.now();
        device.queue.submit([encoder.finish()]);
        reportCpu("Viewport/1:submit", performance.now() - t);
    } else {
        device.queue.submit([encoder.finish()]);
    }
    drainAfterSubmit();
}

interface CachedContext {
    ctx: ExecutionContext;
    afterSubmitQueue: (() => void)[];
}

function createCachedContext(
    device: GPUDevice,
    resources: ComputeResources,
    profile: ProfileState | null,
): CachedContext {
    const afterSubmitQueue: (() => void)[] = [];
    const tsWriter = profile ? (name: string) => allocSlot(profile, name) : undefined;
    const ctx: ExecutionContext = {
        device,
        queue: device.queue,
        encoder: null as unknown as GPUCommandEncoder,
        context: null as unknown as GPUCanvasContext,
        format: navigator.gpu.getPreferredCanvasFormat(),
        canvasView: null as unknown as GPUTextureView,
        timestampWrites: tsWriter,
        getTexture(name: string) {
            return resources.textures.get(name) ?? null;
        },
        getTextureView(name: string) {
            return resources.textureViews.get(name) ?? null;
        },
        getBuffer(name: string) {
            return resources.buffers.get(name) ?? null;
        },
        setTexture(name: string, texture: GPUTexture) {
            resources.textures.set(name, texture);
        },
        setTextureView(name: string, view: GPUTextureView) {
            resources.textureViews.set(name, view);
        },
        setBuffer(name: string, buffer: GPUBuffer) {
            resources.buffers.set(name, buffer);
        },
        afterSubmit(fn: () => void) {
            afterSubmitQueue.push(fn);
        },
        subGraph: "",
    };
    return { ctx, afterSubmitQueue };
}

function updateContext(
    cached: CachedContext,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvasView: GPUTextureView,
): void {
    const c = cached.ctx as {
        context: GPUCanvasContext;
        format: GPUTextureFormat;
        canvasView: GPUTextureView;
    };
    c.context = context;
    c.format = format;
    c.canvasView = canvasView;
}

function resolveSubGraph(graph: ComputeGraph, cameraEid: number): string {
    let defaultSg: string | null = null;
    for (const [name, sg] of graph.subGraphs) {
        if (sg.check?.(cameraEid)) return name;
        if (!sg.check && defaultSg === null) defaultSg = name;
    }
    return defaultSg ?? "";
}

let _profile: ProfileState | null = null;

let _frameCached: CachedContext | null = null;
let _frameResources: ComputeResources | null = null;

function executeFrameGraph(
    compute: Compute,
    subGraph: string,
    reportCpu?: (name: string, ms: number) => void,
): void {
    const { device, graph, resources } = compute;
    const plan = graph.compile(subGraph);

    if (plan.frame.length === 0) return;

    if (!_frameCached || _frameResources !== resources) {
        _frameCached = createCachedContext(device, resources, _profile);
        _frameResources = resources;
    }
    updateContext(
        _frameCached,
        null as unknown as GPUCanvasContext,
        "" as GPUTextureFormat,
        null as unknown as GPUTextureView,
    );

    executeNodes(
        plan.frame,
        device,
        _frameCached.ctx,
        _profile,
        _frameCached.afterSubmitQueue,
        reportCpu,
    );
}

let _viewCached: CachedContext | null = null;
let _viewResources: ComputeResources | null = null;

function executeViewGraph(
    compute: Compute,
    view: View,
    subGraph: string,
    reportCpu?: (name: string, ms: number) => void,
): void {
    const { device, graph, resources } = compute;
    const { context, format } = view;
    const plan = graph.compile(subGraph);

    if (plan.view.length === 0) return;

    view.textures.forEach((v, k) => {
        resources.textures.set(k, v);
    });
    view.textureViews.forEach((v, k) => {
        resources.textureViews.set(k, v);
    });

    const canvasTexture = context.getCurrentTexture();
    if (!canvasTexture) return;
    const canvasView = canvasTexture.createView();

    if (!_viewCached || _viewResources !== resources) {
        _viewCached = createCachedContext(device, resources, _profile);
        _viewResources = resources;
    }
    updateContext(_viewCached, context, format, canvasView);
    (_viewCached.ctx as { subGraph: string }).subGraph = subGraph;

    executeNodes(
        plan.view,
        device,
        _viewCached.ctx,
        _profile,
        _viewCached.afterSubmitQueue,
        reportCpu,
    );

    if (_profile) readProfile(_profile);
}

const _canvasEntities: number[] = [];

function resolveViews(state: State): void {
    const compute = Compute.from(state);
    const views = Views.from(state);
    if (!compute || !views) return;

    const { device } = compute;

    _canvasEntities.length = 0;
    for (const eid of state.query([Canvas])) {
        _canvasEntities.push(eid);
    }
    const canvasEntities = _canvasEntities;

    if (canvasEntities.length === 0) {
        const element = document.querySelector("canvas");
        if (element) {
            const eid = state.addEntity();
            state.addComponent(eid, Canvas);
            canvasEntities.push(eid);
        }
    }

    for (const eid of canvasEntities) {
        if (views.has(eid)) continue;

        const selector = Canvas.selector[eid];
        const element = selector
            ? (document.querySelector(selector) as HTMLCanvasElement | null)
            : document.querySelector("canvas");

        if (!element) {
            if (selector) console.warn(`Canvas selector "${selector}" matched no element`);
            continue;
        }

        views.set(eid, createWebView(element, device));
    }
}

const ViewSyncSystem: System = {
    group: "setup",
    annotations: { mode: "always" },

    setup(state: State) {
        resolveViews(state);
    },

    update(state: State) {
        const views = Views.from(state);
        if (!views) return;

        for (const view of views.values()) {
            if (view.dirty) {
                if (view.element) {
                    view.width = view.element.width;
                    view.height = view.element.height;
                }
                view.dirty = false;
            }
        }
    },

    dispose(state: State) {
        const views = Views.from(state);
        if (!views) return;
        for (const view of views.values()) {
            view.observer?.disconnect();
        }
    },
};

const ViewportSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,

    update(state: State) {
        const compute = Compute.from(state);
        const views = Views.from(state);
        if (!compute || !views || views.size === 0) return;

        let allZero = true;
        for (const view of views.values()) {
            if (view.width > 0 && view.height > 0) {
                allZero = false;
                break;
            }
        }
        if (allZero) return;

        if (_profile) {
            drainProfile(_profile);
            resetProfile(_profile);
        }

        const report = state.scheduler.reportCpu.bind(state.scheduler);
        const hooks = ViewHooks.from(state);

        let frameGraphExecuted = false;
        let t0: number;

        for (const [canvasEid, view] of views) {
            if (hooks) {
                t0 = performance.now();
                for (const hook of hooks) hook(state, canvasEid, view);
                report("Viewport/1:hooks", performance.now() - t0);
            }

            const cameraEid = ActiveCamera.from(state)?.eid ?? -1;
            const sg = resolveSubGraph(compute.graph, cameraEid);

            if (!frameGraphExecuted) {
                frameGraphExecuted = true;
                t0 = performance.now();
                executeFrameGraph(compute, sg, report);
                report("Viewport/1:frame", performance.now() - t0);
            }

            t0 = performance.now();
            executeViewGraph(compute, view, sg, report);
            report("Viewport/1:view", performance.now() - t0);
        }

        compute.frameIndex++;
    },
};

export const ViewportPlugin: Plugin = {
    name: "Viewport",
    systems: [ViewSyncSystem, ViewportSystem],
    components: { Canvas },
    dependencies: [ComputePlugin],

    async initialize(state) {
        Canvas.selector = {} as Record<number, string>;
        state.setResource(Views, new Map());
        state.setResource(ViewHooks, []);

        const compute = Compute.from(state);
        if (compute) {
            _profile = compute.device.features.has("timestamp-query")
                ? createProfileState(compute.device, 64)
                : null;
            const profiles: Map<string, number>[] = [];
            if (_profile) profiles.push(_profile.durations);
            state.setResource(GpuProfile, profiles);
        }
    },
};
