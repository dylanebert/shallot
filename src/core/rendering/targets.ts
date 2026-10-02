import type { World } from "../../engine";
import { unpackColor } from "../../engine";
import { Camera } from "./camera";
import { RenderContext } from "./render";
import type { View } from "./view";

/** Reverse-Z depth format shared by view passes and matching renderer pipelines. */
export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";
/** Main-pass sample count when Camera.antialias is enabled; prepasses stay single-sample. */
export const SAMPLE_COUNT = 4;
export const PICKING_ID_FORMAT: GPUTextureFormat = "r32uint";
/** Reserved picking value for pixels with no owner. */
export const PICKING_ID_NONE = 0xffffffff;

/** Opt a camera into stored single-sample prepass depth, published as view.depth. */
export const DepthPrepass = {};
/** Opt a camera into the single-sample picking lane, published as view.pickingId. */
export const PickingPrepass = {};

export interface ColorLane {
    name: string;
    marker: object;
    format: GPUTextureFormat;
    usage: number;
    clear: GPUColor;
    set(view: View, texture: GPUTexture): void;
}

export const COLOR_LANES: ColorLane[] = [
    {
        name: "tag",
        marker: PickingPrepass,
        format: PICKING_ID_FORMAT,
        get usage() {
            return (
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC
            );
        },
        clear: { r: PICKING_ID_NONE, g: 0, b: 0, a: 0 },
        set(view, texture) {
            view.pickingId = texture;
        },
    },
];

/** Stable key for the requested color attachment set; depth-only is the empty key. */
export function laneKey(lanes: ColorLane[]): string {
    return lanes.map((lane) => lane.name).join("-");
}

interface Target {
    texture: GPUTexture;
    view: GPUTextureView;
    w: number;
    h: number;
}
interface ColorTargets {
    color: GPUTexture | null;
    colorView: GPUTextureView | null;
    depth: GPUTexture;
    depthView: GPUTextureView;
    w: number;
    h: number;
    aa: boolean;
    label: string;
}
type ColorAttachment = Omit<GPURenderPassColorAttachment, "view" | "resolveTarget"> & {
    view: GPUTextureView;
    resolveTarget?: GPUTextureView;
};

function createTargets() {
    const clearValue = { r: 0, g: 0, b: 0, a: 1 };
    const msaaColor: ColorAttachment = {
        view: null!,
        resolveTarget: null!,
        loadOp: "clear",
        storeOp: "discard",
        clearValue,
    };
    const directColor: ColorAttachment = {
        view: null!,
        loadOp: "clear",
        storeOp: "store",
        clearValue,
    };
    const colorAttachments = [msaaColor];
    const colorDepth: GPURenderPassDepthStencilAttachment = {
        view: null!,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
        depthClearValue: 0,
    };
    const colorPass: GPURenderPassDescriptor = {
        label: "",
        colorAttachments,
        depthStencilAttachment: colorDepth,
    };
    return {
        depth: new Map<number, Target>(),
        laneTargets: new Map<string, Target>(),
        colorTargets: new Map<number, ColorTargets>(),
        clearValue,
        clearPacked: -1,
        msaaColor,
        directColor,
        colorAttachments,
        colorDepth,
        colorPass,
    };
}
export const viewTargetsKey = { create: createTargets };

/** Initialize this world's target caches. */
export function initializeViewTargets(world: World): void {
    world.resource(viewTargetsKey);
}

/** Release cached view targets at plugin disposal, not camera detachment. */
export function disposeViewTargets(world: World): void {
    const state = world.resource(viewTargetsKey);
    for (const c of state.depth.values()) c.texture.destroy();
    for (const c of state.laneTargets.values()) c.texture.destroy();
    for (const c of state.colorTargets.values()) {
        c.color?.destroy();
        c.depth.destroy();
    }
    state.depth.clear();
    state.laneTargets.clear();
    state.colorTargets.clear();
}

/** Single-sample depth for any requested prepass lane; recreated on resize. */
function depthView(world: World, eid: number, w: number, h: number): GPUTextureView {
    const state = world.resource(viewTargetsKey);
    const cached = state.depth.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `standard-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    state.depth.set(eid, { texture, view, w, h });
    return view;
}

/** Reset published outputs and select only lanes whose camera markers are present. */
export function prepassLanes(
    world: World,
    eid: number,
    view: View,
): { lanes: ColorLane[]; storeDepth: boolean } | null {
    view.pickingId = null;
    view.depth = null;
    const storeDepth = world.has(eid, DepthPrepass);
    let marked = false;
    for (let i = 0; i < COLOR_LANES.length; i++) {
        if (world.has(eid, COLOR_LANES[i].marker)) {
            marked = true;
            break;
        }
    }
    if (!marked && !storeDepth) return null;
    const lanes: ColorLane[] = [];
    for (let i = 0; i < COLOR_LANES.length; i++) {
        if (world.has(eid, COLOR_LANES[i].marker)) lanes.push(COLOR_LANES[i]);
    }
    return { lanes, storeDepth };
}

/** Prepass attachments publish the requested outputs; the caller opens and records the pass. */
export function prepassDescriptor(
    world: World,
    eid: number,
    view: View,
    lanes: ColorLane[],
    storeDepth: boolean,
): GPURenderPassDescriptor {
    const depth = depthView(world, eid, view.width, view.height);
    view.depth = storeDepth ? depth : null;
    return {
        label: `standard-prepass/${eid}`,
        timestampWrites: world.gpu.span?.("standard:prepass"),
        colorAttachments: lanes.map((lane) => {
            const target = laneTarget(world, eid, lane, view.width, view.height);
            lane.set(view, target.texture);
            return { view: target.view, loadOp: "clear", storeOp: "store", clearValue: lane.clear };
        }),
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            depthStoreOp: storeDepth ? "store" : "discard",
            depthClearValue: 0,
        },
    };
}

/** A camera's requested color lane; recreated on resize. */
function laneTarget(world: World, eid: number, lane: ColorLane, w: number, h: number): Target {
    const state = world.resource(viewTargetsKey);
    const key = `${eid}:${lane.name}`;
    const cached = state.laneTargets.get(key);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `standard-${lane.name}-${eid}`,
        size: { width: w, height: h },
        format: lane.format,
        usage: lane.usage,
    });
    const entry = { texture, view: texture.createView(), w, h };
    state.laneTargets.set(key, entry);
    return entry;
}

/** Main-pass MSAA color and depth, recreated on resize or AA toggle. */
export function colorTargets(
    world: World,
    eid: number,
    w: number,
    h: number,
    aa: boolean,
): ColorTargets {
    const state = world.resource(viewTargetsKey);
    const cached = state.colorTargets.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.aa === aa) return cached;
    cached?.color?.destroy();
    cached?.depth.destroy();
    const color = aa
        ? world.gpu.device.createTexture({
              label: `standard-color-msaa-${eid}`,
              size: { width: w, height: h },
              format: world.resource(RenderContext).format,
              sampleCount: SAMPLE_COUNT,
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null;
    const depth = world.gpu.device.createTexture({
        label: `standard-color-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        sampleCount: aa ? SAMPLE_COUNT : 1,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const entry = {
        color,
        colorView: color?.createView() ?? null,
        depth,
        depthView: depth.createView(),
        w,
        h,
        aa,
        label: `standard-color/${eid}`,
    };
    state.colorTargets.set(eid, entry);
    return entry;
}

/** Borrow the main-pass descriptor until the next call; core owns clear and resolve. */
export function colorPassDescriptor(
    world: World,
    eid: number,
    targets: ColorTargets,
    framebuffer: GPUTextureView,
): GPURenderPassDescriptor {
    const state = world.resource(viewTargetsKey);
    const packed = world.storage(Camera).clearColor.get(eid);
    if (packed !== state.clearPacked) {
        const clear = unpackColor(packed);
        state.clearValue.r = clear.r;
        state.clearValue.g = clear.g;
        state.clearValue.b = clear.b;
        state.clearPacked = packed;
    }
    if (targets.colorView) {
        state.msaaColor.view = targets.colorView;
        state.msaaColor.resolveTarget = framebuffer;
        state.colorAttachments[0] = state.msaaColor;
    } else {
        state.directColor.view = framebuffer;
        state.colorAttachments[0] = state.directColor;
    }
    state.colorDepth.view = targets.depthView;
    state.colorPass.label = targets.label;
    state.colorPass.timestampWrites = world.gpu.span?.("standard:color");
    return state.colorPass;
}
