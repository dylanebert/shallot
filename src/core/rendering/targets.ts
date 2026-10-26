import type { World } from "../../engine";
import { component, unpackColor } from "../../engine";
import { Camera } from "./camera";
import { RenderContext } from "./render";
import type { View } from "./view";

/** Reverse-Z depth format shared by view passes and matching renderer pipelines. */
export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";
/** Main-pass sample count when Camera.antialias is enabled; the shared depth lane matches it. */
export const SAMPLE_COUNT = 4;

/** Explicitly request the camera's stored depth lane, published as `view.depth`. */
export const DepthPrepass = component("DepthPrepass", {});

/** Request world-space normals, encoded as normal * 0.5 + 0.5 in `view.normal`. */
export const NormalPrepass = component("NormalPrepass", {});
export const NORMAL_FORMAT: GPUTextureFormat = "rgb10a2unorm";
/** Plugins request normals per camera; a normal request also requests stored depth.
 * Preparation checks initial cameras before they necessarily have a bound view. */
export const NormalPrepassRequests = {
    create: () => [] as ((world: World, eid: number, view?: View) => boolean)[],
};

/** A plugin's per-camera request for the shared stored depth lane. */
export type DepthPrepassRequest = (world: World, eid: number, view: View) => boolean;

/** Plugins register per-camera consumers here; Core records one shared lane when any request returns true. */
export const DepthPrepassRequests = { create: () => [] as DepthPrepassRequest[] };
interface Target {
    texture: GPUTexture;
    view: GPUTextureView;
    w: number;
    h: number;
    sampleCount: number;
}
/** Stored depth with the camera's prepass descriptor, built once per depth texture. */
interface DepthTarget extends Target {
    pass: GPURenderPassDescriptor;
    colorAttachments: GPURenderPassColorAttachment[];
    normalAttachment: ColorAttachment;
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
        depth: new Map<number, DepthTarget>(),
        normal: new Map<number, Target>(),
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
// Outside the u32 clearColor range: the cached clear is transparent black.
const TRANSPARENT = -2;

export function initializeViewTargets(world: World): void {
    world.resource(viewTargetsKey);
}

/** Retire one camera's prepass depth and main-pass targets; detachView calls it. Safe on unbound eids. */
export function releaseViewTargets(world: World, eid: number): void {
    const state = world.resource(viewTargetsKey);
    const depth = state.depth.get(eid);
    const color = state.colorTargets.get(eid);
    const normal = state.normal.get(eid);
    if (normal) world.retireGpuResource(normal.texture);
    state.normal.delete(eid);
    if (depth) world.retireGpuResource(depth.texture);
    if (color?.color) world.retireGpuResource(color.color);
    if (color) world.retireGpuResource(color.depth);
    state.depth.delete(eid);
    state.colorTargets.delete(eid);
}

/** Release every camera's cached view targets at plugin disposal. */
export function disposeViewTargets(world: World): void {
    const state = world.resource(viewTargetsKey);
    for (const c of state.depth.values()) c.texture.destroy();
    for (const c of state.colorTargets.values()) {
        c.color?.destroy();
        c.depth.destroy();
    }
    for (const c of state.normal.values()) c.texture.destroy();
    state.normal.clear();
    state.depth.clear();
    state.colorTargets.clear();
}

/** Stored depth matches the main pass's sample count; recreated on resize or AA toggle. */
function depthTarget(world: World, eid: number, w: number, h: number): DepthTarget {
    const state = world.resource(viewTargetsKey);
    const sampleCount = world.storage(Camera).antialias.get(eid) !== 0 ? SAMPLE_COUNT : 1;
    const cached = state.depth.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.sampleCount === sampleCount)
        return cached;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `standard-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    const colorAttachments: GPURenderPassColorAttachment[] = [];
    const target: DepthTarget = {
        texture,
        view,
        w,
        h,
        sampleCount,
        pass: {
            label: `standard-prepass/${eid}`,
            colorAttachments,
            depthStencilAttachment: {
                view,
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 0,
            },
        },
        colorAttachments,
        normalAttachment: {
            view: null!,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
    };
    state.depth.set(eid, target);
    return target;
}

export function releaseNormalTarget(world: World, eid: number): void {
    const targets = world.resource(viewTargetsKey).normal;
    const normal = targets.get(eid);
    if (normal) world.retireGpuResource(normal.texture);
    targets.delete(eid);
}

/** Borrow the camera's prepass descriptor until its next call; its attachments publish the requested
 * outputs, and the caller opens and records the pass. */
export function prepassDescriptor(
    world: World,
    eid: number,
    view: View,
    normals = false,
): GPURenderPassDescriptor {
    const depth = depthTarget(world, eid, view.width, view.height);
    view.depth = depth.view;
    const state = world.resource(viewTargetsKey);
    let normal = state.normal.get(eid);
    const sampleCount = depth.sampleCount;
    if (
        normal &&
        (!normals ||
            normal.w !== view.width ||
            normal.h !== view.height ||
            normal.sampleCount !== sampleCount)
    ) {
        world.retireGpuResource(normal.texture);
        state.normal.delete(eid);
        normal = undefined;
    }
    if (normals && !normal) {
        const texture = world.gpu.device.createTexture({
            label: `standard-normal-${eid}`,
            size: { width: view.width, height: view.height },
            format: NORMAL_FORMAT,
            sampleCount,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
        normal = {
            texture,
            view: texture.createView(),
            w: view.width,
            h: view.height,
            sampleCount,
        };
        state.normal.set(eid, normal);
    }
    view.normal = normal?.view ?? null;
    if (normal) {
        depth.normalAttachment.view = normal.view;
        depth.colorAttachments[0] = depth.normalAttachment;
    } else depth.colorAttachments.length = 0;
    depth.pass.timestampWrites = world.gpu.span?.("standard:prepass");
    return depth.pass;
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

/** Borrow the main-pass descriptor until the next call; core owns clear and resolve. The bottom camera
 * of a target clears to its clearColor; a camera above it clears to transparent black. */
export function colorPassDescriptor(
    world: World,
    eid: number,
    targets: ColorTargets,
    framebuffer: GPUTextureView,
    bottom: boolean,
): GPURenderPassDescriptor {
    const state = world.resource(viewTargetsKey);
    const packed = bottom ? world.storage(Camera).clearColor.get(eid) : TRANSPARENT;
    if (packed !== state.clearPacked) {
        if (bottom) {
            const clear = unpackColor(packed);
            state.clearValue.r = clear.r;
            state.clearValue.g = clear.g;
            state.clearValue.b = clear.b;
            state.clearValue.a = 1;
        } else {
            state.clearValue.r = 0;
            state.clearValue.g = 0;
            state.clearValue.b = 0;
            state.clearValue.a = 0;
        }
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
