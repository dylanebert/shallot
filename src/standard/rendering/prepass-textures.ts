import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as s from "typegpu/std";
import { Camera, MainPassSystem, PrepassSystem, type View, Views } from "../../core/rendering";
import { precompile, type System, type World } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups2,
    endComputePass,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";

const layout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepthMultisampled2d() },
    normal: { texture: d.textureMultisampled2d(), sampleType: "unfilterable-float" },
    outputDepth: { storageTexture: d.textureStorage2d("r32float", "write-only") },
    outputNormal: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
});
const resolve = tgpu.computeFn({ workgroupSize: [8, 8], in: { id: d.builtin.globalInvocationId } })(
    (input) => {
        "use gpu";
        if (s.any(s.ge(input.id.xy, s.textureDimensions(layout.$.outputDepth)))) return;
        const px = d.vec2i(input.id.xy);
        s.textureStore(
            layout.$.outputDepth,
            px,
            d.vec4f(s.textureLoad(layout.$.depth, px, 0), 0, 0, 0),
        );
        s.textureStore(layout.$.outputNormal, px, s.textureLoad(layout.$.normal, px, 0));
    },
);
interface Resolved {
    depth: GPUTextureView;
    normal: GPUTextureView | null;
    width: number;
    height: number;
    textures: GPUTexture[];
    outputDepth: GPUTextureView;
    outputNormal: GPUTextureView;
    group: GPUBindGroup;
}
export const prepassTextures = {
    create: () => ({
        cameras: new Map<number, Resolved>(),
        depth: [] as (GPUTextureView | null)[],
        normal: [] as (GPUTextureView | null)[],
        stubDepth: null as GPUTextureView | null,
        stubNormal: null as GPUTextureView | null,
        msaaNormal: null as GPUTextureView | null,
        pipeline: null as TgpuComputePipeline | null,
        requested: null as ((world: World, eid: number) => boolean) | null,
        pass: { label: "standard:prepass-resolve" } as GPUComputePassDescriptor,
    }),
};
export function warmPrepassTextures(world: World) {
    const state = world.resource(prepassTextures);
    const { device, root } = world.gpu;
    function stub(format: GPUTextureFormat, sampleCount = 1) {
        const texture = world.gpu.device.createTexture({
            label: "standard:prepass-stub",
            size: [1, 1],
            format,
            sampleCount,
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                (sampleCount === 1 ? GPUTextureUsage.COPY_DST : GPUTextureUsage.RENDER_ATTACHMENT),
        });
        world.own(texture);
        if (sampleCount === 1)
            device.queue.writeTexture(
                { texture },
                new Uint8Array(format === "r32float" ? 4 : 8),
                {},
                [1, 1],
            );
        return texture.createView();
    }
    state.stubDepth = stub("r32float");
    state.stubNormal = stub("rgba16float");
    state.msaaNormal = stub("rgba16float", 4);
    state.pipeline = root.createComputePipeline({ compute: resolve });
    precompile(world, "standard:prepass-resolve", () => state.pipeline!);
}
function retire(world: World, camera: Resolved) {
    for (const texture of camera.textures) world.retireGpuResource(texture);
}
function pruneCamera(this: World, camera: Resolved, eid: number) {
    const state = this.resource(prepassTextures);
    const view = this.resource(Views).get(eid);
    if (
        !view?.depth ||
        !state.requested!(this, eid) ||
        this.storage(Camera).antialias.get(eid) === 0
    ) {
        retire(this, camera);
        state.cameras.delete(eid);
    }
}
function resolveView(this: World, view: View, eid: number) {
    const state = this.resource(prepassTextures);
    if (!view.framebuffer || !view.depth) return;
    if (this.storage(Camera).antialias.get(eid) === 0) {
        state.depth[view.slot] = view.depth;
        state.normal[view.slot] = view.normal;
        return;
    }
    if (!state.requested!(this, eid)) return;
    let camera = state.cameras.get(eid);
    if (
        !camera ||
        camera.depth !== view.depth ||
        camera.normal !== view.normal ||
        camera.width !== view.width ||
        camera.height !== view.height
    ) {
        if (camera) retire(this, camera);
        const textures: GPUTexture[] = [];
        for (const format of ["r32float", "rgba16float"] as const) {
            textures.push(
                this.gpu.device.createTexture({
                    label: "standard:prepass-resolved",
                    size: [view.width, view.height],
                    format,
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
                }),
            );
        }
        const outputDepth = textures[0]!.createView();
        const outputNormal = textures[1]!.createView();
        camera = {
            depth: view.depth,
            normal: view.normal,
            width: view.width,
            height: view.height,
            textures,
            outputDepth,
            outputNormal,
            group: this.gpu.root.unwrap(
                this.gpu.root.createBindGroup(layout, {
                    depth: view.depth,
                    normal: view.normal ?? state.msaaNormal!,
                    outputDepth,
                    outputNormal,
                }),
            ),
        };
        state.cameras.set(eid, camera);
    }
    state.pass.timestampWrites = this.gpu.span?.(state.pass.label!);
    const pass = beginComputePass(this.frameEncoder()!, state.pass);
    setComputePipeline(pass, this.gpu.root.unwrap(state.pipeline!));
    setComputeBindGroup(pass, 0, camera.group);
    dispatchWorkgroups2(pass, Math.ceil(view.width / 8), Math.ceil(view.height / 8));
    endComputePass(pass);
    state.depth[view.slot] = camera.outputDepth;
    state.normal[view.slot] = view.normal ? camera.outputNormal : null;
}
export const ResolvePrepassTexturesSystem: System = {
    name: "ResolvePrepassTextures",
    group: "draw",
    after: [PrepassSystem],
    before: [MainPassSystem],
    update(world) {
        const state = world.resource(prepassTextures);
        state.depth.fill(null);
        state.normal.fill(null);
        state.cameras.forEach(pruneCamera, world);
        world.resource(Views).forEach(resolveView, world);
    },
};
export function disposePrepassTextures(world: World) {
    const state = world.resource(prepassTextures);
    for (const camera of state.cameras.values())
        for (const texture of camera.textures) texture.destroy();
    state.cameras.clear();
}
