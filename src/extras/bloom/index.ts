import {
    Camera,
    CorePipelinePlugin,
    type EffectPass,
    EffectPasses,
    fullscreenVertex,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import { component, f32, type Plugin, u32, vec2, type World } from "../../engine";
import {
    beginRenderPass,
    draw,
    endRenderPass,
    precompile,
    setRenderBindGroup,
    setRenderPipeline,
} from "../../engine/runtime";
import { composite, down, first, layout, Settings, up } from "./shaders";
import { BloomCompositeMode, blendFactor, type CameraState, type State, stateKey } from "./state";

export { BloomCompositeMode };

/** HDR light scattering before tonemapping. Intensity is the scattered fraction in
 * EnergyConserving mode, or added light in Additive mode. Threshold filters HDR brightness;
 * thresholdSoftness is clamped to [0, 1]. Nonzero thresholds suit Additive mode.
 * maxMipDimension is the chain's height in pixels; scale stretches the sampling kernel per axis.
 * A stacked camera scatters only within its original coverage and keeps its alpha.
 */
export const Bloom = component(
    "Bloom",
    {
        intensity: f32,
        lowFrequencyBoost: f32,
        lowFrequencyBoostCurvature: f32,
        highPassFrequency: f32,
        threshold: f32,
        thresholdSoftness: f32,
        compositeMode: u32,
        maxMipDimension: u32,
        scale: vec2,
    },
    {
        defaults: () => ({
            intensity: 0.15,
            lowFrequencyBoost: 0.7,
            lowFrequencyBoostCurvature: 0.95,
            highPassFrequency: 1,
            threshold: 0,
            thresholdSoftness: 0,
            compositeMode: BloomCompositeMode.EnergyConserving,
            maxMipDimension: 512,
            scale: [1, 1],
        }),
    },
);

const CAMERAS = [Camera, Bloom];
function group(
    world: World,
    state: State,
    camera: CameraState,
    input: GPUTextureView,
    original = input,
): GPUBindGroup {
    return world.gpu.device.createBindGroup({
        layout: state.layout!,
        entries: [
            { binding: 0, resource: input },
            { binding: 1, resource: state.sampler! },
            { binding: 2, resource: { buffer: camera.buffer } },
            { binding: 3, resource: original },
        ],
    });
}
function effect(state: State, camera: CameraState): EffectPass {
    return (world, _eid, _view, input, output) => {
        const chain = camera.chain!;
        let groups = camera.groups.get(input);
        if (!groups) {
            groups = {
                first: group(world, state, camera, input),
                final: group(world, state, camera, chain.views[0], input),
            };
            camera.groups.set(input, groups);
        }
        const attachment = (state.pass.colorAttachments as GPURenderPassColorAttachment[])[0];
        const encoder = world.frameEncoder()!;
        for (let mip = 0; mip < chain.views.length; mip++) {
            attachment.view = chain.views[mip];
            attachment.loadOp = "clear";
            const pass = beginRenderPass(encoder, state.pass);
            setRenderPipeline(pass, mip === 0 ? state.first! : state.down!);
            setRenderBindGroup(pass, 0, mip === 0 ? groups.first : chain.groups[mip - 1]);
            draw(pass, 3);
            endRenderPass(pass);
        }
        for (let mip = chain.views.length - 1; mip > 0; mip--) {
            attachment.view = chain.views[mip - 1];
            attachment.loadOp = "load";
            const pass = beginRenderPass(encoder, state.pass);
            setRenderPipeline(pass, state.up[camera.mode]);
            setRenderBindGroup(pass, 0, chain.groups[mip]);
            pass.setBlendConstant(chain.blends[mip]);
            draw(pass, 3);
            endRenderPass(pass);
        }
        attachment.view = output;
        attachment.loadOp = "clear";
        const pass = beginRenderPass(encoder, state.pass);
        setRenderPipeline(pass, state.composite!);
        setRenderBindGroup(pass, 0, groups.final);
        draw(pass, 3);
        endRenderPass(pass);
    };
}
function unregister(world: World, eid: number, camera: CameraState) {
    const before = world.resource(EffectPasses).get(eid)?.before;
    const index = before?.indexOf(camera.effect) ?? -1;
    if (index >= 0) before!.splice(index, 1);
    camera.chain?.texture.destroy();
    camera.buffer.destroy();
}
/** A pre-tonemapping mip pyramid, ordered before the effect stack regardless of plugin order. */
export const BloomPlugin: Plugin = {
    name: "Bloom",
    gpu: {},
    dependencies: [CorePipelinePlugin],
    components: [Bloom],
    systems: [
        {
            name: "bloom",
            group: "draw",
            before: [TonemappingSystem],
            update(world) {
                const state = world.resource(stateKey);
                const effects = world.resource(EffectPasses);
                const views = world.resource(Views);
                const s = world.storage(Bloom);
                for (const eid of world.query(CAMERAS)) {
                    const view = views.get(eid);
                    if (!view || s.intensity.get(eid) === 0) continue;
                    let camera = state.cameras.get(eid);
                    if (!camera) {
                        const bytes = new Float32Array(12);
                        camera = {
                            buffer: world.gpu.root.unwrap(
                                world.gpu.root.createBuffer(Settings).$usage("uniform"),
                            ),
                            bytes,
                            words: new Uint32Array(bytes.buffer),
                            chain: null,
                            groups: new WeakMap(),
                            effect: null!,
                            mode: 0,
                            settings: {
                                intensity: 0,
                                lowFrequencyBoost: 0,
                                lowFrequencyBoostCurvature: 0,
                                highPassFrequency: 0,
                                compositeMode: BloomCompositeMode.EnergyConserving,
                            },
                        };
                        camera.effect = effect(state, camera);
                        state.cameras.set(eid, camera);
                    }
                    const dimension = s.maxMipDimension.get(eid);
                    const count = Math.max(Math.floor(Math.log2(dimension)), 2) - 1;
                    const width = Math.max(Math.round((view.width * dimension) / view.height), 1);
                    const height = Math.max(dimension, 1);
                    let chain = camera.chain;
                    if (
                        !chain ||
                        chain.width !== width ||
                        chain.height !== height ||
                        chain.views.length !== count
                    ) {
                        chain?.texture.destroy();
                        const texture = world.gpu.device.createTexture({
                            label: "bloom-chain",
                            size: [width, height],
                            mipLevelCount: count,
                            format: "rgba16float",
                            usage:
                                GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                        });
                        const mipViews = Array.from({ length: count }, (_, mip) =>
                            texture.createView({ baseMipLevel: mip, mipLevelCount: 1 }),
                        );
                        chain = {
                            texture,
                            width,
                            height,
                            views: mipViews,
                            groups: mipViews.map((input) => group(world, state, camera!, input)),
                            blends: mipViews.map(() => ({ r: 0, g: 0, b: 0, a: 1 })),
                        };
                        camera.chain = chain;
                        camera.groups = new WeakMap();
                    }
                    const settings = camera.settings;
                    settings.intensity = s.intensity.get(eid);
                    settings.lowFrequencyBoost = s.lowFrequencyBoost.get(eid);
                    settings.lowFrequencyBoostCurvature = s.lowFrequencyBoostCurvature.get(eid);
                    settings.highPassFrequency = s.highPassFrequency.get(eid);
                    settings.compositeMode = s.compositeMode.get(eid);
                    for (let mip = 0; mip < count; mip++) {
                        const blend = blendFactor(settings, mip, count - 1);
                        const color = chain.blends[mip];
                        color.r = color.g = color.b = blend;
                    }
                    const b = camera.bytes;
                    const threshold = s.threshold.get(eid);
                    const knee = threshold * Math.min(Math.max(s.thresholdSoftness.get(eid), 0), 1);
                    b[0] = threshold;
                    b[1] = threshold - knee;
                    b[2] = 2 * knee;
                    b[3] = 0.25 / (knee + 0.00001);
                    b[4] = s.scale.x.get(eid);
                    b[5] = s.scale.y.get(eid);
                    b[6] = chain.blends[0].r!;
                    camera.mode = settings.compositeMode === BloomCompositeMode.Additive ? 1 : 0;
                    camera.words[7] = camera.mode;
                    camera.words[8] = view.sortedIndexForTarget ? 1 : 0;
                    world.gpu.device.queue.writeBuffer(camera.buffer, 0, b);
                    let stack = effects.get(eid);
                    if (!stack) {
                        stack = { before: [], after: [] };
                        effects.set(eid, stack);
                    }
                    if (!stack.before.includes(camera.effect)) stack.before.unshift(camera.effect);
                }
                for (const [eid, camera] of state.cameras) {
                    if (
                        views.has(eid) &&
                        world.has(eid, Camera) &&
                        world.has(eid, Bloom) &&
                        s.intensity.get(eid) !== 0
                    )
                        continue;
                    unregister(world, eid, camera);
                    state.cameras.delete(eid);
                }
            },
        },
    ],
    async warm(world) {
        const state = world.resource(stateKey);
        const root = world.gpu.root;
        state.layout = root.unwrap(layout);
        state.sampler = world.gpu.device.createSampler({
            minFilter: "linear",
            magFilter: "linear",
        });
        state.first = root.unwrap(
            root.createRenderPipeline({
                vertex: fullscreenVertex,
                fragment: first,
                targets: { format: "rgba16float" },
            }),
        );
        state.down = root.unwrap(
            root.createRenderPipeline({
                vertex: fullscreenVertex,
                fragment: down,
                targets: { format: "rgba16float" },
            }),
        );
        state.composite = root.unwrap(
            root.createRenderPipeline({
                vertex: fullscreenVertex,
                fragment: composite,
                targets: { format: "rgba16float" },
            }),
        );
        for (const additive of [false, true]) {
            state.up.push(
                root.unwrap(
                    root.createRenderPipeline({
                        vertex: fullscreenVertex,
                        fragment: up,
                        targets: {
                            format: "rgba16float",
                            blend: {
                                color: {
                                    srcFactor: "constant",
                                    dstFactor: additive ? "one" : "one-minus-constant",
                                    operation: "add",
                                },
                                alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
                            },
                        },
                    }),
                ),
            );
        }
        precompile(world, "bloom", () => [
            state.first!,
            state.down!,
            ...state.up,
            state.composite!,
        ]);
    },
    dispose(world) {
        const state = world.resource(stateKey);
        for (const [eid, camera] of state.cameras) unregister(world, eid, camera);
        state.cameras.clear();
        state.first = state.down = state.composite = null;
        state.up.length = 0;
        state.layout = null;
        state.sampler = null;
    },
};
