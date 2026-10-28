import { Camera, component, type Plugin, u32 } from "@dylanebert/shallot";
import {
    CorePipelinePlugin,
    type EffectPass,
    EffectPasses,
    fullscreenVertex,
    TonemappingSystem,
} from "@dylanebert/shallot/rendering";
import {
    beginRenderPass,
    draw,
    endRenderPass,
    precompile,
    setRenderBindGroup,
    setRenderPipeline,
} from "@dylanebert/shallot/runtime";
import tgpu from "typegpu";
import * as d from "typegpu/data";

export const Posterize = component(
    "Posterize",
    { levels: u32, dither: u32 },
    { defaults: () => ({ levels: 8, dither: 1 }) },
);
const layout = tgpu
    .bindGroupLayout({
        image: {
            texture: d.texture2d(d.f32),
            sampleType: "unfilterable-float",
            visibility: ["fragment"],
        },
        settings: { uniform: d.vec2u, visibility: ["fragment"] },
    })
    .$idx(0);
const quantize = tgpu.fn(
    [d.vec4f, d.texture2d(d.f32), d.vec2u],
    d.vec4f,
)(`(position: vec4f, image: texture_2d<f32>, settings: vec2u) -> vec4f {
    let bayer = array<f32, 16>(0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
    let pixel = vec2u(position.xy);
    let bands = f32(settings.x - 1u);
    let offset = select(0.0, (bayer[pixel.y % 4u * 4u + pixel.x % 4u] + 0.5) / 16.0 - 0.5, settings.y != 0u);
    let color = textureLoad(image, pixel, 0).rgb;
    return vec4f(clamp(floor(color * bands + offset + 0.5) / bands, vec3f(0), vec3f(1)), 1);
}`);
const fragment = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
    (input) => {
        "use gpu";
        return quantize(input.position, layout.$.image, layout.$.settings);
    },
);
const State = {
    create: () => ({
        pipelines: new Map<GPUTextureFormat, GPURenderPipeline>(),
        effects: new Map<number, EffectPass>(),
        buffer: null as GPUBuffer | null,
    }),
};
export const PosterizePlugin: Plugin = {
    name: "Posterize",
    gpu: {},
    dependencies: [CorePipelinePlugin],
    components: [Posterize],
    async warm(world) {
        const state = world.resource(State);
        const root = world.gpu.root;
        state.buffer = root.unwrap(root.createBuffer(d.vec2u).$usage("uniform"));
        for (const format of new Set<GPUTextureFormat>([
            "rgba8unorm",
            navigator.gpu.getPreferredCanvasFormat(),
        ])) {
            const pipeline = root.unwrap(
                root.createRenderPipeline({
                    vertex: fullscreenVertex,
                    fragment,
                    targets: { format },
                }),
            );
            state.pipelines.set(format, pipeline);
            precompile(world, `posterize-${format}`, () => [pipeline]);
        }
    },
    systems: [
        {
            name: "posterize",
            group: "draw",
            before: [TonemappingSystem],
            update(world) {
                const state = world.resource(State);
                for (const eid of world.query([Camera, Posterize])) {
                    if (state.effects.has(eid)) continue;
                    const bytes = new Uint32Array(2);
                    const groups = new WeakMap<GPUTextureView, GPUBindGroup>();
                    const effect: EffectPass = (w, camera, view, input, output) => {
                        const values = w.storage(Posterize);
                        bytes[0] = values.levels.get(camera);
                        bytes[1] = values.dither.get(camera);
                        w.gpu.device.queue.writeBuffer(state.buffer!, 0, bytes);
                        let group = groups.get(input);
                        if (!group) {
                            group = w.gpu.device.createBindGroup({
                                layout: w.gpu.root.unwrap(layout),
                                entries: [
                                    { binding: 0, resource: input },
                                    { binding: 1, resource: { buffer: state.buffer! } },
                                ],
                            });
                            groups.set(input, group);
                        }
                        const pass = beginRenderPass(w.frameEncoder()!, {
                            label: "posterize",
                            colorAttachments: [{ view: output, loadOp: "clear", storeOp: "store" }],
                        });
                        setRenderPipeline(
                            pass,
                            state.pipelines.get(
                                output === view.present
                                    ? navigator.gpu.getPreferredCanvasFormat()
                                    : "rgba8unorm",
                            )!,
                        );
                        setRenderBindGroup(pass, 0, group);
                        draw(pass, 3);
                        endRenderPass(pass);
                    };
                    state.effects.set(eid, effect);
                    const effects = world.resource(EffectPasses);
                    if (!effects.has(eid)) effects.set(eid, { before: [], after: [] });
                    effects.get(eid)!.after.push(effect);
                }
            },
        },
    ],
};
