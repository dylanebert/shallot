// Bevy 24661940b42fa8c8a89538a1d3572f7fe6dbd49a, effect_stack/vignette.wgsl.
// MIT OR Apache-2.0, copyright Bevy contributors; see ../../core/rendering/bevy-LICENSE-MIT.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    Camera,
    CorePipelinePlugin,
    type EffectPass,
    EffectPasses,
    fullscreenVertex,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import { component, f32, type Plugin, vec2, vec4, type World } from "../../engine";
import { precompile } from "../../engine/runtime";

/** Bevy's HDR vignette. Color is sRGB, center is UV; only carrying cameras run the pass. */
export const Vignette = component(
    "Vignette",
    {
        intensity: f32,
        radius: f32,
        smoothness: f32,
        roundness: f32,
        center: vec2,
        edgeCompensation: f32,
        color: vec4,
    },
    {
        defaults: () => ({
            intensity: 1,
            radius: 0.75,
            smoothness: 5,
            roundness: 1,
            center: [0.5, 0.5],
            edgeCompensation: 1,
            color: [0, 0, 0, 1],
        }),
    },
);
const Settings = d.struct({
    intensity: d.f32,
    radius: d.f32,
    smoothness: d.f32,
    roundness: d.f32,
    center: d.vec2f,
    edgeCompensation: d.f32,
    unused: d.u32,
    color: d.vec4f,
});
const layout = tgpu
    .bindGroupLayout({
        input: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
        settings: { uniform: Settings, visibility: ["fragment"] },
    })
    .$idx(0);
const vignette = tgpu
    .fn(
        [d.vec2f, d.vec3f, d.vec2f, Settings],
        d.vec3f,
    )(`(uv: vec2f, color: vec3f, resolution: vec2f, g: Settings) -> vec3f {
    if (g.intensity < 1.19209290e-07) { return color; }
    let intensity = saturate(g.intensity);
    let radius = max(g.radius, 0.0);
    let smoothness = max(g.smoothness, 0.0);
    let roundness = clamp(g.roundness, 1.19209290e-07, 2.0-1.19209290e-07);
    let edge_comp = saturate(g.edgeCompensation);
    let screen_aspect = resolution.x / resolution.y;
    let aspect_ratio = resolution / min(resolution.x, resolution.y);
    let centered_uv = uv - 0.5;
    let offset = (g.center - 0.5) * vec2f(1.0, resolution.y / resolution.x);
    let uv_from_center = centered_uv - offset;
    var scale_vec = aspect_ratio * vec2f(1.0, 1.0 / roundness);
    if (screen_aspect >= 1.0) { scale_vec.x *= mix(1.0, 1.0 / screen_aspect, edge_comp); }
    else { scale_vec.y *= mix(1.0, screen_aspect, edge_comp); }
    let final_uv = uv_from_center * scale_vec;
    let dist = length(final_uv) * (1.0 / radius);
    let factor = pow(clamp(1.0 - dist * dist, 0.0, 1.0), smoothness);
    return mix(color, g.color.rgb, (1.0 - factor) * intensity);
}`)
    .$uses({ Settings });
const fragment = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
    (input) => {
        "use gpu";
        const resolution = d.vec2f(std.textureDimensions(layout.$.input));
        const color = std.textureLoad(layout.$.input, d.vec2u(input.position.xy), 0).xyz;
        return d.vec4f(
            vignette(std.div(input.position.xy, resolution), color, resolution, layout.$.settings),
            1,
        );
    },
);
const stateKey = {
    create: () => ({
        pipeline: null as GPURenderPipeline | null,
        rawLayout: null as GPUBindGroupLayout | null,
        cameras: new Map<
            number,
            {
                buffer: GPUBuffer;
                bytes: Float32Array;
                effect: EffectPass;
                groups: WeakMap<GPUTextureView, GPUBindGroup>;
            }
        >(),
        pass: {
            label: "vignette",
            colorAttachments: [{ view: null!, loadOp: "clear", storeOp: "store" }],
        } as GPURenderPassDescriptor,
    }),
};

export const VignettePlugin: Plugin = {
    gpu: {},
    name: "Vignette",
    dependencies: [CorePipelinePlugin],
    components: [Vignette],
    systems: [
        {
            name: "vignette",
            group: "draw",
            before: [TonemappingSystem],
            update(world) {
                const state = world.resource(stateKey);
                const s = world.storage(Vignette);
                const effects = world.resource(EffectPasses);
                for (const [eid] of world.resource(Views)) {
                    if (
                        !world.has(eid, Camera) ||
                        !world.has(eid, Vignette) ||
                        s.intensity.get(eid) <= 0
                    )
                        continue;
                    let camera = state.cameras.get(eid);
                    if (!camera) {
                        const buffer = world.gpu.root.unwrap(
                            world.gpu.root.createBuffer(Settings).$usage("uniform"),
                        );
                        const groups = new WeakMap<GPUTextureView, GPUBindGroup>();
                        const effect: EffectPass = (w, _eid, _view, input, output) => {
                            let group = groups.get(input);
                            if (!group) {
                                group = w.gpu.device.createBindGroup({
                                    layout: state.rawLayout!,
                                    entries: [
                                        { binding: 0, resource: input },
                                        { binding: 1, resource: { buffer } },
                                    ],
                                });
                                groups.set(input, group);
                            }
                            (
                                state.pass.colorAttachments as GPURenderPassColorAttachment[]
                            )[0].view = output;
                            const pass = w.frameEncoder()!.beginRenderPass(state.pass);
                            pass.setPipeline(state.pipeline!);
                            pass.setBindGroup(0, group);
                            pass.draw(3);
                            pass.end();
                        };
                        camera = { buffer, bytes: new Float32Array(12), groups, effect };
                        state.cameras.set(eid, camera);
                    }
                    let stack = effects.get(eid);
                    if (!stack) {
                        stack = { before: [], after: [] };
                        effects.set(eid, stack);
                    }
                    if (!stack.before.includes(camera.effect)) stack.before.push(camera.effect);
                    const b = camera.bytes;
                    b[0] = s.intensity.get(eid);
                    b[1] = s.radius.get(eid);
                    b[2] = s.smoothness.get(eid);
                    b[3] = s.roundness.get(eid);
                    b[4] = s.center.x.get(eid);
                    b[5] = s.center.y.get(eid);
                    b[6] = s.edgeCompensation.get(eid);
                    b[7] = 0;
                    b[8] = s.color.x.get(eid);
                    b[9] = s.color.y.get(eid);
                    b[10] = s.color.z.get(eid);
                    b[11] = s.color.w.get(eid);
                    world.gpu.device.queue.writeBuffer(camera.buffer, 0, b);
                }
                for (const [eid, camera] of state.cameras) {
                    if (
                        world.has(eid, Vignette) &&
                        world.resource(Views).has(eid) &&
                        s.intensity.get(eid) > 0
                    )
                        continue;
                    const stack = effects.get(eid);
                    const index = stack?.before.indexOf(camera.effect) ?? -1;
                    if (index >= 0) stack!.before.splice(index, 1);
                    camera.buffer.destroy();
                    state.cameras.delete(eid);
                }
            },
        },
    ],
    async warm(world: World) {
        const state = world.resource(stateKey);
        const pipeline = world.gpu.root.createRenderPipeline({
            vertex: fullscreenVertex,
            fragment,
            targets: { format: "rgba16float" },
        });
        state.pipeline = world.gpu.root.unwrap(pipeline);
        state.rawLayout = world.gpu.root.unwrap(layout);
        precompile(world, "vignette", () => {
            const input = world.gpu.device.createTexture({
                size: [1, 1],
                format: "rgba16float",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const buffer = world.gpu.root.createBuffer(Settings).$usage("uniform");
            const bound = pipeline.with(
                world.gpu.root.createBindGroup(layout, {
                    input: input.createView(),
                    settings: buffer,
                }),
            );
            input.destroy();
            buffer.destroy();
            return bound;
        });
    },
    dispose(world) {
        const state = world.resource(stateKey);
        for (const [eid, camera] of state.cameras) {
            const stack = world.resource(EffectPasses).get(eid);
            const index = stack?.before.indexOf(camera.effect) ?? -1;
            if (index >= 0) stack!.before.splice(index, 1);
            camera.buffer.destroy();
        }
        state.cameras.clear();
        state.pipeline = null;
        state.rawLayout = null;
    },
};
