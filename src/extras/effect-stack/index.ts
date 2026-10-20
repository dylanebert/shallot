// Bevy b3bdd75be, crates/bevy_post_process/src/effect_stack/.
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
import { ImageBindings } from "../../core/rendering/image";
import { component, f32, i32, type Plugin, u32, vec2, type World } from "../../engine";
import {
    beginRenderPass,
    draw,
    endRenderPass,
    precompile,
    setRenderBindGroup,
    setRenderPipeline,
} from "../../engine/runtime";

/** Radial streaks, with intensity as a fraction of the view size. colorLut is a registerImage id,
 * or -1 for the default red/green/blue gradient. The LUT is sampled along its vertical center. */
export const ChromaticAberration = component(
    "ChromaticAberration",
    {
        colorLut: i32,
        intensity: f32,
        maxSamples: u32,
    },
    { defaults: () => ({ colorLut: -1, intensity: 0.02, maxSamples: 8 }) },
);

/** Brown-Conrady radial distortion. Center is UV; scale zooms the result, multiplier weights each
 * axis and edgeCurvature controls k2. Negative intensity gives pincushion distortion. */
export const LensDistortion = component(
    "LensDistortion",
    {
        intensity: f32,
        scale: f32,
        multiplier: vec2,
        center: vec2,
        edgeCurvature: f32,
    },
    {
        defaults: () => ({
            intensity: 0.5,
            scale: 1,
            multiplier: [1, 1],
            center: [0.5, 0.5],
            edgeCurvature: 0,
        }),
    },
);

/** HDR vignette. Color is 0xRRGGBB sRGB, mixed unconverted; center is UV.
 * Stacked cameras mix toward color scaled by their premultiplied coverage. */
export const Vignette = component(
    "Vignette",
    {
        intensity: f32,
        radius: f32,
        smoothness: f32,
        roundness: f32,
        center: vec2,
        edgeCompensation: f32,
        color: f32,
    },
    {
        defaults: () => ({
            intensity: 1,
            radius: 0.75,
            smoothness: 5,
            roundness: 1,
            center: [0.5, 0.5],
            edgeCompensation: 1,
            color: 0,
        }),
    },
);

const CAMERAS = [Camera];
const Settings = d.struct({
    intensity: d.f32,
    radius: d.f32,
    smoothness: d.f32,
    roundness: d.f32,
    center: d.vec2f,
    edgeCompensation: d.f32,
    stacked: d.u32,
    color: d.vec4f,
    lensIntensity: d.f32,
    scale: d.f32,
    multiplier: d.vec2f,
    lensCenter: d.vec2f,
    edgeCurvature: d.f32,
    unused: d.u32,
    aberrationIntensity: d.f32,
    maxSamples: d.u32,
    unusedA: d.u32,
    unusedB: d.u32,
});
const layout = tgpu
    .bindGroupLayout({
        input: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
        settings: { uniform: Settings, visibility: ["fragment"] },
        commonSampler: { sampler: "filtering", visibility: ["fragment"] },
        lut: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
    })
    .$idx(0);
const lensDistortion = tgpu
    .fn(
        [d.vec2f, Settings],
        d.vec2f,
    )(`(uv: vec2f, g: Settings) -> vec2f {
    let intensity = g.lensIntensity;
    if (abs(intensity) < 1e-4) { return uv; }
    let uv_centered = uv - g.lensCenter;
    let radius = max(length(uv_centered), 1e-6);
    let direction = uv_centered / radius;
    let adjust = dot(abs(direction), g.multiplier);
    let k1 = intensity * adjust;
    let k2 = k1 * intensity * g.edgeCurvature;
    let r2 = radius * radius;
    let r_distorted = radius * (1.0 + (k1 + k2 * r2) * r2);
    let uv_distorted = direction * r_distorted + g.lensCenter;
    return clamp((uv_distorted - g.lensCenter) / g.scale + g.lensCenter, vec2f(0.0), vec2f(1.0));
}`)
    .$uses({ Settings });
const aberration = tgpu
    .fn(
        [d.vec2f],
        d.vec4f,
    )(`(start_pos: vec2f) -> vec4f {
    let g = bound.settings;
    let end_pos = mix(start_pos, vec2f(0.5), g.aberrationIntensity);
    let texel_length = length((end_pos - start_pos) * vec2f(textureDimensions(bound.input)));
    let sample_count = min(u32(ceil(texel_length)), g.maxSamples);
    if (sample_count > 1u) {
        let lut_u_offset = 0.5 / f32(textureDimensions(bound.lut).x);
        var sample_sum = vec3f(0.0);
        var coverage_sum = vec3f(0.0);
        var modulate_sum = vec3f(0.0);
        for (var sample_index = 0u; sample_index < sample_count; sample_index += 1u) {
            let t = (f32(sample_index) + 0.5) / f32(sample_count);
            let sample_uv = mix(start_pos, end_pos, t);
            let value = textureSampleLevel(bound.input, bound.commonSampler, sample_uv, 0.0);
            let lut_u = mix(lut_u_offset, 1.0 - lut_u_offset, t);
            let modulate = textureSampleLevel(bound.lut, bound.commonSampler, vec2f(lut_u, 0.5), 0.0).rgb;
            sample_sum += value.rgb * modulate;
            coverage_sum += value.a * modulate;
            modulate_sum += modulate;
        }
        let color = sample_sum / modulate_sum;
        // Compositing has one alpha for three displaced channels. Their maximum coverage keeps
        // Bevy's RGB sum premultiplied without discarding any channel's fringe.
        if (g.stacked != 0u) {
            let coverage = coverage_sum / modulate_sum;
            return vec4f(color, max(max(coverage.r, coverage.g), coverage.b));
        }
        return vec4f(color, textureSampleLevel(bound.input, bound.commonSampler, start_pos, 0.0).a);
    }
    return textureSampleLevel(bound.input, bound.commonSampler, start_pos, 0.0);
}`)
    .$uses({ bound: layout.$ });
const vignette = tgpu
    .fn(
        [d.vec2f, d.vec3f, d.f32, d.vec2f, Settings],
        d.vec3f,
    )(`(uv: vec2f, color: vec3f, alpha: f32, resolution: vec2f, g: Settings) -> vec3f {
    if (g.intensity < 1e-4) { return color; }
    let screen_aspect = resolution.x / resolution.y;
    let aspect_ratio = resolution / min(resolution.x, resolution.y);
    let offset = (g.center - 0.5) * vec2f(1.0, resolution.y / resolution.x);
    let uv_from_center = uv - 0.5 - offset;
    var scale_vec = aspect_ratio * vec2f(1.0, 1.0 / g.roundness);
    if (screen_aspect >= 1.0) { scale_vec.x *= mix(1.0, 1.0 / screen_aspect, g.edgeCompensation); }
    else { scale_vec.y *= mix(1.0, screen_aspect, g.edgeCompensation); }
    let dist = length(uv_from_center * scale_vec) * (1.0 / g.radius);
    let factor = pow(clamp(1.0 - dist * dist, 0.0, 1.0), g.smoothness);
    let coverage = select(1.0, alpha, g.stacked != 0u);
    return mix(color, g.color.rgb * coverage, (1.0 - factor) * g.intensity);
}`)
    .$uses({ Settings });
const fragment = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
    (input) => {
        "use gpu";
        const resolution = d.vec2f(std.textureDimensions(layout.$.input));
        const uv = std.div(input.position.xy, resolution);
        const color = aberration(lensDistortion(uv, layout.$.settings));
        return d.vec4f(vignette(uv, color.xyz, color.w, resolution, layout.$.settings), color.w);
    },
);
const stateKey = {
    create: () => ({
        pipeline: null as GPURenderPipeline | null,
        rawLayout: null as GPUBindGroupLayout | null,
        sampler: null as GPUSampler | null,
        defaultLut: null as GPUTextureView | null,
        cameras: new Map<
            number,
            {
                buffer: GPUBuffer;
                bytes: Float32Array;
                words: Uint32Array;
                effect: EffectPass;
                groups: WeakMap<GPUTextureView, GPUBindGroup>;
                lutId: number;
                lut: GPUTextureView;
            }
        >(),
        pass: {
            label: "effect-stack",
            colorAttachments: [{ view: null!, loadOp: "clear", storeOp: "store" }],
        } as GPURenderPassDescriptor,
    }),
};
type StackState = ReturnType<typeof stateKey.create>;
function stackEffect(
    state: StackState,
    camera: {
        buffer: GPUBuffer;
        groups: WeakMap<GPUTextureView, GPUBindGroup>;
        lut: GPUTextureView;
    },
): EffectPass {
    return (world, _eid, _view, input, output) => {
        let group = camera.groups.get(input);
        if (!group) {
            group = world.gpu.device.createBindGroup({
                layout: state.rawLayout!,
                entries: [
                    { binding: 0, resource: input },
                    { binding: 1, resource: { buffer: camera.buffer } },
                    { binding: 2, resource: state.sampler! },
                    { binding: 3, resource: camera.lut },
                ],
            });
            camera.groups.set(input, group);
        }
        (state.pass.colorAttachments as GPURenderPassColorAttachment[])[0].view = output;
        const pass = beginRenderPass(world.frameEncoder()!, state.pass);
        setRenderPipeline(pass, state.pipeline!);
        setRenderBindGroup(pass, 0, group);
        draw(pass, 3);
        endRenderPass(pass);
    };
}
function active(world: World, eid: number): boolean {
    return (
        (world.has(eid, Vignette) && world.storage(Vignette).intensity.get(eid) > 1e-4) ||
        (world.has(eid, LensDistortion) &&
            Math.abs(world.storage(LensDistortion).intensity.get(eid)) > 1e-4) ||
        (world.has(eid, ChromaticAberration) &&
            world.storage(ChromaticAberration).intensity.get(eid) > 1e-4)
    );
}
/** One pre-tonemapping pass for cameras carrying any active stack effect. */
export const EffectStackPlugin: Plugin = {
    gpu: {},
    name: "EffectStack",
    dependencies: [CorePipelinePlugin],
    components: [Vignette, LensDistortion, ChromaticAberration],
    systems: [
        {
            name: "effect-stack",
            group: "draw",
            before: [TonemappingSystem],
            update(world) {
                const state = world.resource(stateKey);
                const effects = world.resource(EffectPasses);
                const views = world.resource(Views);
                const s = world.storage(Vignette);
                const l = world.storage(LensDistortion);
                const a = world.storage(ChromaticAberration);
                for (const eid of world.query(CAMERAS)) {
                    if (!views.has(eid) || !active(world, eid)) continue;
                    let camera = state.cameras.get(eid);
                    if (!camera) {
                        const buffer = world.gpu.root.unwrap(
                            world.gpu.root.createBuffer(Settings).$usage("uniform"),
                        );
                        const bytes = new Float32Array(24);
                        camera = {
                            buffer,
                            bytes,
                            words: new Uint32Array(bytes.buffer),
                            effect: null!,
                            groups: new WeakMap(),
                            lutId: -1,
                            lut: state.defaultLut!,
                        };
                        camera.effect = stackEffect(state, camera);
                        state.cameras.set(eid, camera);
                    }
                    const lutId = world.has(eid, ChromaticAberration) ? a.colorLut.get(eid) : -1;
                    if (camera.lutId !== lutId) {
                        camera.lutId = lutId;
                        camera.lut =
                            lutId < 0
                                ? state.defaultLut!
                                : world.resource(ImageBindings).images.createView({
                                      dimension: "2d",
                                      baseArrayLayer: lutId,
                                      arrayLayerCount: 1,
                                  });
                        camera.groups = new WeakMap();
                    }
                    let stack = effects.get(eid);
                    if (!stack) {
                        stack = { before: [], after: [] };
                        effects.set(eid, stack);
                    }
                    if (!stack.before.includes(camera.effect)) stack.before.push(camera.effect);
                    const b = camera.bytes;
                    const v = world.has(eid, Vignette);
                    b[0] = v ? Math.min(s.intensity.get(eid), 1) : 0;
                    b[1] = v ? Math.max(s.radius.get(eid), 1e-6) : 1;
                    b[2] = v ? Math.max(s.smoothness.get(eid), 0) : 0;
                    b[3] = v ? Math.min(Math.max(s.roundness.get(eid), 1e-6), 2 - 1e-6) : 1;
                    b[4] = v ? s.center.x.get(eid) : 0.5;
                    b[5] = v ? s.center.y.get(eid) : 0.5;
                    b[6] = v ? s.edgeCompensation.get(eid) : 0;
                    camera.words[7] = views.get(eid)!.sortedIndexForTarget ? 1 : 0;
                    const color = v ? s.color.get(eid) : 0;
                    b[8] = ((color >> 16) & 255) / 255;
                    b[9] = ((color >> 8) & 255) / 255;
                    b[10] = (color & 255) / 255;
                    b[11] = 1;
                    const lens = world.has(eid, LensDistortion);
                    b[12] = lens ? l.intensity.get(eid) : 0;
                    b[13] = lens ? Math.max(l.scale.get(eid), 1e-6) : 1;
                    b[14] = lens ? l.multiplier.x.get(eid) : 1;
                    b[15] = lens ? l.multiplier.y.get(eid) : 1;
                    b[16] = lens ? l.center.x.get(eid) : 0.5;
                    b[17] = lens ? l.center.y.get(eid) : 0.5;
                    b[18] = lens ? l.edgeCurvature.get(eid) : 0;
                    const ca = world.has(eid, ChromaticAberration);
                    b[20] = ca ? a.intensity.get(eid) : 0;
                    camera.words[21] = ca ? a.maxSamples.get(eid) : 0;
                    world.gpu.device.queue.writeBuffer(camera.buffer, 0, b);
                }
                for (const [eid, camera] of state.cameras) {
                    if (views.has(eid) && world.has(eid, Camera) && active(world, eid)) continue;
                    const stack = effects.get(eid);
                    const index = stack?.before.indexOf(camera.effect) ?? -1;
                    if (index >= 0) stack!.before.splice(index, 1);
                    camera.buffer.destroy();
                    state.cameras.delete(eid);
                }
            },
        },
    ],
    async warm(world) {
        const state = world.resource(stateKey);
        const device = world.gpu.device;
        const lut = device.createTexture({
            label: "chromatic-aberration-default-lut",
            size: [3, 1],
            format: "rgba8unorm-srgb",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(lut);
        device.queue.writeTexture(
            { texture: lut },
            new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]),
            { bytesPerRow: 12 },
            [3, 1],
        );
        state.defaultLut = lut.createView();
        state.sampler = device.createSampler({
            minFilter: "linear",
            magFilter: "linear",
            mipmapFilter: "linear",
        });
        state.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createRenderPipeline({
                vertex: fullscreenVertex,
                fragment,
                targets: { format: "rgba16float" },
            }),
        );
        state.rawLayout = world.gpu.root.unwrap(layout);
        precompile(world, "effect-stack", () => [state.pipeline!]);
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
        state.defaultLut = null;
        state.sampler = null;
    },
};
