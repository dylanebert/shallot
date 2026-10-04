import type { Plugin, System, World } from "../../engine";
import { component, u32 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { ColorGrading, GradingConfig } from "./color-grading";
import {
    BeginFrameSystem,
    Camera,
    MAX_VIEWS,
    PresentationSystem,
    RenderContext,
    RenderingPlugin,
    type View,
    Views,
} from "./substrate";
import { composite } from "./tonemapping";

export { ColorGrading } from "./color-grading";
export { TonemappingMethod } from "./tonemap";

/** Per-camera operator; absent cameras use TonyMcMapface. None accepts display-ready linear images. */
export const Tonemapping = component("Tonemapping", { method: u32 });

/** Camera marker: core skips this view's tonemapping and EffectPasses.
 * The replacement owns grading and encoding and must write every presented pixel
 * on RenderContext.encoder after PresentationSystem and before EndFrameSystem.
 * Removing the marker resumes core presentation on the next draw.
 */
export const CustomPresentation = component("CustomPresentation", {});

/** An effect records commands on the frame encoder. Input and output never alias.
 * Before-tonemapping passes receive linear HDR; after-tonemapping passes receive
 * encoded display-referred values in an rgba8unorm intermediate. The last pass
 * writes the presented target in the preferred canvas format. Before outputs are
 * rgba16float; non-final after outputs are rgba8unorm. Register before the view draws; keep registrations
 * stable during draw. A pass must write every output pixel.
 */
export type EffectPass = (
    world: World,
    eid: number,
    view: View,
    input: GPUTextureView,
    output: GPUTextureView,
) => void;
export const EffectPasses = {
    create: () => new Map<number, { before: EffectPass[]; after: EffectPass[] }>(),
};

interface Target {
    texture: GPUTexture;
    view: GPUTextureView;
}
interface Targets {
    eid: number;
    width: number;
    height: number;
    hdr: Target[];
    display: Target[];
}
const tonemappingState = () => {
    const bytes = new Float32Array(28);
    return {
        configs: [] as ReturnType<typeof configBuffer>[],
        built: null as ReturnType<typeof composite> | null,
        pipeline: null as GPURenderPipeline | null,
        displayPipeline: null as GPURenderPipeline | null,
        buffers: [] as GPUBuffer[],
        groups: new WeakMap<View, { input: GPUTextureView; group: GPUBindGroup }>(),
        targets: new Map<View, Targets>(),
        bytes,
        words: new Uint32Array(bytes.buffer),
        pass: {
            label: "tonemapping",
            colorAttachments: [{ view: null!, loadOp: "clear", storeOp: "store" }],
        } as GPURenderPassDescriptor,
    };
};
export const tonemappingStateKey = { create: tonemappingState };

function configBuffer(world: World, slot: number) {
    return world.gpu.root
        .createBuffer(GradingConfig)
        .$usage("uniform")
        .$name(`tonemapping-config-${slot}`);
}

function target(world: World, view: View, format: GPUTextureFormat): Target {
    const texture = world.gpu.device.createTexture({
        label: "effect-intermediate",
        size: [view.width, view.height],
        format,
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return { texture, view: texture.createView() };
}
function destroyTargets(targets: Targets) {
    for (const t of targets.hdr) t.texture.destroy();
    for (const t of targets.display) t.texture.destroy();
}
function intermediates(world: World, eid: number, view: View, before: number, after: number) {
    const state = world.resource(tonemappingStateKey);
    let targets = state.targets.get(view);
    if (!targets || targets.width !== view.width || targets.height !== view.height) {
        if (targets) destroyTargets(targets);
        targets = { eid, width: view.width, height: view.height, hdr: [], display: [] };
        state.targets.set(view, targets);
    }
    const hdrCount = Math.min(before, 2);
    const displayCount = Math.min(after, 2);
    while (targets.hdr.length < hdrCount) targets.hdr.push(target(world, view, "rgba16float"));
    while (targets.display.length < displayCount)
        targets.display.push(target(world, view, "rgba8unorm"));
    while (targets.hdr.length > hdrCount) targets.hdr.pop()!.texture.destroy();
    while (targets.display.length > displayCount) targets.display.pop()!.texture.destroy();
    return targets;
}

const CAMERAS = [Camera];
const SECTION_FIELDS = ["saturation", "contrast", "gamma", "gain", "lift"] as const;
export const TonemappingSystem: System = {
    name: "tonemapping",
    group: "draw",
    after: [BeginFrameSystem, PresentationSystem],
    update(world) {
        const state = world.resource(tonemappingStateKey);
        const encoder = world.resource(RenderContext).encoder;
        if (!encoder || !state.built || !state.pipeline) return;
        const device = world.gpu.device;
        const grading = world.storage(ColorGrading);
        const methods = world.storage(Tonemapping);
        for (const eid of world.query(CAMERAS)) {
            const view = world.resource(Views).get(eid);
            if (!view?.present || !view.framebuffer || world.has(eid, CustomPresentation)) continue;
            const effects = world.resource(EffectPasses).get(eid);
            const before = effects?.before.length ?? 0;
            const after = effects?.after.length ?? 0;
            const targets = intermediates(world, eid, view, before, after);
            let input = view.framebuffer;
            for (let i = 0; i < before; i++) {
                const output = targets.hdr[i % 2].view;
                effects!.before[i](world, eid, view, input, output);
                input = output;
            }
            const b = state.bytes;
            const hasGrade = world.has(eid, ColorGrading);
            b[0] = hasGrade ? grading.exposure.get(eid) : 0;
            b[1] = hasGrade ? grading.temperature.get(eid) : 0;
            b[2] = hasGrade ? grading.tint.get(eid) : 0;
            b[3] = hasGrade ? grading.hue.get(eid) : 0;
            b[4] = hasGrade ? grading.postSaturation.get(eid) : 1;
            b[5] = 0;
            b[6] = hasGrade ? grading.midtonesRange.x.get(eid) : 0.2;
            b[7] = hasGrade ? grading.midtonesRange.y.get(eid) : 0.7;
            for (let i = 0; i < 5; i++) {
                const field = grading[SECTION_FIELDS[i]];
                b[8 + i * 4] = hasGrade ? field.x.get(eid) : i === 4 ? 0 : 1;
                b[9 + i * 4] = hasGrade ? field.y.get(eid) : i === 4 ? 0 : 1;
                b[10 + i * 4] = hasGrade ? field.z.get(eid) : i === 4 ? 0 : 1;
                b[11 + i * 4] = 0;
            }
            state.words[5] = world.has(eid, Tonemapping) ? methods.method.get(eid) : 0;
            device.queue.writeBuffer(state.buffers[view.slot], 0, b);
            let cached = state.groups.get(view);
            if (!cached || cached.input !== input) {
                cached = {
                    input,
                    group: device.createBindGroup({
                        layout: world.gpu.root.unwrap(state.built.layout),
                        entries: [
                            { binding: 0, resource: input },
                            { binding: 1, resource: { buffer: state.buffers[view.slot] } },
                            { binding: 2, resource: state.built.lut },
                            { binding: 3, resource: state.built.sampler },
                        ],
                    }),
                };
                state.groups.set(view, cached);
            }
            const attachment = (state.pass.colorAttachments as GPURenderPassColorAttachment[])[0];
            attachment.view = after ? targets.display[0].view : view.present;
            state.pass.timestampWrites = world.gpu.span?.("tonemapping");
            const pass = encoder.beginRenderPass(state.pass);
            pass.setPipeline(after ? state.displayPipeline! : state.pipeline);
            pass.setBindGroup(0, cached.group);
            pass.draw(3);
            pass.end();
            input = attachment.view;
            for (let i = 0; i < after; i++) {
                const output = i === after - 1 ? view.present : targets.display[(i + 1) % 2].view;
                effects!.after[i](world, eid, view, input, output);
                input = output;
            }
        }
        for (const [view, targets] of state.targets) {
            if (world.resource(Views).get(targets.eid) !== view) {
                destroyTargets(targets);
                state.targets.delete(view);
            }
        }
    },
};

/** Included by CorePipelinePlugin, not a separate presentation pass. */
export const TonemappingPlugin: Plugin = {
    name: "Tonemapping",
    dependencies: [RenderingPlugin],
    components: [Tonemapping, CustomPresentation, ColorGrading],
    initialize(world) {
        world.resource(tonemappingStateKey);
        world.resource(EffectPasses);
    },
    async warm(world) {
        const state = world.resource(tonemappingStateKey);
        for (const buffer of state.configs) buffer.destroy();
        state.configs.length = 0;
        state.groups = new WeakMap();
        for (let slot = 0; slot < MAX_VIEWS; slot++) state.configs.push(configBuffer(world, slot));
        state.buffers = state.configs.map((buffer) => world.gpu.root.unwrap(buffer));
        state.built = composite(world, navigator.gpu.getPreferredCanvasFormat());
        state.pipeline = world.gpu.root.unwrap(state.built.pipeline);
        const display = composite(world, "rgba8unorm");
        state.displayPipeline = world.gpu.root.unwrap(display.pipeline);
        precompile(world, "tonemapping-display", () => [state.displayPipeline!]);
        const { pipeline, layout, lut, sampler } = state.built;
        precompile(world, "tonemapping", () => {
            const source = world.gpu.device.createTexture({
                size: [1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const bound = pipeline.with(
                world.gpu.root.createBindGroup(layout, {
                    input: source.createView(),
                    grading: state.configs[0],
                    lut,
                    sampler,
                }),
            );
            source.destroy();
            return bound;
        });
    },
    dispose(world) {
        const state = world.resource(tonemappingStateKey);
        for (const targets of state.targets.values()) destroyTargets(targets);
        state.targets.clear();
        for (const buffer of state.configs) buffer.destroy();
        state.configs.length = 0;
        state.buffers.length = 0;
        state.groups = new WeakMap();
        state.pipeline = null;
        state.displayPipeline = null;
        state.built = null;
        world.resource(EffectPasses).clear();
    },
};
