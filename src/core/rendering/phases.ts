import type { Plugin, System, World } from "../../engine";
import { Camera } from "./camera";
import { BeginFrameSystem, OverlaySystem, RenderingPlugin } from "./substrate";
import {
    colorPassDescriptor,
    colorTargets,
    DepthPrepass,
    DepthPrepassRequests,
    disposeViewTargets,
    initializeViewTargets,
    prepassDescriptor,
} from "./targets";
import { TonemappingPlugin, TonemappingSystem } from "./tonemapping-state";
import { type View, Views } from "./view";

/** Records into a core-owned pass. Records must not end the pass. Each phase runs renderers in the order they are pushed to `RenderPhases`; plugins push in `initialize`, which runs in the composition's dependency order. */
export interface PhaseRenderer {
    prepass?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
    opaque?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
    transparent?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
}
export const RenderPhases = { create: (): PhaseRenderer[] => [] };
export const PrepassSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer) continue;
            const encoder = world.frameEncoder()!;
            view.depth = null;
            let requested = world.has(eid, DepthPrepass);
            if (!requested) {
                for (const request of world.resource(DepthPrepassRequests)) {
                    if (request(world, eid, view)) {
                        requested = true;
                        break;
                    }
                }
            }
            if (!requested) continue;
            const pass = encoder.beginRenderPass(prepassDescriptor(world, eid, view));
            for (const renderer of world.resource(RenderPhases))
                renderer.prepass?.(world, eid, view, pass);
            pass.end();
        }
    },
};
export const MainPassSystem: System = {
    group: "draw",
    after: [PrepassSystem],
    before: [OverlaySystem],
    update(world) {
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer) continue;
            const encoder = world.frameEncoder()!;
            const targets = colorTargets(
                world,
                eid,
                view.width,
                view.height,
                world.storage(Camera).antialias.get(eid) !== 0,
            );
            const pass = encoder.beginRenderPass(
                colorPassDescriptor(world, eid, targets, view.framebuffer),
            );
            for (const renderer of world.resource(RenderPhases))
                renderer.opaque?.(world, eid, view, pass);
            for (const renderer of world.resource(RenderPhases))
                renderer.transparent?.(world, eid, view, pass);
            pass.end();
        }
    },
};

/** Optional shared view pipeline: clear, targets, depth prepass and opaque/transparent records. */
export const CorePipelinePlugin: Plugin = {
    gpu: {},
    name: "CorePipeline",
    dependencies: [RenderingPlugin],
    systems: [PrepassSystem, MainPassSystem, TonemappingSystem],
    components: [...(TonemappingPlugin.components ?? []), DepthPrepass],
    initialize(world) {
        initializeViewTargets(world);
        world.resource(RenderPhases);
        world.resource(DepthPrepassRequests);
        TonemappingPlugin.initialize?.(world);
    },
    warm: TonemappingPlugin.warm,
    dispose(world) {
        disposeViewTargets(world);
        TonemappingPlugin.dispose?.(world);
    },
};
