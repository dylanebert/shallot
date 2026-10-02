import { type Plugin, registration, type System, type World } from "../../engine";
import { Camera } from "./camera";
import { Render } from "./render";
import { BeginFrameSystem, OverlaySystem, RenderingPlugin } from "./substrate";
import {
    type ColorLane,
    colorPassDescriptor,
    colorTargets,
    DepthPrepass,
    disposeViewTargets,
    initializeViewTargets,
    PickingPrepass,
    prepassDescriptor,
    prepassLanes,
} from "./targets";
import { TonemappingPlugin, TonemappingSystem } from "./tonemapping-state";
import { type View, Views } from "./view";

/** Records into a core-owned pass. Records must not end the pass. Order within each phase is registration order. */
export interface PhaseRenderer {
    prepass?(
        world: World,
        eid: number,
        view: View,
        pass: GPURenderPassEncoder,
        lanes: ColorLane[],
    ): void;
    opaque?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
    transparent?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
}
export const RenderPhases = { create: (): PhaseRenderer[] => [] };
export const PrepassSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        const encoder = world.resource(Render).encoder;
        if (!encoder) return;
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer) continue;
            const requested = prepassLanes(world, eid, view);
            if (!requested) continue;
            const pass = encoder.beginRenderPass(
                prepassDescriptor(world, eid, view, requested.lanes, requested.storeDepth),
            );
            for (const renderer of world.resource(RenderPhases))
                renderer.prepass?.(world, eid, view, pass, requested.lanes);
            pass.end();
        }
    },
};
export const MainPassSystem: System = {
    group: "draw",
    after: [PrepassSystem],
    before: [OverlaySystem],
    update(world) {
        const encoder = world.resource(Render).encoder;
        if (!encoder) return;
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer) continue;
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

/** Optional shared view pipeline: clear, targets, prepass lanes and opaque/transparent records. */
export const CorePipelinePlugin: Plugin = {
    name: "CorePipeline",
    dependencies: [RenderingPlugin],
    systems: [PrepassSystem, MainPassSystem, TonemappingSystem],
    components: [
        ...(TonemappingPlugin.components ?? []),
        registration("DepthPrepass", DepthPrepass),
        registration("PickingPrepass", PickingPrepass),
    ],
    initialize(world) {
        initializeViewTargets(world);
        world.resource(RenderPhases);
        TonemappingPlugin.initialize?.(world);
    },
    warm: TonemappingPlugin.warm,
    dispose(world) {
        disposeViewTargets(world);
        TonemappingPlugin.dispose?.(world);
    },
};
