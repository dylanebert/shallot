import type { System } from "../../engine";
import { renderFrameKey } from "./frame-state";
import { Render } from "./render";
import { type View, Views } from "./view";

function clearTargets(view: View): void {
    view.framebuffer = null;
    view.framebufferFormat = undefined;
    view.present = null;
}

/** closes the frame: submits the encoder, advances `world.gpu.frame` */
export const EndFrameSystem: System = {
    group: "draw",
    terminal: true,
    update(world) {
        const _render = world.resource(Render);
        const _renderFrame = world.resource(renderFrameKey);

        const device = world.gpu.device;
        if (!device) return;
        const encoder = _render.encoder;
        if (!encoder)
            throw new Error("render submission requires BeginFrameSystem to open an encoder");
        _renderFrame.submit[0] = encoder.finish();
        device.queue.submit(_renderFrame.submit);
        for (const view of world.resource(Views).values()) {
            if (view.texture && view.present) view.presented = true;
        }
        world.endGpuFrame();
        _render.encoder = null;
        world.resource(Views).forEach(clearTargets);
    },
};
