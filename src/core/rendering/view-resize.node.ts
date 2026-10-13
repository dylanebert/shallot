import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { createTestGpuApp } from "../../../scripts/gpu.fixture";
import { Viewports } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { Transform } from "../transform";
import { attachCanvas, Camera, RenderingPlugin, Views } from "./index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("a canvas display resize is sampled by the next frame without a resize callback", async () => {
    const app = await createTestGpuApp({ defaults: false, plugins: [RenderingPlugin] });
    const { world } = app;
    let width = 32;
    let height = 24;
    let context: CanvasContext;
    const canvas = {
        width,
        height,
        clientWidth: width,
        clientHeight: height,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width, height }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, width, height);
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    attachCanvas(camera, canvas, world);

    try {
        world.step(0);
        const view = world.resource(Views).get(camera)!;
        expect([view.width, view.height]).toEqual([32, 24]);

        width = 48;
        height = 36;
        Object.assign(canvas, { clientWidth: width, clientHeight: height });
        context.setSize(width, height);
        world.step(0);

        expect([view.width, view.height]).toEqual([48, 36]);
        expect(world.resource(Viewports).get(view.viewportIndex)).toMatchObject({
            cssWidth: 48,
            cssHeight: 36,
        });
    } finally {
        app.dispose();
    }
});
