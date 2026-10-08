import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import type { Plugin } from "../../engine";
import { createApp } from "../../engine/app";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    CorePipelinePlugin,
    CustomPresentation,
    captureTexture,
    EffectPasses,
    PresentationSystem,
    RenderContext,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "./index";
import { tonemappingStateKey } from "./tonemapping-state";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

const Replacement: Plugin = {
    name: "Replacement",
    dependencies: [CorePipelinePlugin],
    systems: [
        {
            name: "replacement-presentation",
            group: "draw",
            after: [PresentationSystem],
            update(world) {
                for (const eid of world.query([Camera, CustomPresentation])) {
                    const view = world.resource(Views).get(eid);
                    if (!view?.present) continue;
                    const pass = world.resource(RenderContext).encoder!.beginRenderPass({
                        label: "replacement-presentation",
                        colorAttachments: [
                            {
                                view: view.present,
                                loadOp: "clear",
                                storeOp: "store",
                                clearValue: [0, 1, 0, 1],
                            },
                        ],
                    });
                    pass.end();
                }
            },
        },
    ],
};

function pixels(bytes: Uint8ClampedArray, color: number[]) {
    for (let i = 0; i < bytes.length; i += 4)
        expect(Array.from(bytes.subarray(i, i + 4))).toEqual(color);
}

test("a replacement presents one view while core presents another without steady bind groups", async () => {
    const app = await createApp({ plugins: [Replacement] });
    const { world } = app;
    const device = world.gpu.device;
    console.log("custom presentation adapter:", world.gpu.adapter);
    device.pushErrorScope("validation");
    const createBindGroup = device.createBindGroup.bind(device);
    let groups = 0;
    device.createBindGroup = (descriptor) => {
        groups++;
        return createBindGroup(descriptor);
    };
    try {
        const custom = world.create();
        const core = world.create();
        for (const eid of [custom, core]) {
            world.add(eid, Transform, { translation: [0, 0, 5, 0] });
            world.add(eid, Camera, { clearColor: 0xffffff });
            world.add(eid, Tonemapping, { method: TonemappingMethod.None });
            attachTexture(world, eid, { width: 8, height: 4 });
        }
        world.add(custom, CustomPresentation);
        world.resource(EffectPasses).set(custom, {
            before: [
                () => {
                    throw new Error("core ran an effect on a replaced view");
                },
            ],
            after: [],
        });
        world.step(0);
        pixels((await captureTexture(world, custom)).rgba, [0, 255, 0, 255]);
        pixels((await captureTexture(world, core)).rgba, [255, 255, 255, 255]);
        const state = world.resource(tonemappingStateKey);
        expect(state.groups.has(world.resource(Views).get(custom)!)).toBe(false);
        expect(state.groups.has(world.resource(Views).get(core)!)).toBe(true);
        groups = 0;
        for (let i = 0; i < 20; i++) world.step(1 / 60);
        expect(groups).toBe(0);
        pixels((await captureTexture(world, custom)).rgba, [0, 255, 0, 255]);
        pixels((await captureTexture(world, core)).rgba, [255, 255, 255, 255]);
        world.resource(EffectPasses).delete(custom);
        world.remove(custom, CustomPresentation);
        world.step(0);
        pixels((await captureTexture(world, custom)).rgba, [255, 255, 255, 255]);
    } finally {
        device.createBindGroup = createBindGroup;
        app.dispose();
        expect(await device.popErrorScope()).toBeNull();
    }
});
