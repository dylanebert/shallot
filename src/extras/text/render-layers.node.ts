import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, RenderLayers, TransparentPhases } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { isolationFont } from "./font.fixture";
import { internText, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

for (const [itemMask, cameraMask] of [
    [2, 1],
    [1, 2],
])
    test(`text mask ${itemMask} is not drawn by camera mask ${cameraMask}`, () => {
        const { world } = subjects()[0];
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { antialias: 0 });
        if (cameraMask !== 1) world.add(camera, RenderLayers, { mask: cameraMask });
        attachTexture(world, camera, { width: 32, height: 32 });
        const label = world.create();
        world.add(label, Transform);
        world.add(label, Text, { font, content: internText(world, "i") });
        if (itemMask !== 1) world.add(label, RenderLayers, { mask: itemMask });
        try {
            world.step(1 / 60);
            expect(world.resource(TransparentPhases).get(camera)!.metrics.draws).toBe(0);
            if (!world.has(camera, RenderLayers)) world.add(camera, RenderLayers);
            world.storage(RenderLayers).mask.set(camera, 3);
            world.step(1 / 60);
            expect(world.resource(TransparentPhases).get(camera)!.metrics.draws).toBe(1);
        } finally {
            world.destroy(label);
            world.destroy(camera);
        }
    });
