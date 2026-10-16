import { Camera, CameraMode, attachTexture, Tonemapping, TonemappingMethod } from "../../src/core/rendering";
import { MeshInstance } from "../../src/core/mesh";
import { Transform } from "../../src/core/transform";
import { createApp } from "../../src/engine";
import { AlphaMode, Materials, MeshMaterial, MeshRenderPlugin, StandardMaterial, StandardRenderer, StandardRenderingPlugin } from "../../src/standard/rendering";
import { Line, LinesPlugin } from "../../src/extras/lines";
import { isolationFont } from "../../src/extras/text/font.fixture";
import { internText, registerFont, Text, TextPlugin } from "../../src/extras/text";

const FIXED_DT = 1 / 60;
const ITEM_COUNT = 8;

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    let font = 0;
    const app = await createApp({
        defaults: false,
        plugins: [TextPlugin, LinesPlugin, MeshRenderPlugin, StandardRenderingPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    });
    const world = app.world;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, 3);
    world.storage(Camera).clearColor.set(camera, 0);
    attachTexture(world, camera, { width: 64, height: 64 });

    const materials = world.resource(Materials);
    const red = materials.add(StandardMaterial({ baseColor: [1, 0.1, 0.1, 0.35], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const blue = materials.add(StandardMaterial({ baseColor: [0.1, 0.2, 1, 0.35], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    for (let i = 0; i < ITEM_COUNT; i++) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [-1.05 + (i % 4) * 0.7, -0.45 + Math.floor(i / 4) * 0.9, -0.2 - i * 0.04, 0],
            scale: [0.55, 0.55, 0.08, 0],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, i % 2 ? red : blue);
    }

    const label = world.create();
    world.add(label, Transform, { translation: [-1.2, 1.05, -0.4, 0] });
    world.add(label, Text, {
        content: internText(world, "isolation"),
        font,
        fontSize: 0.22,
        color: 0xffffff,
    });

    const line = world.create();
    world.add(line, Transform, { translation: [-1.15, -1.1, -0.35, 0] });
    world.add(line, Line, { offset: [2.3, 0, 0, 0], thickness: 2, color: 0xffcc55, opacity: 0.7 });

    return {
        step: () => world.step(FIXED_DT),
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
