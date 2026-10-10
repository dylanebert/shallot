import { createApp } from "../../src/engine/app";
import { Camera } from "../../src/core/rendering";
import { Transform } from "../../src/core/transform";
import { StandardRenderer } from "../../src/standard/rendering";
import { Sprite, SpritePlugin } from "../../src/extras/sprite";
import { internText, registerFont, Text, TextPlugin } from "../../src/extras/text";
import { isolationFont } from "../../src/extras/text/font.fixture";

const FIXED_DT = 1 / 60;
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    const app = await createApp({
        defaults: false,
        plugins: [SpritePlugin, TextPlugin],
        setup(world) {
            registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
                "allocation",
            );
        },
    });
    const world = app.world;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);

    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite);

    const label = world.create();
    world.add(label, Transform, { translation: [0, 1, 0, 0] });
    world.add(label, Text, { content: internText(world, "isolation"), fontSize: 0.2 });

    return {
        step: () => world.step(FIXED_DT),
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
