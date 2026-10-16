import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_FONT } from "./font";
import { isolationFont } from "./font.fixture";
import { internText, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
let defaultFont = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin],
        setup(world) {
            registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
            defaultFont = registerFont(world, DEFAULT_FONT);
        },
    },
]);

test("a world registering a font draws a font-0 label in the default face", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    attachTexture(world, camera, { width: 128, height: 64 });
    const label = world.create();
    world.add(label, Transform);
    world.add(label, Text, { content: internText(world, "isolation"), fontSize: 1 });
    const frame = async () => {
        world.step(1 / 60);
        return (await captureTexture(world, camera)).rgba;
    };
    const implicit = await frame();
    world.storage(Text).font.set(label, defaultFont);
    const explicit = await frame();
    expect(implicit).toEqual(explicit);
    world.remove(label, Text);
    expect(explicit).not.toEqual(await frame());
});
