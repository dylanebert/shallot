import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, Billboard, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { isolationFont } from "./font.fixture";
import { internText, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin],
        setup(world) {
            registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

test("an edge-on label covers pixels in Screen mode and none in World mode", async () => {
    const { world } = subjects()[0];
    console.info("text billboard adapter:", world.gpu.adapter);
    const camera = world.create();
    world.add(camera, Transform, {
        translation: [5, 0, 0, 0],
        rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2],
    });
    world.add(camera, Camera);
    attachTexture(world, camera, { width: 64, height: 64 });
    const frame = async () => {
        world.step(1 / 60);
        world.step(1 / 60);
        return (await captureTexture(world, camera)).rgba;
    };
    const baseline = await frame();
    const label = world.create();
    world.add(label, Transform);
    world.add(label, Text, {
        content: internText(world, "i"),
        fontSize: 2,
        anchor: [0.5, 0.5],
    });
    const coverage = (rgba: Uint8ClampedArray) => {
        let pixels = 0;
        for (let i = 0; i < rgba.length; i += 4) {
            if (
                rgba[i] !== baseline[i] ||
                rgba[i + 1] !== baseline[i + 1] ||
                rgba[i + 2] !== baseline[i + 2]
            )
                pixels++;
        }
        return pixels;
    };
    expect(world.storage(Text).billboard.get(label)).toBe(Billboard.World);
    expect(coverage(await frame()), "default World glyph coverage").toBe(0);
    world.storage(Text).billboard.set(label, Billboard.Screen);
    const screenPixels = coverage(await frame());
    world.storage(Text).billboard.set(label, Billboard.World);
    const worldPixels = coverage(await frame());
    console.info("edge-on glyph pixels:", { screen: screenPixels, world: worldPixels });
    expect(screenPixels, "Screen glyph coverage").toBeGreaterThan(0);
    expect(worldPixels, "World glyph coverage").toBe(0);
    world.storage(Text).billboard.set(label, Billboard.YLocked);
    expect(coverage(await frame()), "YLocked glyph coverage").toBeGreaterThan(0);
});
