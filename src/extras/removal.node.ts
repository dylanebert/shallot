import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../scripts/gpu.fixture";
import { attachTexture, Camera, captureTexture } from "../core/rendering";
import { ImageBindings } from "../core/rendering/image";
import { Transform } from "../core/transform";
import { StandardRenderer } from "../standard/rendering";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./lines";
import { Sprite, SpritePlugin } from "./sprite";
import { internText, registerFont, Text, TextPlugin } from "./text";
import { isolationFont } from "./text/font.fixture";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [SpritePlugin, TextPlugin, LinesPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

test("removing and re-adding Sprite, Text and Gizmo removes and restores each drawing", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    const image = device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(image);
    device.queue.writeTexture(
        { texture: image },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4 },
        [1, 1, 1],
    );
    world.resource(ImageBindings).images = image;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });
    const entity = world.create();
    world.add(entity, Transform);
    const frame = async () => {
        world.step(1 / 60);
        world.step(1 / 60);
        return (await captureTexture(world, camera)).rgba;
    };
    const empty = await frame();
    const sprite = { size: [2, 2] as const, color: 0xff8000 };
    world.add(entity, Sprite, sprite);
    const spriteFrame = await frame();
    expect(spriteFrame).not.toEqual(empty);
    world.remove(entity, Sprite);
    expect(await frame()).toEqual(empty);
    world.add(entity, Sprite, sprite);
    expect(await frame()).toEqual(spriteFrame);
    world.remove(entity, Sprite);

    const text = { content: internText(world, "i"), font, fontSize: 2 };
    world.add(entity, Text, text);
    const textFrame = await frame();
    expect(textFrame).not.toEqual(empty);
    world.remove(entity, Text);
    expect(await frame()).toEqual(empty);
    world.add(entity, Text, text);
    expect(await frame()).toEqual(textFrame);
    world.remove(entity, Text);

    const asset = new GizmoAsset();
    drawLine(asset, [0, 0, 0], [2, 0, 0], 0xffffff);
    const line = { handle: registerGizmo(world, asset), width: 8 };
    world.add(entity, Gizmo, line);
    const lineFrame = await frame();
    expect(lineFrame).not.toEqual(empty);
    world.remove(entity, Gizmo);
    expect(await frame()).toEqual(empty);
    world.add(entity, Gizmo, line);
    expect(await frame()).toEqual(lineFrame);
});
