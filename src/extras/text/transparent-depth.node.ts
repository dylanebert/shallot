import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { Anchor, attachTexture, Billboard, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { Line, LinesPlugin } from "../lines";
import { isolationFont } from "./font.fixture";
import { internText, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin, LinesPlugin, MeshRenderPlugin, StandardRenderingPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

async function frame(world: ReturnType<typeof subjects>[number]["world"], camera: number) {
    world.step(1 / 60);
    world.step(1 / 60);
    return (await captureTexture(world, camera)).rgba;
}

for (const pose of [
    {
        name: "Screen from a pitched camera",
        mode: Billboard.Screen,
        eye: [0, -2, 5],
        label: [0, -0.02 / Math.sqrt(29), 0.05 / Math.sqrt(29)],
        anchor: Anchor.BOTTOM_CENTER,
    },
    {
        name: "YLocked from a side-on camera",
        mode: Billboard.YLocked,
        eye: [-5, 0, 0],
        label: [-0.01, 0, 0],
        anchor: Anchor.BOTTOM_LEFT,
    },
] as const)
    test(`a ${pose.name} label in front of a blended plane stays visible`, async () => {
        const { world } = subjects()[0];
        const camera = world.create();
        const rotation = lookAtRotation(pose.eye[0], pose.eye[1], pose.eye[2], 0, 0, 0);
        world.add(camera, Transform, { translation: pose.eye, rotation });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 64, height: 64 });
        const plane = world.create();
        world.add(plane, Transform, { scale: [4, 4, 0.001], rotation });
        world.add(plane, MeshInstance);
        world.add(
            plane,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [1, 1, 1, 1], unlit: true }), {
                    alphaMode: AlphaMode.Blend,
                }),
        );
        const baseline = await frame(world, camera);
        const label = world.create();
        world.add(label, Transform, {
            translation: pose.label,
        });
        world.add(label, Anchor, { offset: pose.anchor });
        world.add(label, Text, {
            content: internText(world, "i"),
            font,
            fontSize: 1,
            color: 0x000000,
            billboard: pose.mode,
        });
        expect(await frame(world, camera)).not.toEqual(baseline);
        world.destroy(label);
        world.destroy(plane);
        world.destroy(camera);
    });

test("Text and Line are both visible alone and occluded by an opaque mesh", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });

    const cube = world.create();
    world.add(cube, Transform, { scale: [5, 5, 0.1, 0] });
    world.add(cube, MeshInstance);
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 0, 0, 1], unlit: true }));
    world.add(cube, MeshMaterial, material);
    const baseline = await frame(world, camera);

    const label = world.create();
    world.add(label, Transform, { translation: [-1.3, -0.25, -0.5, 0] });
    world.add(label, Text, {
        content: internText(world, "isolation"),
        font,
        fontSize: 0.5,
    });
    world.storage(Transform).translation.x.set(cube, -10);
    const textVisible = await frame(world, camera);
    expect(textVisible).not.toEqual(baseline);
    world.storage(Transform).translation.x.set(cube, 0);
    const textOccluded = await frame(world, camera);
    expect(textOccluded).toEqual(baseline);

    world.remove(label, Text);
    const line = world.create();
    world.add(line, Transform, { translation: [-1.3, -0.5, -0.5, 0] });
    world.add(line, Line, { offset: [2.6, 0, 0, 0], thickness: 8, color: 0xffffff });
    world.storage(Transform).translation.x.set(cube, -10);
    const lineVisible = await frame(world, camera);
    expect(lineVisible).not.toEqual(baseline);
    world.storage(Transform).translation.x.set(cube, 0);
    const lineOccluded = await frame(world, camera);
    expect(lineOccluded).toEqual(baseline);
});
