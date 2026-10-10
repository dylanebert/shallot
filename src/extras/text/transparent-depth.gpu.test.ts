import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
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

setDefaultTimeout(CEILING.gpu);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin, LinesPlugin, MeshRenderPlugin, StandardRenderingPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
                "occlusion",
            );
        },
    },
]);

async function frame(world: ReturnType<typeof subjects>[number]["world"], camera: number) {
    world.step(1 / 60);
    world.step(1 / 60);
    return (await captureTexture(world, camera)).rgba;
}

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
        anchor: [0, 0],
    });
    world.storage(Transform).translation.x.set(cube, -10);
    const textVisible = await frame(world, camera);
    expect(textVisible).not.toEqual(baseline);
    world.storage(Transform).translation.x.set(cube, 0);
    const textOccluded = await frame(world, camera);
    expect(textOccluded).toEqual(baseline);

    world.storage(Text).visible.set(label, 0);
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
