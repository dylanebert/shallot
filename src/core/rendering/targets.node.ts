import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { Fog, FogPlugin } from "../../extras/fog";
import { Outline, OutlinePlugin } from "../../extras/outline";
import {
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    PartPlugin,
    registerBackground,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { Glaze, GlazePlugin } from "../../transitional/glaze";
import { MeshInstance } from "../mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    PickingPrepass,
    RenderingPlugin,
} from "./index";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            RenderingPlugin,
            StandardRenderingPlugin,
            PartPlugin,
            FogPlugin,
            OutlinePlugin,
            GlazePlugin,
        ],
    },
]);

test("view targets preserve non-uniform lit background, fog and outline frames for every AA and lane set", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 4, clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    world.add(camera, Glaze);
    const layout = backgroundLayout({});
    const id = registerBackground(world, {
        name: "target-proof",
        layout,
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return d.vec3f(0.12, 0.2, 0.35);
        }),
    });
    world.add(camera, CameraBackground, { name: id });
    attachTexture(world, camera, { width: 64, height: 64 });
    const mesh = world.create();
    world.add(mesh, Transform, { translation: [0.2, 0, 1, 0] });
    world.add(mesh, MeshInstance);
    world.add(mesh, Outline, { width: 3, color: [0.1, 1, 0.2, 1] });
    world.add(world.create(), AmbientLight, { intensity: 0.4 });
    world.add(world.create(), DirectionalLight);
    world.add(world.create(), Fog, { density: 0.1, jitter: 0 });
    const directory = process.env.SHALLOT_TARGET_FRAMES;
    for (const aa of [0, 1]) {
        world.storage(Camera).antialias.set(camera, aa);
        for (const lanes of [0, 1, 2, 3]) {
            for (const [bit, marker] of [
                [1, DepthPrepass],
                [2, PickingPrepass],
            ] as const) {
                if (lanes & bit) {
                    if (!world.has(camera, marker)) world.add(camera, marker);
                } else world.remove(camera, marker);
            }
            world.gpu.device.pushErrorScope("validation");
            world.step(0);
            world.step(0);
            const { rgba } = await captureTexture(world, camera);
            expect(await world.gpu.device.popErrorScope()).toBeNull();
            const colors = new Set<string>();
            for (let i = 0; i < rgba.length; i += 4)
                colors.add(`${rgba[i]},${rgba[i + 1]},${rgba[i + 2]}`);
            expect(colors.size).toBeGreaterThan(1);
            if (directory) {
                await mkdir(directory, { recursive: true });
                const path = `${directory}/${aa}-${lanes}.rgba`;
                if (process.env.SHALLOT_RECORD_TARGET_FRAMES) await writeFile(path, rgba);
                else expect(Buffer.from(rgba).equals(await readFile(path))).toBe(true);
            }
            console.log(
                `targets AA=${aa} lanes=${lanes}: ${colors.size} distinct RGB values${directory ? (process.env.SHALLOT_RECORD_TARGET_FRAMES ? "; frame recorded" : "; matches parent bytes") : ""}`,
            );
        }
    }
});
