import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { Fog, FogPlugin } from "../../extras/fog";
import { Outline, OutlinePlugin } from "../../extras/outline";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    Materials,
    MeshMaterial3d,
    registerBackground,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Mesh3d } from "../mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    PickingPrepass,
    PointLight,
    Spot,
    Tonemapping,
    TonemappingMethod,
    Volumetric,
} from "./index";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [...DEFAULT_PLUGINS, FogPlugin, OutlinePlugin],
    },
]);

test("view targets preserve non-uniform lit background, fog and outline frames for every AA and lane set", async () => {
    const { world } = subjects()[0];
    let renderPasses = 0;
    const device = world.gpu.device;
    const createEncoder = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (descriptor) => {
        const encoder = createEncoder(descriptor);
        const begin = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (descriptor) => {
            renderPasses++;
            return begin(descriptor);
        };
        return encoder;
    };
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
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
    world.add(mesh, Transform);
    const material = world.resource(Materials).register({
        name: "target-proof",
        ...StandardMaterial({ base_color: [1, 0, 1, 1], perceptual_roughness: 1 }),
    });
    world.add(mesh, Mesh3d);
    world.add(mesh, MeshMaterial3d, { material });
    world.add(mesh, Outline, { width: 3, color: [0.1, 1, 0.2, 1] });
    const edgeMesh = world.create();
    world.add(edgeMesh, Transform, {
        translation: [-1.2, -0.55, 0.2, 0],
        rotation: [0.0996005, 0.199201, 0, 0.974884],
        scale: [0.65, 0.65, 0.65, 0],
    });
    world.add(edgeMesh, Mesh3d);
    world.add(edgeMesh, MeshMaterial3d, { material });
    world.add(world.create(), AmbientLight, { intensity: 0.2 });
    const sun = world.create();
    world.add(sun, DirectionalLight, { direction: [-0.4, -0.8, -0.5, 0] });
    world.add(sun, Volumetric);
    const point = world.create();
    world.add(point, Transform, { translation: [1, 1, 2, 0] });
    world.add(point, PointLight, { intensity: 8, range: 10, color: 0xff8844 });
    world.add(point, Volumetric);
    const spot = world.create();
    world.add(spot, Transform, { translation: [-1, 1, 3, 0] });
    world.add(spot, PointLight, { intensity: 12, range: 10, color: 0x4488ff });
    world.add(spot, Spot, { inner: 20, outer: 40 });
    world.add(spot, Volumetric);
    world.add(world.create(), Fog);
    const directory = process.env.SHALLOT_TARGET_FRAMES;
    const frames: Uint8ClampedArray[][] = [[], []];
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
            renderPasses = 0;
            world.step(0);
            expect(renderPasses).toBe(lanes ? 7 : 6);
            const { rgba } = await captureTexture(world, camera);
            frames[aa][lanes] = rgba;
            expect(await world.gpu.device.popErrorScope()).toBeNull();
            const colors = new Set<string>();
            let outlinePixels = 0;
            for (let i = 0; i < rgba.length; i += 4) {
                colors.add(`${rgba[i]},${rgba[i + 1]},${rgba[i + 2]}`);
                if (rgba[i + 1] > rgba[i] && rgba[i + 1] > rgba[i + 2]) outlinePixels++;
            }
            expect(colors.size).toBeGreaterThan(200);
            expect(outlinePixels).toBeGreaterThan(20);
            if (directory) {
                await mkdir(directory, { recursive: true });
                const path = `${directory}/${aa}-${lanes}.rgba`;
                if (process.env.SHALLOT_RECORD_TARGET_FRAMES) await writeFile(path, rgba);
                else expect(Buffer.from(rgba).equals(await readFile(path))).toBe(true);
            }
            console.log(
                `targets AA=${aa} lanes=${lanes}: ${colors.size} distinct RGB values; ${outlinePixels} green outline pixels; ${renderPasses} render passes${directory ? (process.env.SHALLOT_RECORD_TARGET_FRAMES ? "; frame recorded" : "; matches parent bytes") : ""}`,
            );
        }
    }
    for (let lanes = 0; lanes < 4; lanes++) {
        expect(Buffer.from(frames[0][lanes]).equals(Buffer.from(frames[1][lanes]))).toBe(false);
    }
});
