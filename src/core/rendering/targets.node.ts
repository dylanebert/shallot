import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { lookAtRotation } from "../../engine";
import { Fog, FogPlugin, NoFog } from "../../extras/fog";
import { Outline, OutlinePlugin } from "../../extras/outline";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    Materials,
    MeshMaterial,
    registerBackground,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { MeshInstance } from "../mesh";
import { Transform } from "../transform";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    PointLight,
    SpotLight,
    Tonemapping,
    TonemappingMethod,
    VolumetricLight,
} from "./index";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [...DEFAULT_PLUGINS, FogPlugin, OutlinePlugin],
    },
]);

test("view targets preserve non-uniform lit background, fog and outline frames for every AA and explicit depth request", async () => {
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
    world.add(camera, AmbientLight, { brightness: 199.61915 });
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
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 0, 1, 1], perceptualRoughness: 1 }));
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, material);
    world.add(mesh, Outline, { width: 3, color: [0.1, 1, 0.2, 1] });
    const edgeMesh = world.create();
    world.add(edgeMesh, Transform, {
        translation: [-1.2, -0.55, 0.2, 0],
        rotation: [0.0996005, 0.199201, 0, 0.974884],
        scale: [0.65, 0.65, 0.65, 0],
    });
    world.add(edgeMesh, MeshInstance);
    world.add(edgeMesh, MeshMaterial, material);
    const sun = world.create();
    const sunRotation = lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5);
    world.add(sun, Transform, {
        rotation: [sunRotation.x, sunRotation.y, sunRotation.z, sunRotation.w],
    });
    world.add(sun, DirectionalLight, { illuminance: 4703.4155 });
    world.add(sun, VolumetricLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const fill = world.create();
    world.add(fill, Transform);
    world.add(fill, DirectionalLight, { color: 0x80c0ff, illuminance: 2500 });
    const point = world.create();
    world.add(point, Transform, { translation: [1, 1, 2, 0] });
    world.add(point, PointLight, { intensity: 315225.92, range: 10, color: 0xff8844 });
    world.add(point, VolumetricLight);
    world.storage(PointLight).shadowMapsEnabled.set(point, 1);
    const spot = world.create();
    world.add(spot, Transform, { translation: [-1, 1, 3, 0] });
    world.add(spot, SpotLight, {
        intensity: 472838.89,
        range: 10,
        color: 0x4488ff,
        innerAngle: 20,
        outerAngle: 40,
        shadowMapsEnabled: 1,
    });
    world.add(spot, VolumetricLight);
    world.add(world.create(), Fog);
    const directory = process.env.SHALLOT_TARGET_FRAMES;
    const frames: Uint8ClampedArray[][] = [[], []];
    for (const aa of [0, 1]) {
        world.storage(Camera).antialias.set(camera, aa);
        for (const depth of [0, 1]) {
            if (depth) {
                world.add(camera, DepthPrepass);
                world.remove(camera, NoFog);
            } else {
                world.remove(camera, DepthPrepass);
                world.add(camera, NoFog);
            }
            world.gpu.device.pushErrorScope("validation");
            world.step(0);
            renderPasses = 0;
            world.step(0);
            // Count and populate each add a color-disabled raster pass before the view's shading passes.
            expect(renderPasses).toBe(depth ? 11 : 10);
            const { rgba } = await captureTexture(world, camera);
            frames[aa][depth] = rgba;
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
                const path = `${directory}/${aa}-${depth}.rgba`;
                if (process.env.SHALLOT_RECORD_TARGET_FRAMES) await writeFile(path, rgba);
                else expect(Buffer.from(rgba).equals(await readFile(path))).toBe(true);
            }
            console.log(
                `targets AA=${aa} depth=${depth}: ${colors.size} distinct RGB values; ${outlinePixels} green outline pixels; ${renderPasses} render passes${directory ? (process.env.SHALLOT_RECORD_TARGET_FRAMES ? "; frame recorded" : "; matches parent bytes") : ""}`,
            );
        }
    }
    for (let depth = 0; depth < 2; depth++) {
        expect(Buffer.from(frames[0][depth]).equals(Buffer.from(frames[1][depth]))).toBe(false);
    }

    // Adding VolumetricLight to a second directional changes the fog result without changing either light's
    // photometric contribution to the surface pass.
    world.add(fill, VolumetricLight);
    world.storage(Camera).antialias.set(camera, 0);
    world.step(0);
    world.step(0);
    const withSecondDirectionalShaft = await captureTexture(world, camera);
    expect(Buffer.from(withSecondDirectionalShaft.rgba).equals(Buffer.from(frames[0][1]))).toBe(
        false,
    );
});
