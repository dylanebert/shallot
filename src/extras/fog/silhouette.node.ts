import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DebandDither,
    Exposure,
    GlobalAmbientLight,
    SpotLight,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const WIDTH = 640;
const HEIGHT = 360;
const linear = (value: number) => {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};

// lamp.node.ts's spot scene. The pillar's top edge crosses froxel columns whose centre depth lands on the pillar
// while the far floor shows above it. Light repeated from in front of the pillar through every cell behind it lit
// the haze above the edge up to 0.0034 brighter than the same air without the pillar.
test("the haze above a pillar's top edge matches the same air without the pillar", async () => {
    const { world } = subjects()[0]!;
    world.resource(GlobalAmbientLight).brightness = 0;
    const camera = world.create();
    world.add(camera, Transform, {
        translation: [0, 1.65, 9, 0],
        rotation: lookAtRotation(0, 1.65, 9, 0, 1.65, 8),
    });
    world.add(camera, Camera, { fov: 55, clearColor: 0, antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Exposure, { ev100: 8 });
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: WIDTH, height: HEIGHT });
    const grey = world.resource(Materials).add(StandardMaterial({ baseColor: [0.4, 0.4, 0.4, 1] }));
    const floor = world.create();
    world.add(floor, Transform, { translation: [0, -0.25, 0], scale: [20, 0.5, 40] });
    world.add(floor, MeshInstance);
    world.add(floor, MeshMaterial, grey);
    const pillar = world.create();
    world.add(pillar, Transform, { translation: [1, 0.8, 0], scale: [0.35, 1.6, 0.35] });
    world.add(pillar, MeshInstance);
    world.add(pillar, MeshMaterial, grey);
    const lamp = world.create();
    world.add(lamp, Transform, {
        translation: [0, 3.35, 0, 0],
        rotation: lookAtRotation(0, 3.35, 0, 0, 0, 0, 0, 0, -1),
    });
    world.add(lamp, SpotLight, {
        color: 0xffc58a,
        intensity: 45000,
        range: 7,
        innerAngle: 0.5,
        outerAngle: 1.1,
        shadowMapsEnabled: 1,
    });
    world.add(lamp, VolumetricLight);
    world.add(world.create(), Fog, {
        density: 0.07,
        color: 0,
        brightness: 0,
        absorption: 0,
        scattering: 1,
        scatteringAsymmetry: 0.6,
        lightIntensity: 2 / Math.PI,
        heightFalloff: 0,
    });
    // The pillar's top edge is at row 183; its columns span x 350 to 363.
    const band = async () => {
        world.tick();
        for (let i = 0; i < 48; i++) world.step(0);
        const { rgba } = await captureTexture(world, camera);
        const values: number[] = [];
        for (let y = 170; y <= 182; y++) {
            for (let x = 352; x <= 362; x++) {
                const at = (y * WIDTH + x) * 4;
                values.push(
                    0.2126 * linear(rgba[at]!) +
                        0.7152 * linear(rgba[at + 1]!) +
                        0.0722 * linear(rgba[at + 2]!),
                );
            }
        }
        return values;
    };
    const withPillar = await band();
    world.destroy(pillar);
    const without = await band();
    const excess = Math.max(...withPillar.map((value, i) => value - without[i]!));
    console.log("silhouette band excess", { excess });
    expect(excess).toBeLessThan(0.0005);
});
