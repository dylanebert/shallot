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
import { Fog, FogPlugin, NoFog } from ".";

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
function patch(rgba: Uint8ClampedArray, [cx, cy]: readonly [number, number]) {
    let sum = 0;
    for (let y = cy - 2; y <= cy + 2; y++) {
        for (let x = cx - 2; x <= cx + 2; x++) {
            const at = (y * WIDTH + x) * 4;
            sum +=
                0.2126 * linear(rgba[at]!) +
                0.7152 * linear(rgba[at + 1]!) +
                0.0722 * linear(rgba[at + 2]!);
        }
    }
    return sum / 25;
}
// Patch centres of the camera below: just below the lamp, mid-cone, and in the cone's penumbra.
const PATCHES = { lamp: [320, 118], core: [320, 167], edge: [247, 182] } as const;
type Patch = keyof typeof PATCHES;

// A warm shadowed downward spot in dense forward-scattering fog at 360p, where one froxel covers 4×4 pixels and
// the 0.75 m slice holding the lamp sees its inverse-square core. Uniform jitter along that slice made the lamp's
// cell swing with the sample's depth; equiangular sampling toward the light (HDRP) holds it still.
let measured: Promise<{ swing: Record<Patch, number>; error: Record<Patch, number> }> | undefined;
function measure() {
    measured ??= lampTemporalError();
    return measured;
}
async function lampTemporalError() {
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
    for (const [translation, scale] of [
        [
            [0, -0.25, 0],
            [20, 0.5, 40],
        ],
        [
            [1, 0.8, 0],
            [0.35, 1.6, 0.35],
        ],
    ] as const) {
        const block = world.create();
        world.add(block, Transform, { translation, scale });
        world.add(block, MeshInstance);
        world.add(block, MeshMaterial, grey);
    }
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
    world.tick();
    world.step(1 / 60);
    for (let i = 0; i < 64; i++) {
        world.step(0);
        await world.gpu.sync();
    }
    const names = Object.keys(PATCHES) as Patch[];
    const frames: Record<Patch, number>[] = [];
    for (let i = 0; i < 16; i++) {
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        const frame = {} as Record<Patch, number>;
        for (const name of names) frame[name] = patch(rgba, PATCHES[name]);
        frames.push(frame);
    }
    // The converged reference: the equal linear average of all 16 phases, each from fresh history.
    const reference = { lamp: 0, core: 0, edge: 0 };
    for (let i = 0; i < 16; i++) {
        // Opt-out discards history. A three-frame stride visits all 16 Halton phases.
        world.add(camera, NoFog);
        world.step(0);
        world.step(0);
        world.remove(camera, NoFog);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        for (const name of names) reference[name] += patch(rgba, PATCHES[name]) / 16;
    }
    const swing = {} as Record<Patch, number>;
    const error = {} as Record<Patch, number>;
    for (const name of names) {
        const values = frames.map((frame) => frame[name]);
        swing[name] = Math.max(...values) - Math.min(...values);
        error[name] = Math.max(...values.map((value) => Math.abs(value - reference[name])));
    }
    console.log("spot lamp temporal error", { swing, error, reference });
    return { swing, error };
}

test("a still patch at a shadowed spot lamp at 360p stays calm across 16 consecutive phases", async () => {
    expect((await measure()).swing.lamp).toBeLessThan(0.02);
});

test("every converged frame at the lamp agrees with the equal 16-phase linear average", async () => {
    expect((await measure()).error.lamp).toBeLessThan(0.01);
});

test("still patches in the cone's core and penumbra stay calm", async () => {
    const { swing } = await measure();
    expect(swing.core).toBeLessThan(0.005);
    expect(swing.edge).toBeLessThan(0.005);
});
