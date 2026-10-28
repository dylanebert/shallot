import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
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
import { lookAtRotation, type World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);
const config = { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] };
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
// Patch center just below the lamp, for the camera below.
const LAMP = [320, 118] as const;

// A warm shadowed downward spot in dense forward-scattering fog at 360p, where one froxel covers 4×4 pixels and
// the 0.75 m slice holding the lamp sees its inverse-square core. Uniform jitter along that slice made the lamp's
// cell swing with the sample's depth; equiangular sampling toward the light (HDRP), on the column's center ray in
// two depth strata, holds it to the haze's limit.
let measured: Promise<{ swing: number; error: number }> | undefined;
function measure() {
    measured ??= lampTemporalError();
    return measured;
}
function scene(world: World) {
    world.resource(GlobalAmbientLight).brightness = 0;
    const gray = world.resource(Materials).add(StandardMaterial({ baseColor: [0.4, 0.4, 0.4, 1] }));
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
        world.add(block, MeshMaterial, gray);
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
}
/** A camera at eye height looking down −z. */
function cameraAt(world: World, z: number) {
    const eid = world.create();
    world.add(eid, Transform, {
        translation: [0, 1.65, z, 0],
        rotation: lookAtRotation(0, 1.65, z, 0, 1.65, z - 1),
    });
    world.add(eid, Camera, { fov: 55, clearColor: 0, antialias: 0 });
    world.add(eid, StandardRenderer);
    world.add(eid, Exposure, { ev100: 8 });
    world.add(eid, AmbientLight, { brightness: 0 });
    world.add(eid, DebandDither, { enabled: 0 });
    attachTexture(world, eid, { width: WIDTH, height: HEIGHT });
    return eid;
}
const stillScene = gpuScene(import.meta.path, config, (world) => {
    scene(world);
    return cameraAt(world, 9);
});
const walkingScene = gpuScene(import.meta.path, config, (world) => {
    scene(world);
    return cameraAt(world, WALK.start);
});

async function lampTemporalError() {
    const { world, scene: camera } = stillScene();
    world.tick();
    world.step(1 / 60);
    for (let i = 0; i < 32; i++) {
        world.step(0);
        await world.gpu.sync();
    }
    const values: number[] = [];
    for (let i = 0; i < 16; i++) {
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        values.push(patch(rgba, LAMP));
    }
    // The converged reference: the equal linear average of all 16 phases, each from fresh history.
    let reference = 0;
    for (let i = 0; i < 16; i++) {
        // Opt-out discards history. A three-frame stride visits all 16 Halton phases.
        world.add(camera, NoFog);
        world.step(0);
        world.step(0);
        world.remove(camera, NoFog);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        reference += patch(rgba, LAMP) / 16;
    }
    const swing = Math.max(...values) - Math.min(...values);
    const error = Math.max(...values.map((value) => Math.abs(value - reference)));
    console.log("spot lamp temporal error", { swing, error, reference });
    world.destroy(camera);
    return { swing, error };
}

/** Pixel center of a world point for {@link cameraAt}'s camera at depth z. */
function project(z: number, [x, y, pz]: readonly [number, number, number]) {
    const tangent = Math.tan((55 * Math.PI) / 360);
    const distance = z - pz;
    return [
        Math.round(((x / distance / (tangent * (WIDTH / HEIGHT)) + 1) * WIDTH) / 2),
        Math.round(((1 - (y - 1.65) / distance / tangent) * HEIGHT) / 2),
    ] as const;
}

// A walk at 1.5 m/s toward the lamp from 10 m, where one slice is 0.79 m deep. Each sampled frame's patch just
// below the lamp, a world-fixed point, is compared with the converged still reference at the same pose.
const WALK = { speed: 1.5, start: 10, frames: 48, every: 12 } as const;
let walked: Promise<number> | undefined;
async function walkError() {
    const { world, scene: camera } = walkingScene();
    const at = (frame: number) => WALK.start - (WALK.speed / 60) * frame;
    const place = (frame: number) => {
        world.storage(Transform).translation.set(camera, 0, 1.65, at(frame), 0);
        world.tick();
        world.step(0);
    };
    for (let i = 0; i < 32; i++) {
        place(0);
        await world.gpu.sync();
    }
    const point = [0, 3.2, 0] as const;
    const moving = new Map<number, number>();
    for (let frame = 1; frame <= WALK.frames; frame++) {
        place(frame);
        if (frame % WALK.every !== 0) continue;
        const { rgba } = await captureTexture(world, camera);
        moving.set(frame, patch(rgba, project(at(frame), point)));
    }
    let error = 0;
    const errors: number[] = [];
    for (const [frame, value] of moving) {
        let reference = 0;
        for (let i = 0; i < 16; i++) {
            world.add(camera, NoFog);
            place(frame);
            place(frame);
            world.remove(camera, NoFog);
            place(frame);
            const { rgba } = await captureTexture(world, camera);
            reference += patch(rgba, project(at(frame), point)) / 16;
        }
        errors.push(value - reference);
        error = Math.max(error, Math.abs(value - reference));
    }
    console.log("spot lamp walk error", { error, errors });
    world.destroy(camera);
    return error;
}

test("a world-fixed patch below the lamp holds the still reference while walking toward it", async () => {
    walked ??= walkError();
    expect(await walked).toBeLessThan(0.005);
});

test("a still patch at a shadowed spot lamp at 360p stays calm across 16 consecutive phases", async () => {
    expect((await measure()).swing).toBeLessThan(0.001);
});

test("every converged frame at the lamp agrees with the equal 16-phase linear average", async () => {
    expect((await measure()).error).toBeLessThan(0.005);
});
