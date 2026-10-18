import { afterEach, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DebandDither,
    detachView,
    PointLight,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const entities: number[] = [];
function entity() {
    const eid = subjects()[0].world.create();
    entities.push(eid);
    return eid;
}
afterEach(() => {
    for (const eid of entities) subjects()[0].world.destroy(eid);
    entities.length = 0;
});
function camera(size = 32, x = 0) {
    const { world } = subjects()[0];
    const eid = entity();
    world.add(eid, Transform, { translation: [x, 0, 5, 0] });
    world.add(eid, Camera, {
        mode: CameraMode.Orthographic,
        size: 6,
        near: 0.1,
        far: 10,
        clearColor: 0,
        antialias: 0,
    });
    world.add(eid, StandardRenderer);
    world.add(eid, DebandDither, { enabled: 0 });
    world.add(eid, AmbientLight, { brightness: 0 });
    attachTexture(world, eid, { width: size, height: size });
    return eid;
}
function scene() {
    const { world } = subjects()[0];
    world.add(entity(), Fog, { color: 0, density: 0.15, lightIntensity: 3, viewDistance: 10 });
    const light = entity();
    world.add(light, Transform, { translation: [-1, 0, 2, 0] });
    world.add(light, PointLight, { intensity: 100000, range: 3 });
    world.add(light, VolumetricLight);
    return light;
}
function frame() {
    const { world } = subjects()[0];
    world.tick();
    world.step(0);
}
async function image(eid: number) {
    return (await captureTexture(subjects()[0].world, eid)).rgba;
}

test("first frames after opt-out, resize and camera replacement equal an unblended fresh view", async () => {
    const { world } = subjects()[0];
    const light = scene();
    let old = camera();
    frame();
    for (let i = 0; i < 32; i++) world.step(0);
    world.storage(Transform).translation.set(light, 1, 0, 2, 0);
    const current = camera();
    frame();
    expect(await image(old)).not.toEqual(await image(current));
    world.add(old, NoFog);
    frame();
    world.remove(old, NoFog);
    const fresh = camera();
    frame();
    expect(await image(old)).toEqual(await image(fresh));
    detachView(world, old);
    attachTexture(world, old, { width: 24, height: 24 });
    const resized = camera(24);
    frame();
    expect(await image(old)).toEqual(await image(resized));
    world.destroy(old);
    entities.splice(entities.indexOf(old), 1);
    old = camera();
    const replacement = camera();
    frame();
    expect(await image(old)).toEqual(await image(replacement));
});

test("history outside the previous frustum takes the current sample alone", async () => {
    const { world } = subjects()[0];
    const light = scene();
    const old = camera();
    frame();
    for (let i = 0; i < 32; i++) world.step(0);
    world.storage(Transform).translation.set(old, 100, 0, 5, 0);
    world.storage(Transform).translation.set(light, 100, 0, 2, 0);
    const fresh = camera(32, 100);
    frame();
    expect(await image(old)).toEqual(await image(fresh));
    expect((await image(old)).some((value, i) => i % 4 !== 3 && value > 10)).toBe(true);
});
