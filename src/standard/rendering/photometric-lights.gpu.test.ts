import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    Exposure,
    GlobalAmbientLight,
    PointLight,
    SpotLight,
    SUN_DISK_EARTH_ANGULAR_SIZE,
    SunDisk,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import {
    DirectionalLightGpu,
    Lighting,
    LightingGpu,
    MAX_DIRECTIONAL_LIGHTS,
    shadowDirectionalLight,
    writeLighting,
} from "./lighting";

setDefaultTimeout(CEILING.gpu);

const config = { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] };
const subjects = gpuApps(import.meta.path, [
    config,
    config,
    config,
    config,
    config,
    config,
    config,
    config,
    config,
]);

function scene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(
        StandardMaterial({
            baseColor: [1, 1, 1, 1],
            metallic: 0,
            perceptualRoughness: 1,
            diffuseWrap: 1,
        }),
    );
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, material);
    return { camera, mesh };
}

async function center(world: World, camera: number): Promise<number[]> {
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    return Array.from(rgba.subarray((16 * 32 + 16) * 4, (16 * 32 + 16) * 4 + 4));
}

test("two directional lights both contribute to one fragment", async () => {
    const { world } = subjects()[0];
    const { camera } = scene(world);
    world.add(camera, AmbientLight, { brightness: 0 });
    const red = world.create();
    world.add(red, Transform);
    world.add(red, DirectionalLight, { color: 0xff0000, illuminance: 3135.6103 });
    const redPixel = await center(world, camera);

    const green = world.create();
    world.add(green, Transform);
    world.add(green, DirectionalLight, { color: 0x00ff00, illuminance: 3135.6103 });
    const combined = await center(world, camera);
    expect(redPixel[0]).toBeGreaterThan(0);
    expect(combined[0]).toBe(redPixel[0]);
    expect(combined[1]).toBeGreaterThan(redPixel[1]);
});

test("a directional light follows the entity's transformed local -Z", async () => {
    const { world } = subjects()[1];
    const { camera } = scene(world);
    world.add(camera, AmbientLight, { brightness: 0 });
    const light = world.create();
    world.add(light, Transform);
    world.add(light, DirectionalLight, { illuminance: 3135.6103 });
    const before = await center(world, camera);

    world.storage(Transform).rotation.set(light, 0, Math.SQRT1_2, 0, Math.SQRT1_2);
    const after = await center(world, camera);
    expect(before[0]).toBeGreaterThan(after[0]);
});

test("the lighting uniform packs ten directionals and warns once on overflow", () => {
    const { world } = subjects()[7];
    for (let i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, DirectionalLight);
    }
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
        writeLighting(world);
        const countAt =
            d.memoryLayoutOf(LightingGpu, (lighting) => lighting.directionalCount).offset / 4;
        expect(new Uint32Array(world.resource(Lighting).staging.buffer)[countAt]).toBe(10);
        const overflow = world.create();
        world.add(overflow, Transform);
        world.add(overflow, DirectionalLight);
        writeLighting(world);
        expect(new Uint32Array(world.resource(Lighting).staging.buffer)[countAt]).toBe(10);
        expect(warn).toHaveBeenCalledTimes(1);
    } finally {
        warn.mockRestore();
    }
});

test("directional lux preserves the former linear-light frame after recalibration", async () => {
    const { world } = subjects()[6];
    const { camera } = scene(world);
    world.add(camera, AmbientLight, { brightness: 0 });
    const sun = world.create();
    world.add(sun, Transform);
    world.add(sun, DirectionalLight, { illuminance: 627.12205 });

    const pixel = await center(world, camera);
    const encoded = Math.round((1.055 * 0.2 ** (1 / 2.4) - 0.055) * 255);
    // One byte covers f32 exposure/BRDF arithmetic and the final 8-bit sRGB quantization.
    expect(Math.abs(pixel[0] - encoded)).toBeLessThanOrEqual(1);
    expect(Math.abs(pixel[1] - encoded)).toBeLessThanOrEqual(1);
    expect(Math.abs(pixel[2] - encoded)).toBeLessThanOrEqual(1);
});

test("point lumens preserve the former linear-light frame after recalibration", async () => {
    const { world } = subjects()[3];
    const { camera } = scene(world);
    world.add(camera, AmbientLight, { brightness: 0 });
    const light = world.create();
    world.add(light, Transform, { translation: [0, 0, 2, 0] });
    world.add(light, PointLight, { intensity: 7880.648, range: 10, radius: 0.1 });

    const pixel = await center(world, camera);
    const distanceSq = 2.25;
    const window = (1 - (distanceSq / 100) ** 2) ** 2;
    const legacyLinear = (0.2 * window) / distanceSq;
    const encoded = Math.round((1.055 * legacyLinear ** (1 / 2.4) - 0.055) * 255);
    expect(pixel[0]).toBeCloseTo(encoded, 0);
    expect(pixel[1]).toBeCloseTo(encoded, 0);
    expect(pixel[2]).toBeCloseTo(encoded, 0);
});

test("spot lumens preserve the former linear-light frame after recalibration", async () => {
    const { world } = subjects()[4];
    const { camera } = scene(world);
    world.add(camera, AmbientLight, { brightness: 0 });
    const light = world.create();
    world.add(light, Transform, { translation: [0, 0, 2, 0] });
    world.add(light, SpotLight, {
        intensity: 7880.648,
        range: 10,
        radius: 0.1,
        innerAngle: 20,
        outerAngle: 30,
    });

    const pixel = await center(world, camera);
    const distanceSq = 2.25;
    const window = (1 - (distanceSq / 100) ** 2) ** 2;
    const legacyLinear = (0.2 * window) / distanceSq;
    const encoded = Math.round((1.055 * legacyLinear ** (1 / 2.4) - 0.055) * 255);
    expect(pixel[0]).toBeCloseTo(encoded, 0);
    expect(pixel[1]).toBeCloseTo(encoded, 0);
    expect(pixel[2]).toBeCloseTo(encoded, 0);
});

test("default light values match Bevy's photometric and shadow defaults", () => {
    const { world } = subjects()[8];
    const ambientEid = world.create();
    const directionalEid = world.create();
    const pointEid = world.create();
    const spotEid = world.create();
    world.add(ambientEid, AmbientLight);
    world.add(directionalEid, DirectionalLight);
    world.add(directionalEid, SunDisk);
    world.add(pointEid, PointLight);
    world.add(spotEid, SpotLight);

    expect(world.resource(GlobalAmbientLight)).toEqual({ color: 0xffffff, brightness: 80 });
    const ambient = world.storage(AmbientLight);
    expect(ambient.color.get(ambientEid)).toBe(0xffffff);
    expect(ambient.brightness.get(ambientEid)).toBe(80);

    const directional = world.storage(DirectionalLight);
    expect(directional.color.get(directionalEid)).toBe(0xffffff);
    expect(directional.illuminance.get(directionalEid)).toBe(10_000);
    expect(directional.shadowMapsEnabled.get(directionalEid)).toBe(0);
    expect(directional.maximumDistance.get(directionalEid)).toBe(150);
    expect(directional.numCascades.get(directionalEid)).toBe(4);
    expect(directional.firstCascadeFarBound.get(directionalEid)).toBe(10);
    expect(directional.overlapProportion.get(directionalEid)).toBeCloseTo(0.2, 6);
    expect(directional.depthBias.get(directionalEid)).toBeCloseTo(0.02, 6);
    expect(directional.shadowNormalBias.get(directionalEid)).toBeCloseTo(1.8, 6);
    const sunDisk = world.storage(SunDisk);
    expect(sunDisk.angularSize.get(directionalEid)).toBeCloseTo(SUN_DISK_EARTH_ANGULAR_SIZE, 7);
    expect(sunDisk.intensity.get(directionalEid)).toBe(1);
    expect(sunDisk.glow.get(directionalEid)).toBe(0);

    const point = world.storage(PointLight);
    expect(point.color.get(pointEid)).toBe(0xffffff);
    expect(point.intensity.get(pointEid)).toBe(1_000_000);
    expect(point.range.get(pointEid)).toBe(20);
    expect(point.radius.get(pointEid)).toBe(0);
    expect(point.shadowMapsEnabled.get(pointEid)).toBe(0);
    expect(point.depthBias.get(pointEid)).toBeCloseTo(0.08, 6);
    expect(point.shadowNormalBias.get(pointEid)).toBeCloseTo(0.6, 6);

    const spot = world.storage(SpotLight);
    expect(spot.color.get(spotEid)).toBe(0xffffff);
    expect(spot.intensity.get(spotEid)).toBe(1_000_000);
    expect(spot.range.get(spotEid)).toBe(20);
    expect(spot.radius.get(spotEid)).toBe(0);
    expect(spot.innerAngle.get(spotEid)).toBe(0);
    expect(spot.outerAngle.get(spotEid)).toBe(45);
    expect(spot.shadowMapsEnabled.get(spotEid)).toBe(0);
    expect(spot.depthBias.get(spotEid)).toBeCloseTo(0.02, 6);
    expect(spot.shadowNormalBias.get(spotEid)).toBeCloseTo(1.8, 6);
});

test("Bevy photometric defaults render a finite, exposed frame", async () => {
    const { world } = subjects()[5];
    const { camera } = scene(world);
    world.storage(Tonemapping).method.set(camera, TonemappingMethod.KhronosPbrNeutral);
    world.add(camera, AmbientLight, { brightness: 80 });
    world.add(camera, Exposure, { ev100: 9.7 });
    const sun = world.create();
    world.add(sun, Transform);
    world.add(sun, DirectionalLight, { illuminance: 10_000 });
    const point = world.create();
    world.add(point, Transform, { translation: [2, 2, 6, 0] });
    world.add(point, PointLight, { intensity: 1_000_000, range: 20 });
    const spot = world.create();
    world.add(spot, Transform, { translation: [-2, 2, 6, 0] });
    world.add(spot, SpotLight, { intensity: 1_000_000, range: 20 });

    const pixel = await center(world, camera);
    expect(pixel[3]).toBe(255);
    expect(Math.max(...pixel.slice(0, 3))).toBeGreaterThan(0);
    expect(Math.max(...pixel.slice(0, 3))).toBeLessThan(255);
});

test("the selected directional shadow caster is brightest, then lowest eid, and warns once", () => {
    const { world } = subjects()[2];
    const makeLight = (illuminance: number) => {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, DirectionalLight, { illuminance, shadowMapsEnabled: 1 });
        return eid;
    };
    const low = makeLight(5000);
    const tie = makeLight(5000);
    const bright = makeLight(8000);
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
        expect(shadowDirectionalLight(world)).toBe(bright);
        world.storage(DirectionalLight).illuminance.set(bright, 5000);
        expect(shadowDirectionalLight(world)).toBe(low);
        writeLighting(world);
        const params = d.memoryLayoutOf(DirectionalLightGpu, (light) => light.params).offset / 4;
        const stride = d.sizeOf(DirectionalLightGpu) / 4;
        const staging = world.resource(Lighting).staging;
        expect([
            staging[params + 1],
            staging[stride + params + 1],
            staging[2 * stride + params + 1],
        ]).toEqual([1, 0, 0]);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(tie).toBeGreaterThan(low);
    } finally {
        warn.mockRestore();
    }
});
