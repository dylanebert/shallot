import {
    Camera,
    CameraBackground,
    ColorGrading,
    DirectionalLight,
    Exposure,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    SunDisk,
    Tonemapping,
    TonemappingMethod,
    Transform,
    VolumetricLight,
    type World,
} from "@dylanebert/shallot";
import { Fog, FogPlugin, Orbit, OrbitPlugin, Sky, SkyPlugin } from "@dylanebert/shallot/extras";
import { Backgrounds } from "@dylanebert/shallot/standard/rendering";

/** Hour in [0,24); one real minute is one day. Scrubbing replaces the current hour. */
export const Clock: Resource<{ hour: number }> = { create: () => ({ hour: 10 }) };

function blend(a: number, b: number, t: number): number {
    let color = 0;
    for (let shift = 16; shift >= 0; shift -= 8)
        color |= Math.round(((a >> shift) & 255) * (1 - t) + ((b >> shift) & 255) * t) << shift;
    return color;
}

interface Atmosphere {
    direction: [number, number, number];
    moonDirection: [number, number, number];
    moonColor: number;
    moonIlluminance: number;
    sunColor: number;
    illuminance: number;
    ambientColor: number;
    ambientBrightness: number;
    zenith: number;
    horizon: number;
    hazeColor: number;
    starIntensity: number;
    fogColor: number;
    fogDensity: number;
    fogScattering: number;
    exposure: number;
}

/** Periodic stylized atmosphere; direction is the sun's light-travel direction. */
export function atmosphere(
    hour: number,
    out: Atmosphere = {
        direction: [0, 0, 0],
        moonDirection: [0, 0, 0],
        moonColor: 0,
        moonIlluminance: 0,
        sunColor: 0,
        illuminance: 0,
        ambientColor: 0,
        ambientBrightness: 0,
        zenith: 0,
        horizon: 0,
        hazeColor: 0,
        starIntensity: 0,
        fogColor: 0,
        fogDensity: 0,
        fogScattering: 0,
        exposure: 0,
    },
): Atmosphere {
    const angle = ((hour - 6) * Math.PI) / 12;
    const elevation = Math.sin(angle);
    const day = Math.max(0, elevation);
    const daylight = Math.max(0, Math.min(1, (elevation + 0.18) / 0.5));
    const dusk = Math.exp(-((elevation / 0.25) ** 2));
    out.direction[0] = -Math.cos(angle);
    out.direction[1] = -elevation;
    out.direction[2] = -0.25;
    out.moonDirection[0] = Math.cos(angle);
    out.moonDirection[1] = elevation - 0.12;
    out.moonDirection[2] = 0.25;
    out.moonColor = 0xb8ceff;
    out.moonIlluminance = 650 * Math.max(0, Math.min(1, (0.12 - elevation) / 0.4));
    out.sunColor = blend(0xff9b52, 0xffe8cc, day);
    out.illuminance = (2800 * day) / (0.15 + 0.85 * day);
    out.ambientColor = 0xd8e4ff;
    out.ambientBrightness = 180 + 340 * daylight;
    out.zenith = blend(0x080e26, 0x719aca, daylight);
    out.horizon = blend(0x182039, blend(0xbcc5d4, 0xe9a175, dusk), daylight);
    out.hazeColor = blend(0x182039, blend(0xbcc5d4, 0xe9a175, dusk), daylight);
    out.starIntensity = 1 - daylight;
    out.fogColor = blend(0x182039, blend(0xbcc5d4, 0xe9a175, dusk), daylight);
    out.fogDensity = 0.003 + 0.012 * dusk;
    out.fogScattering = 0.3 + (4.7 * day) / (0.02 + day);
    out.exposure = -0.2 * (1 - daylight);
    return out;
}

const AtmosphereValues = { create: () => atmosphere(0) };

export const CycleScene: Resource<{
    sun: number;
    moon: number;
    camera: number;
    sky: number;
    fog: number;
}> = {
    create: () => ({ sun: -1, moon: -1, camera: -1, sky: -1, fog: -1 }),
};

function makeScene(world: World): void {
    const scene = world.resource(CycleScene);
    scene.sun = world.create();
    world.add(scene.sun, Transform);
    world.add(scene.sun, DirectionalLight, { shadowMapsEnabled: 1 });
    world.add(scene.sun, SunDisk, { angularSize: 0.025, glow: 0.8 });
    world.add(scene.sun, VolumetricLight);
    scene.moon = world.create();
    world.add(scene.moon, Transform);
    world.add(scene.moon, DirectionalLight, { shadowMapsEnabled: 1 });
    world.add(scene.moon, SunDisk, { angularSize: 0.018, glow: 0.25 });
    scene.camera = world.create();
    world.add(scene.camera, Camera, { clearColor: 0x191816 });
    world.add(scene.camera, CameraBackground, { name: world.resource(Backgrounds).id("sky") });
    world.add(scene.camera, StandardRenderer);
    world.add(scene.camera, Exposure, { ev100: 9.7 });
    world.add(scene.camera, Tonemapping, { method: TonemappingMethod.AgX });
    world.add(scene.camera, ColorGrading);
    const yaw = -1.3,
        pitch = 0.18,
        distance = 12;
    world.add(scene.camera, Orbit, { distance, yaw, pitch });
    const x = distance * Math.cos(pitch) * Math.sin(yaw);
    const y = distance * Math.sin(pitch);
    const z = distance * Math.cos(pitch) * Math.cos(yaw);
    world.add(scene.camera, Transform, {
        translation: [x, y, z],
        rotation: lookAtRotation(x, y, z, 0, 0, 0),
    });
    scene.sky = world.create();
    world.add(scene.sky, Sky, { cloudCoverage: 0.12, cloudDensity: 0.3, starAmount: 0.7 });
    scene.fog = world.create();
    world.add(scene.fog, Fog, {
        heightFalloff: 0.12,
        scattering: 0.3,
        anisotropy: 0.65,
        steps: 16,
    });

    const materials = world.resource(Materials);
    const accents = [
        [0.72, 0.4, 0.05, 1],
        [0.04, 0.43, 0.53, 1],
        [0.658, 0.301, 0.117, 1],
    ] as const;
    const neutral = materials.add(
        StandardMaterial({
            baseColor: [0.085, 0.08, 0.072, 1],
            perceptualRoughness: 0.82,
        }),
    );
    const shapes: { position: [number, number, number]; scale: [number, number, number] }[] = [
        { position: [0, 1, 0], scale: [1, 2, 1] },
        { position: [-1, 0.6, -2.8], scale: [1.2, 1.2, 1.2] },
        { position: [1, 0.4, 2.8], scale: [1.4, 0.8, 1.4] },
    ];
    for (const [index, shape] of shapes.entries()) {
        const eid = world.create();
        world.add(eid, MeshInstance);
        world.add(
            eid,
            MeshMaterial,
            materials.add(
                StandardMaterial({ baseColor: accents[index]!, perceptualRoughness: 0.38 }),
            ),
        );
        world.add(eid, Transform, { translation: shape.position, scale: shape.scale });
    }
    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(ground, MeshMaterial, neutral);
    world.add(ground, Transform, { translation: [0, -0.1, 0], scale: [200, 0.2, 200] });
    CycleSystem.update(world);
}

function pointLight(world: World, eid: number, direction: [number, number, number]): void {
    const [dx, dy, dz] = direction;
    const yaw = Math.atan2(-dx, -dz) / 2;
    const pitch = Math.atan2(dy, Math.hypot(dx, dz)) / 2;
    world
        .storage(Transform)
        .rotation.set(
            eid,
            Math.sin(pitch) * Math.cos(yaw),
            Math.cos(pitch) * Math.sin(yaw),
            -Math.sin(pitch) * Math.sin(yaw),
            Math.cos(pitch) * Math.cos(yaw),
        );
}

export const CycleSystem = {
    name: "Cycle/Atmosphere",
    update(world: World): void {
        const clock = world.resource(Clock);
        clock.hour = (clock.hour + (world.time.deltaTime * 24) / 60) % 24;
        const v = atmosphere(clock.hour, world.resource(AtmosphereValues));
        const s = world.resource(CycleScene);
        const lights = world.storage(DirectionalLight);
        pointLight(world, s.sun, v.direction);
        pointLight(world, s.moon, v.moonDirection);
        lights.color.set(s.sun, v.sunColor);
        lights.illuminance.set(s.sun, v.illuminance);
        lights.color.set(s.moon, v.moonColor);
        lights.illuminance.set(s.moon, v.moonIlluminance);
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = v.ambientColor;
        ambient.brightness = v.ambientBrightness;
        const sky = world.storage(Sky);
        sky.zenith.set(s.sky, v.zenith);
        sky.horizon.set(s.sky, v.horizon);
        sky.hazeColor.set(s.sky, v.hazeColor);
        sky.hazeDensity.set(s.sky, v.fogDensity);
        sky.starIntensity.set(s.sky, v.starIntensity);
        const fog = world.storage(Fog);
        fog.color.set(s.fog, v.fogColor);
        fog.density.set(s.fog, v.fogDensity);
        fog.scattering.set(s.fog, v.fogScattering);
        world.storage(ColorGrading).exposure.set(s.camera, v.exposure);
    },
};

export const CyclePlugin: Plugin = {
    name: "Cycle",
    dependencies: [OrbitPlugin, SkyPlugin, FogPlugin],
    initialize: makeScene,
    systems: [CycleSystem],
};
