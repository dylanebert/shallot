import { expect, test } from "bun:test";
import {
    ColorGrading,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Transform,
    World,
} from "@dylanebert/shallot";
import { Fog, Sky } from "@dylanebert/shallot/extras";
import { atmosphere, Clock, CycleScene, CycleSystem } from "./index";

test("noon has the highest and brightest sun", () => {
    const noon = atmosphere(12);
    for (let hour = 0; hour < 24; hour += 0.25) {
        expect(atmosphere(hour).direction[1]).toBeGreaterThanOrEqual(noon.direction[1]);
        expect(atmosphere(hour).illuminance).toBeLessThanOrEqual(noon.illuminance);
    }
});

test("the sun is lit whenever it is above the horizon", () => {
    for (let hour = 0; hour < 24; hour += 0.001) {
        const values = atmosphere(hour);
        if (values.direction[1] < 0) expect(values.illuminance).toBeGreaterThan(0);
        else expect(values.illuminance).toBe(0);
    }
});

test("dusk is on the horizon, warmer and hazier than noon", () => {
    const dusk = atmosphere(18),
        noon = atmosphere(12);
    expect(Math.abs(dusk.direction[1])).toBeLessThan(0.001);
    expect(dusk.sunColor & 255).toBeLessThan(noon.sunColor & 255);
    expect(dusk.fogDensity).toBeGreaterThan(noon.fogDensity);
});

test("midnight has no sunlight, visible stars and a darker sky", () => {
    const night = atmosphere(0),
        noon = atmosphere(12);
    expect(night.illuminance).toBe(0);
    expect(night.starIntensity).toBeGreaterThan(0);
    for (const shift of [0, 8, 16])
        expect((night.zenith >> shift) & 255).toBeLessThan((noon.zenith >> shift) & 255);
});

test("values are continuous around the day including the midnight wrap", () => {
    const colors = [
        "sunColor",
        "moonColor",
        "ambientColor",
        "zenith",
        "horizon",
        "hazeColor",
        "fogColor",
    ];
    for (let hour = 0; hour <= 24; hour += 0.05) {
        const a = atmosphere(hour - 0.00001),
            b = atmosphere(hour + 0.00001);
        for (const key of Object.keys(a) as (keyof typeof a)[]) {
            if (key === "direction" || key === "moonDirection") {
                for (let i = 0; i < 3; i++)
                    expect(Math.abs(a[key][i] - b[key][i])).toBeLessThan(0.0001);
            } else if (colors.includes(key)) {
                for (const shift of [0, 8, 16])
                    expect(
                        Math.abs(
                            (((a[key] as number) >> shift) & 255) -
                                (((b[key] as number) >> shift) & 255),
                        ),
                    ).toBeLessThanOrEqual(1);
            } else
                expect(Math.abs((a[key] as number) - (b[key] as number))).toBeLessThan(
                    key === "illuminance" ? 0.2 : 0.02,
                );
        }
    }
});

test("the hour resource drives every atmosphere value written by the system", () => {
    const world = new World();
    const scene = world.resource(CycleScene);
    scene.sun = world.create();
    world.add(scene.sun, Transform);
    world.add(scene.sun, DirectionalLight);
    scene.moon = world.create();
    world.add(scene.moon, Transform);
    world.add(scene.moon, DirectionalLight);
    scene.sky = world.create();
    world.add(scene.sky, Sky);
    scene.fog = world.create();
    world.add(scene.fog, Fog);
    scene.camera = world.create();
    world.add(scene.camera, ColorGrading);
    for (const hour of [0, 6, 12, 18, 23.99]) {
        world.resource(Clock).hour = hour;
        CycleSystem.update(world);
        const v = atmosphere(hour);
        const rotation = world.storage(Transform).rotation;
        const expected = lookAtRotation(0, 0, 0, ...v.direction);
        const actual = [
            rotation.x.get(scene.sun),
            rotation.y.get(scene.sun),
            rotation.z.get(scene.sun),
            rotation.w.get(scene.sun),
        ];
        expect(
            Math.abs(actual.reduce((sum, value, i) => sum + value * expected[i], 0)),
        ).toBeCloseTo(1, 6);
        expect(world.storage(DirectionalLight).illuminance.get(scene.sun)).toBeCloseTo(
            v.illuminance,
            3,
        );
        expect(world.storage(DirectionalLight).color.get(scene.sun)).toBe(v.sunColor);
        expect(world.storage(DirectionalLight).color.get(scene.moon)).toBe(v.moonColor);
        expect(world.storage(DirectionalLight).illuminance.get(scene.moon)).toBeCloseTo(
            v.moonIlluminance,
            3,
        );
        const moonRotation = lookAtRotation(0, 0, 0, ...v.moonDirection);
        expect(
            Math.abs(
                rotation.x.get(scene.moon) * moonRotation[0] +
                    rotation.y.get(scene.moon) * moonRotation[1] +
                    rotation.z.get(scene.moon) * moonRotation[2] +
                    rotation.w.get(scene.moon) * moonRotation[3],
            ),
        ).toBeCloseTo(1, 6);
        expect(world.resource(GlobalAmbientLight)).toMatchObject({
            color: v.ambientColor,
            brightness: v.ambientBrightness,
        });
        for (const key of ["zenith", "horizon", "hazeColor", "starIntensity"] as const)
            expect(world.storage(Sky)[key].get(scene.sky)).toBeCloseTo(v[key], 5);
        expect(world.storage(Sky).hazeDensity.get(scene.sky)).toBeCloseTo(v.fogDensity, 6);
        expect(world.storage(Fog).color.get(scene.fog)).toBe(v.fogColor);
        expect(world.storage(Fog).density.get(scene.fog)).toBeCloseTo(v.fogDensity, 6);
        expect(world.storage(Fog).scattering.get(scene.fog)).toBeCloseTo(v.fogScattering, 5);
        expect(world.storage(ColorGrading).exposure.get(scene.camera)).toBeCloseTo(v.exposure, 6);
    }
    world.addSystem(CycleSystem);
    world.resource(Clock).hour = 23.999;
    world.step(1 / 60);
    expect(world.resource(Clock).hour).toBeCloseTo((23.999 + 24 / 60 / 60) % 24, 8);
    world.dispose();
});

test("the moon lights every sunless hour and rests at noon; the brightest light is above the horizon", () => {
    expect(atmosphere(12).moonIlluminance).toBe(0);
    for (let hour = 0; hour < 24; hour += 0.001) {
        const v = atmosphere(hour);
        if (v.direction[1] >= 0) expect(v.moonIlluminance).toBeGreaterThan(0);
        const direction = v.illuminance >= v.moonIlluminance ? v.direction : v.moonDirection;
        expect(direction[1]).toBeLessThan(0);
        expect(v.ambientBrightness).toBeGreaterThan(0);
        if (v.starIntensity === 1)
            for (const shift of [0, 8, 16])
                expect((v.horizon >> shift) & 255).toBeGreaterThan((v.zenith >> shift) & 255);
    }
});
