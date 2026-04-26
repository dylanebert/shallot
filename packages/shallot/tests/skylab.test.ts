import { describe, test, expect, beforeEach } from "bun:test";
import { State, DirectionalLight, AmbientLight, Sky, Sun, Stars, Moon, Haze, Clouds } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import {
    Skylab,
    SkylabPlugin,
    sampleGradient,
    directionToElevation,
    directionToAzimuth,
    toDirection,
} from "../src/extras/skylab";
import { RenderPlugin } from "../src/standard/render";

describe("skylab", () => {
    describe("directionToElevation", () => {
        test("straight down sun gives 90 degrees", () => {
            expect(directionToElevation(0, -1, 0)).toBeCloseTo(90, 1);
        });

        test("horizontal sun gives 0 degrees", () => {
            expect(directionToElevation(-1, 0, 0)).toBeCloseTo(0, 1);
        });

        test("straight up sun gives -90 degrees", () => {
            expect(directionToElevation(0, 1, 0)).toBeCloseTo(-90, 1);
        });

        test("45 degree elevation", () => {
            const s = Math.SQRT1_2;
            expect(directionToElevation(0, -s, -s)).toBeCloseTo(45, 1);
        });

        test("negative elevation (sun below horizon)", () => {
            const s = Math.SQRT1_2;
            expect(directionToElevation(0, s, -s)).toBeCloseTo(-45, 1);
        });

        test("unnormalized direction still works", () => {
            expect(directionToElevation(0, -3, 0)).toBeCloseTo(90, 1);
        });

        test("zero vector returns 90", () => {
            expect(directionToElevation(0, 0, 0)).toBe(90);
        });
    });

    describe("directionToAzimuth", () => {
        test("pointing along -Z gives 0 degrees", () => {
            expect(directionToAzimuth(0, 0, -1)).toBeCloseTo(0, 1);
        });

        test("pointing along -X gives 90 degrees", () => {
            expect(directionToAzimuth(-1, 0, 0)).toBeCloseTo(90, 1);
        });

        test("pointing along +Z gives 180 degrees", () => {
            expect(directionToAzimuth(0, 0, 1)).toBeCloseTo(180, 1);
        });

        test("pointing along +X gives 270 degrees", () => {
            expect(directionToAzimuth(1, 0, 0)).toBeCloseTo(270, 1);
        });

        test("azimuth is always 0-360", () => {
            const az = directionToAzimuth(-0.6, -1, -0.8);
            expect(az).toBeGreaterThanOrEqual(0);
            expect(az).toBeLessThan(360);
        });
    });

    describe("toDirection", () => {
        test("zenith (elevation 90) points straight down", () => {
            const [x, y, z] = toDirection(0, 90);
            expect(x).toBeCloseTo(0, 5);
            expect(y).toBeCloseTo(-1, 5);
            expect(z).toBeCloseTo(0, 5);
        });

        test("horizon (elevation 0, azimuth 0) points along -Z", () => {
            const [x, y, z] = toDirection(0, 0);
            expect(x).toBeCloseTo(0, 5);
            expect(y).toBeCloseTo(0, 5);
            expect(z).toBeCloseTo(-1, 5);
        });

        test("roundtrip with directionToElevation and directionToAzimuth", () => {
            const az = 135;
            const el = 30;
            const [x, y, z] = toDirection(az, el);
            expect(directionToElevation(x, y, z)).toBeCloseTo(el, 1);
            expect(directionToAzimuth(x, y, z)).toBeCloseTo(az, 1);
        });
    });

    describe("sampleGradient", () => {
        test("midday (50 degrees) has high sun intensity", () => {
            const out = sampleGradient(50);
            expect(out.sunIntensity).toBeCloseTo(0.81, 1);
        });

        test("deep night has faint moonlight intensity", () => {
            const out = sampleGradient(-90);
            expect(out.sunIntensity).toBeGreaterThan(0);
            expect(out.sunIntensity).toBeLessThan(0.25);
        });

        test("deep night has high star intensity", () => {
            const out = sampleGradient(-90);
            expect(out.starsIntensity).toBeCloseTo(1.0, 1);
        });

        test("midday has zero star intensity", () => {
            const out = sampleGradient(50);
            expect(out.starsIntensity).toBe(0);
        });

        test("deep night has perceptible ambient", () => {
            const out = sampleGradient(-90);
            expect(out.ambientIntensity).toBeGreaterThan(0.1);
            const r = (out.ambientColor >> 16) & 0xff;
            const g = (out.ambientColor >> 8) & 0xff;
            const b = out.ambientColor & 0xff;
            expect(b).toBeGreaterThan(r);
            expect(b).toBeGreaterThan(g);
        });

        test("deep night has moon disk visible", () => {
            const out = sampleGradient(-90);
            expect(out.moonDisk).toBe(1.0);
        });

        test("midday has zero moon disk", () => {
            const out = sampleGradient(50);
            expect(out.moonDisk).toBe(0);
        });

        test("golden hour has zero moon disk", () => {
            const out = sampleGradient(8);
            expect(out.moonDisk).toBe(0);
        });

        test("horizon (0 degrees) has sun disk visible", () => {
            const out = sampleGradient(0);
            expect(out.sunDisk).toBe(0.7);
        });

        test("clamped at -90", () => {
            const a = sampleGradient(-90);
            const b = sampleGradient(-100);
            expect(a.sunIntensity).toBe(b.sunIntensity);
        });

        test("clamped at 90", () => {
            const a = sampleGradient(50);
            const b = sampleGradient(90);
            expect(a.sunIntensity).toBe(b.sunIntensity);
        });

        test("interpolates between stops", () => {
            const out = sampleGradient(10);
            expect(out.sunIntensity).toBeGreaterThan(0.7);
            expect(out.sunIntensity).toBeLessThan(0.9);
        });

        test("returns valid packed colors", () => {
            const out = sampleGradient(30);
            expect(out.sunColor).toBeGreaterThanOrEqual(0);
            expect(out.sunColor).toBeLessThanOrEqual(0xffffff);
            expect(out.zenith).toBeGreaterThanOrEqual(0);
            expect(out.zenith).toBeLessThanOrEqual(0xffffff);
            expect(out.horizon).toBeGreaterThanOrEqual(0);
            expect(out.horizon).toBeLessThanOrEqual(0xffffff);
        });

        test("haze density is always positive", () => {
            for (const el of [-90, -18, 0, 8, 20, 50]) {
                const out = sampleGradient(el);
                expect(out.hazeDensity).toBeGreaterThan(0);
            }
        });
    });

    describe("SkylabSystem", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            state = new State();
            state.register(RenderPlugin);
            state.register(SkylabPlugin);
        });

        test("updates existing atmosphere components without adding new ones", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            state.addComponent(env, Sky);
            state.addComponent(env, Sun);
            state.addComponent(env, Stars);
            state.addComponent(env, Haze);
            state.addComponent(env, Clouds);
            Skylab.azimuth[env] = 220;
            Skylab.elevation[env] = 45;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            const expected = sampleGradient(45);
            expect(Sky.zenith[env]).toBe(expected.zenith);
            expect(Sky.horizon[env]).toBe(expected.horizon);
            expect(Sun.size[env]).toBe(expected.sunDisk);
            expect(Sun.glow[env]).toBeCloseTo(expected.sunGlow, 2);
            expect(Stars.intensity[env]).toBeCloseTo(expected.starsIntensity, 2);
            expect(Haze.density[env]).toBeCloseTo(expected.hazeDensity, 2);
            expect(Haze.color[env]).toBe(expected.hazeColor);
            expect(Clouds.color[env]).toBe(expected.cloudsColor);
        });

        test("does not add atmosphere components that are absent", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 220;
            Skylab.elevation[env] = 45;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            expect(state.hasComponent(env, Sky)).toBe(false);
            expect(state.hasComponent(env, Sun)).toBe(false);
            expect(state.hasComponent(env, Stars)).toBe(false);
            expect(state.hasComponent(env, Moon)).toBe(false);
            expect(state.hasComponent(env, Haze)).toBe(false);
            expect(state.hasComponent(env, Clouds)).toBe(false);
        });

        test("drives directional light color and intensity", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 0;
            Skylab.elevation[env] = 45;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);

            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            const expected = sampleGradient(45);
            expect(DirectionalLight.color[light]).toBe(expected.sunColor);
            expect(DirectionalLight.intensity[light]).toBeCloseTo(expected.sunIntensity, 2);
        });

        test("drives ambient light", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 0;
            Skylab.elevation[env] = 45;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);

            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            const expected = sampleGradient(45);
            expect(AmbientLight.color[ambient]).toBe(expected.ambientColor);
            expect(AmbientLight.intensity[ambient]).toBeCloseTo(expected.ambientIntensity, 2);
        });

        test("does nothing without Skylab component", () => {
            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            DirectionalLight.color[light] = 0xffffff;
            DirectionalLight.intensity[light] = 0.8;

            state.step();

            expect(DirectionalLight.color[light]).toBe(0xffffff);
            expect(DirectionalLight.intensity[light]).toBe(0.8);
        });

        test("responds to elevation changes", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            Skylab.azimuth[env] = 180;
            Skylab.elevation[env] = 50;
            state.step();
            const highSun = DirectionalLight.intensity[light];

            Skylab.elevation[env] = 0;
            state.step();
            const horizonSun = DirectionalLight.intensity[light];

            expect(highSun).toBeGreaterThan(horizonSun);
        });

        test("night light comes from above", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 180;
            Skylab.elevation[env] = -45;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            expect(DirectionalLight.directionY[light]).toBeLessThan(0);
        });

        test("twilight blends direction", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 180;
            Skylab.elevation[env] = -12;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            expect(DirectionalLight.directionY[light]).toBeLessThanOrEqual(0);
        });

        test("twilight direction remains well-defined", () => {
            const env = state.addEntity();
            state.addComponent(env, Skylab);
            Skylab.azimuth[env] = 180;
            Skylab.elevation[env] = -12;

            const light = state.addEntity();
            state.addComponent(light, DirectionalLight);
            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);

            state.step();

            const dx = DirectionalLight.directionX[light];
            const dy = DirectionalLight.directionY[light];
            const dz = DirectionalLight.directionZ[light];
            expect(Math.hypot(dx, dy, dz)).toBeGreaterThan(0.9);
        });
    });
});
