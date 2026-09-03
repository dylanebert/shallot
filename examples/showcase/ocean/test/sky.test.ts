import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import {
    DUSK_SKY_DEFAULTS,
    DuskSkyGpu,
    sampleCloud,
    sampleElevation,
    sampleHaze,
    sampleSky,
    sampleSun,
} from "../src/sky";

const base = (overrides: Partial<d.Infer<typeof DuskSkyGpu>> = {}) =>
    DuskSkyGpu({
        zenith: d.vec4f(0.12, 0.25, 0.45, 0),
        horizon: d.vec4f(0.68, 0.4, 0.32, 0),
        haze: d.vec4f(0.8, 0.45, 0.3, 0),
        cloud: d.vec4f(0.3, 0.2, 0.4, 0),
        sun: d.vec4f(1, 0.7, 0.3, 0),
        exposure: d.vec4f(1, 0, 0, 0),
        ...overrides,
    });
const sun = d.vec3f(0, -0.3, -0.954);
const dir = d.vec3f(0.2, 0.35, 0.916);
const values = (v: ReturnType<typeof d.vec3f>) => [v.x, v.y, v.z];

function expectChanged(a: number[], b: number[]) {
    expect(a.some((value, i) => Math.abs(value - b[i]) > 1e-6)).toBe(true);
}

describe("demo-local dusk sky", () => {
    test("disabled optional layers preserve the elevation gradient", () => {
        expect(values(sampleSky(base(), dir, sun))).toEqual(values(sampleElevation(base(), dir)));
    });

    test("each parameter moves only its named channel", () => {
        const channels = (sky: d.Infer<typeof DuskSkyGpu>) => ({
            elevation: values(sampleElevation(sky, dir)),
            haze: values(sampleHaze(sky, dir)),
            cloud: values(sampleCloud(sky, dir)),
            sun: values(sampleSun(sky, dir, sun)),
        });
        const unchanged = channels(base());
        const mutations = [
            ["zenith", "elevation"],
            ["horizon", "elevation"],
            ["haze", "haze"],
            ["cloud", "cloud"],
            ["sun", "sun"],
        ] as const;
        for (const [parameter, target] of mutations) {
            const changed = channels(
                base({
                    [parameter]: d.vec4f(
                        0.9,
                        0.8,
                        0.7,
                        parameter === "zenith" || parameter === "horizon" ? 0 : 0.5,
                    ),
                }),
            );
            for (const channel of Object.keys(unchanged) as (keyof typeof unchanged)[]) {
                if (channel === target) expectChanged(unchanged[channel], changed[channel]);
                else expect(changed[channel]).toEqual(unchanged[channel]);
            }
        }

        const sky = base();
        const exposedSky = base({ exposure: d.vec4f(1.5, 0, 0, 0) });
        expect(channels(exposedSky)).toEqual(channels(sky));
        const unexposed = values(sampleSky(sky, dir, sun));
        const exposed = values(sampleSky(exposedSky, dir, sun));
        exposed.forEach((value, i) => {
            expect(value).toBeCloseTo(unexposed[i] * 1.5, 5);
        });
    });

    test("stays finite and continuous across the horizon", () => {
        const sky = base({
            haze: d.vec4f(0.8, 0.45, 0.3, 0.2),
            cloud: d.vec4f(0.3, 0.2, 0.4, 0.2),
            sun: d.vec4f(1, 0.7, 0.3, 0.4),
        });
        const below = values(sampleSky(sky, d.vec3f(0.2, -1e-5, 0.98), sun));
        const above = values(sampleSky(sky, d.vec3f(0.2, 1e-5, 0.98), sun));
        expect([...below, ...above].every(Number.isFinite)).toBe(true);
        above.forEach((value, i) => {
            expect(Math.abs(value - below[i])).toBeLessThan(0.001);
        });
    });

    test("pins reference-shaped sane defaults", () => {
        expect(DUSK_SKY_DEFAULTS).toEqual({
            zenith: 0x31577e,
            horizon: 0xd5a6a0,
            haze: 0xe8b5a4,
            hazeStrength: 0.16,
            cloud: 0x715f78,
            cloudStrength: 0.18,
            sun: 0xffd5a0,
            sunStrength: 0.42,
            exposure: 1.05,
        });
    });
});
