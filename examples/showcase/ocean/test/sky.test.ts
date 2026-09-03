import { describe, expect, test } from "bun:test";
import { unpackColor } from "@dylanebert/shallot";
import * as d from "typegpu/data";
import {
    DuskSkyGpu,
    SOLAR_ANGULAR_RADIUS,
    sampleCloud,
    sampleElevation,
    sampleHaze,
    sampleSky,
    sampleSun,
    solarDiskProfile,
} from "../src/sky";
import fixture from "./reference/look-relations.json";

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
            sun: values(sampleSun(sky, d.vec3f(-sun.x, -sun.y, -sun.z), sun)),
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

    test("solar profile has a finite angular edge and a limb-darkened core", () => {
        expect(SOLAR_ANGULAR_RADIUS).toBeCloseTo(0.00465, 6);
        expect(solarDiskProfile(Math.cos(SOLAR_ANGULAR_RADIUS + 1e-5))).toBe(0);
        const limb = solarDiskProfile(Math.cos(SOLAR_ANGULAR_RADIUS * 0.8));
        const core = solarDiskProfile(1);
        expect(limb).toBeGreaterThan(0);
        expect(core).toBeGreaterThan(limb);
        expect(core).toBeCloseTo(1, 6);
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

    test("CPU sky recipe reaches the marked dusk anchors", () => {
        const rgb = (hex: number) => {
            const value = unpackColor(hex);
            return d.vec3f(value.r, value.g, value.b);
        };
        const anchorCarriers = { zenith: 0xd7a4a0, horizon: 0x976d70, haze: 0xa8797b };
        const samples = {
            zenith: values(
                sampleElevation(
                    base({ zenith: d.vec4f(rgb(anchorCarriers.zenith), 0) }),
                    d.vec3f(0, 1, 0),
                ),
            ),
            horizon: values(
                sampleElevation(
                    base({ horizon: d.vec4f(rgb(anchorCarriers.horizon), 0) }),
                    d.vec3f(1, 0, 0),
                ),
            ),
            haze: values(
                sampleSky(
                    base({
                        horizon: d.vec4f(rgb(anchorCarriers.horizon), 0),
                        haze: d.vec4f(rgb(anchorCarriers.haze), 0.08),
                        cloud: d.vec4f(0),
                        sun: d.vec4f(0),
                        exposure: d.vec4f(1, 0, 0, 0),
                    }),
                    d.vec3f(1, 0, 0),
                    sun,
                ),
            ),
        };
        for (const name of ["zenith", "horizon", "haze"] as const) {
            samples[name].forEach((value, channel) => {
                expect(
                    Math.abs(value - fixture.relations.skyAnchorsLinear[name][channel]!),
                ).toBeLessThanOrEqual(fixture.relations.skyAnchorChannelTolerance);
            });
        }

        const revertedZenith = rgb(0x31577e);
        expect(
            values(revertedZenith).some(
                (value, channel) =>
                    Math.abs(value - fixture.relations.skyAnchorsLinear.zenith[channel]!) >
                    fixture.relations.skyAnchorChannelTolerance,
            ),
        ).toBe(true);
    });
});
