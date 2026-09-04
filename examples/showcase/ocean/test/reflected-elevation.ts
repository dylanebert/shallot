import { unpackColor } from "@dylanebert/shallot";
import * as d from "typegpu/data";
import { CAPTURE } from "../src/conditions";
import { BECKMANN_VARIANCE_FLOOR, WATER_BODY } from "../src/ocean/surface";
import { DUSK_SKY_DEFAULTS, DuskSkyGpu, sampleSky } from "../src/sky";
import fixture from "./reference/look-relations.json";

const VFOV = Math.PI / 3;
const SLOPE_VARIANCE_RANGE = [BECKMANN_VARIANCE_FLOOR, 0.08] as const;
const WATER_BANDS = [
    ["q0", 0.52, 0.59],
    ["q1", 0.59, 0.66],
    ["q2", 0.66, 0.82],
    ["q3", 0.82, 1],
] as const;

export interface ReflectedElevationBand {
    name: string;
    elevation: number;
    chroma: number;
    blueRedRatio: number;
    waterSkyLuma: number;
    reflectionShare: number;
}

export interface ReflectedElevationPrediction {
    cameraPitch: number;
    verticalFov: number;
    slopeVariance: readonly [number, number];
    bands: ReflectedElevationBand[];
    reflectionDominant: boolean;
    clearsS9: boolean;
}

function rgb(hex: number): ReturnType<typeof d.vec3f> {
    const value = unpackColor(hex);
    return d.vec3f(value.r, value.g, value.b);
}

function srgb(linear: number): number {
    return 255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055);
}

function luma(value: readonly number[]): number {
    return 0.2126 * value[0]! + 0.7152 * value[1]! + 0.0722 * value[2]!;
}

function meanFresnel(cosTheta: number, sigma: number): number {
    const exponent = 5 * Math.exp(-2.69 * sigma);
    return (1 - Math.min(Math.max(cosTheta, 0), 1)) ** exponent / (1 + 22.7 * sigma ** 1.5);
}

/** Predicts whether the existing reflected-sky carrier can reach S9 before a capture is attempted. */
export function reflectedElevationPrediction(): ReflectedElevationPrediction {
    const sky = DuskSkyGpu({
        zenith: d.vec4f(rgb(DUSK_SKY_DEFAULTS.zenith), 0),
        horizon: d.vec4f(rgb(DUSK_SKY_DEFAULTS.horizon), 0),
        haze: d.vec4f(rgb(DUSK_SKY_DEFAULTS.haze), DUSK_SKY_DEFAULTS.hazeStrength),
        cloud: d.vec4f(rgb(DUSK_SKY_DEFAULTS.cloud), DUSK_SKY_DEFAULTS.cloudStrength),
        sun: d.vec4f(rgb(DUSK_SKY_DEFAULTS.sun), DUSK_SKY_DEFAULTS.sunStrength),
        exposure: d.vec4f(DUSK_SKY_DEFAULTS.exposure, 0, 0, 0),
    });
    const skyBandLuma = fixture.relations.s9Colour.skyLuma;
    const body = [...WATER_BODY];
    const bands = WATER_BANDS.map(([name, y0, y1], index) => {
        const row = (y0 + y1) / 2;
        const elevation = CAPTURE.camera.pitch + (row - 0.5) * VFOV;
        const direction = d.vec3f(0, Math.sin(elevation), Math.cos(elevation));
        const reflected = sampleSky(sky, direction, d.vec3f(1, 0, 0));
        const skyRgb = [reflected.x, reflected.y, reflected.z];
        const cosTheta = Math.sin(elevation);
        const fresnels = SLOPE_VARIANCE_RANGE.map((variance) => {
            const factor = meanFresnel(cosTheta, Math.sqrt(variance));
            return 0.02 + 0.98 * factor;
        });
        const fresnel = Math.max(...fresnels);
        const linear = body.map(
            (channel, channelIndex) => channel * (1 - fresnel) + skyRgb[channelIndex]! * fresnel,
        );
        const display = linear.map(srgb);
        const reflectedEnergy = fresnel * luma(skyRgb);
        const bodyEnergy = (1 - fresnel) * luma(body);
        return {
            name,
            elevation,
            chroma: Math.max(...display) - Math.min(...display),
            blueRedRatio: display[2]! / display[0]!,
            waterSkyLuma: luma(linear) / skyBandLuma,
            reflectionShare: reflectedEnergy / (reflectedEnergy + bodyEnergy),
            floor: fixture.relations.s9Colour.waterSkyLumaRatios[index]!.min,
        };
    });
    const colour = fixture.relations.s9Colour;
    return {
        cameraPitch: CAPTURE.camera.pitch,
        verticalFov: VFOV,
        slopeVariance: SLOPE_VARIANCE_RANGE,
        bands: bands.map(({ floor: _, ...band }) => band),
        reflectionDominant: bands.every((band) => band.reflectionShare > 0.5),
        clearsS9:
            bands.every((band) => band.waterSkyLuma >= band.floor) &&
            bands.at(-1)!.blueRedRatio >= colour.nearBlueRedRatio.min &&
            bands.at(-1)!.chroma >= colour.nearMeanChroma.min &&
            bands.at(-1)!.chroma <= colour.nearMeanChroma.max,
    };
}

export function printReflectedElevationPrediction(
    prediction = reflectedElevationPrediction(),
): void {
    console.log(
        `CPU reflected-elevation reach pitch=${prediction.cameraPitch.toFixed(3)} vFOV=${prediction.verticalFov.toFixed(3)} variance=${prediction.slopeVariance.join("..")}`,
    );
    for (const band of prediction.bands)
        console.log(
            `  ${band.name} elevation=${band.elevation.toFixed(3)} chroma=${band.chroma.toFixed(2)} B/R=${band.blueRedRatio.toFixed(3)} water/sky=${band.waterSkyLuma.toFixed(3)} reflection-share=${band.reflectionShare.toFixed(3)}`,
        );
    console.log(
        `  reflection-dominant=${prediction.reflectionDominant ? "PASS" : "FAIL"} S9-floors=${prediction.clearsS9 ? "PASS" : "FAIL"}`,
    );
}
