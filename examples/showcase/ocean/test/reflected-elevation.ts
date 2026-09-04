import { unpackColor } from "@dylanebert/shallot";
import * as d from "typegpu/data";
import { tmAgx } from "../../../../packages/shallot/src/standard/glaze/tonemap";
import { CAPTURE, sunDirection } from "../src/conditions";
import {
    AERIAL_DENSITY,
    BECKMANN_VARIANCE_FLOOR,
    meanFresnel,
    WATER_AMBIENT,
    WATER_BODY,
} from "../src/ocean/surface";
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
    rows: readonly [number, number];
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
    ceilings: { nearChroma: number; nearBlueRedRatio: number; waterSkyLuma: number[] };
}

export type ReachabilityRoute = "reachable" | "unreachable";
export type ReachabilityProfile = "s9" | "s20";

export interface ModelCarriers {
    body: readonly [number, number, number];
    bodyStrength: number;
    reflectionStrength: number;
    ambient: number;
    reflectedElevationCeiling: number;
    highElevationBlend: number;
}

const PROFILE_CARRIERS: Record<ReachabilityProfile, ModelCarriers> = {
    s9: {
        body: [0.013, 0.052, 0.085],
        bodyStrength: 1,
        reflectionStrength: 1,
        ambient: WATER_AMBIENT,
        reflectedElevationCeiling: 0.05,
        highElevationBlend: 0,
    },
    s20: {
        body: WATER_BODY,
        bodyStrength: 1,
        reflectionStrength: 1,
        ambient: WATER_AMBIENT,
        reflectedElevationCeiling: 1,
        highElevationBlend: 0,
    },
};

function rgb(hex: number): ReturnType<typeof d.vec3f> {
    const value = unpackColor(hex);
    return d.vec3f(value.r, value.g, value.b);
}

function sky(carriers: ModelCarriers): ReturnType<typeof DuskSkyGpu> {
    const zenith = rgb(DUSK_SKY_DEFAULTS.zenith);
    const horizon = rgb(DUSK_SKY_DEFAULTS.horizon);
    const high = d.vec3f(
        zenith.x + (horizon.x - zenith.x) * carriers.highElevationBlend,
        zenith.y + (horizon.y - zenith.y) * carriers.highElevationBlend,
        zenith.z + (horizon.z - zenith.z) * carriers.highElevationBlend,
    );
    return DuskSkyGpu({
        zenith: d.vec4f(high, 0),
        horizon: d.vec4f(horizon, 0),
        haze: d.vec4f(rgb(DUSK_SKY_DEFAULTS.haze), DUSK_SKY_DEFAULTS.hazeStrength),
        cloud: d.vec4f(rgb(DUSK_SKY_DEFAULTS.cloud), DUSK_SKY_DEFAULTS.cloudStrength),
        sun: d.vec4f(rgb(DUSK_SKY_DEFAULTS.sun), DUSK_SKY_DEFAULTS.sunStrength),
        exposure: d.vec4f(DUSK_SKY_DEFAULTS.exposure, 0, 0, 0),
    });
}

function srgb(value: number): number {
    const clamped = Math.max(0, value);
    return 255 * (clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * clamped ** (1 / 2.4) - 0.055);
}

function luma(value: readonly number[]): number {
    return 0.2126 * value[0]! + 0.7152 * value[1]! + 0.0722 * value[2]!;
}

function sampleBand(
    name: string,
    from: number,
    to: number,
    carriers: ModelCarriers,
): ReflectedElevationBand {
    const elevation = CAPTURE.camera.pitch + ((from + to) / 2 - 0.5) * VFOV;
    const horizontalDirection = d.vec3f(0, 0, 1);
    const light = d.vec3f(...sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation));
    const recipe = sky(carriers);
    const aerial = sampleSky(recipe, horizontalDirection, light);
    const aerialRgb = [aerial.x, aerial.y, aerial.z];
    const reflectedRgb = [0, 0, 0];
    const varianceSteps = 8;
    let fresnel = 0;
    let skySamples = 0;
    for (let step = 0; step <= varianceSteps; step++) {
        const variance =
            SLOPE_VARIANCE_RANGE[0] +
            ((SLOPE_VARIANCE_RANGE[1] - SLOPE_VARIANCE_RANGE[0]) * step) / varianceSteps;
        const sigma = Math.sqrt(variance);
        fresnel += 0.02 + 0.98 * meanFresnel(Math.sin(elevation), sigma);
        for (const sign of [-1, 1]) {
            const reflectedElevation = Math.min(
                Math.max(elevation + sign * 2 * Math.atan(sigma), 0),
                carriers.reflectedElevationCeiling,
            );
            const direction = d.vec3f(
                0,
                Math.sin(reflectedElevation),
                Math.cos(reflectedElevation),
            );
            const reflected = sampleSky(recipe, direction, light);
            reflectedRgb[0]! += reflected.x;
            reflectedRgb[1]! += reflected.y;
            reflectedRgb[2]! += reflected.z;
            skySamples++;
        }
    }
    fresnel /= varianceSteps + 1;
    for (let channel = 0; channel < reflectedRgb.length; channel++)
        reflectedRgb[channel]! /= skySamples;
    const body = carriers.body.map((value) => value * carriers.bodyStrength);
    const reflection = reflectedRgb.map((value) => value * carriers.reflectionStrength);
    const base = body.map((value, index) => value * (1 - fresnel) + reflection[index]! * fresnel);
    const eyeHeight = CAPTURE.camera.distance * Math.sin(CAPTURE.camera.pitch);
    const distance = eyeHeight / Math.max(Math.sin(elevation), 0.001);
    const fade = 1 - Math.exp(-AERIAL_DENSITY * distance);
    const foamShare = fixture.provenance.reachability.model.foamShare;
    const sun = rgb(0xfff3dc);
    const sunRgb = [sun.x * 1.2, sun.y * 1.2, sun.z * 1.2];
    const foamLight = reflection.map(
        (value, index) => value * carriers.ambient + sunRgb[index]! * (1 - carriers.ambient),
    );
    const withFoam = base.map(
        (value, index) => value * (1 - foamShare) + foamLight[index]! * foamShare,
    );
    const linear = withFoam.map((value, index) => value * (1 - fade) + aerialRgb[index]! * fade);
    const graded = tmAgx(d.vec3f(linear[0]!, linear[1]!, linear[2]!));
    const display = [graded.x, graded.y, graded.z].map(srgb);
    const bodyEnergy = luma(body) * (1 - fresnel) * (1 - fade);
    const reflectionEnergy = luma(reflection) * fresnel * (1 - fade) + luma(aerialRgb) * fade;
    return {
        name,
        rows: [from, to],
        elevation,
        chroma: Math.max(...display) - Math.min(...display),
        blueRedRatio: display[2]! / Math.max(display[0]!, 1 / 255),
        waterSkyLuma: luma(display) / fixture.relations.s9Colour.skyLuma / 255,
        reflectionShare: reflectionEnergy / (reflectionEnergy + bodyEnergy),
    };
}

/** Independently evaluates the CPU sky and shader composite in fixture display space. */
export function predictBands(
    profile: ReachabilityProfile,
    carriers: ModelCarriers = PROFILE_CARRIERS[profile],
): ReflectedElevationBand[] {
    return WATER_BANDS.map(([name, from, to]) => sampleBand(name, from, to, carriers));
}

function sweepCarriers(): ModelCarriers[] {
    const bounds = fixture.provenance.reachability.carrierBounds;
    const bodies = [PROFILE_CARRIERS.s20.body, PROFILE_CARRIERS.s9.body] as const;
    return bodies.flatMap((body) =>
        bounds.highElevationBlend.flatMap((highElevationBlend) =>
            bounds.strength.flatMap((strength) => [
                {
                    body,
                    bodyStrength: strength,
                    reflectionStrength: strength,
                    ambient: WATER_AMBIENT,
                    reflectedElevationCeiling: 1,
                    highElevationBlend,
                },
            ]),
        ),
    );
}

export function reflectedElevationPrediction(
    profile: ReachabilityProfile = "s20",
    carriers: ModelCarriers = PROFILE_CARRIERS[profile],
): ReflectedElevationPrediction {
    const bands = predictBands(profile, carriers);
    const samples =
        profile === "s20" ? sweepCarriers().map((value) => predictBands(profile, value)) : [bands];
    const ceilings = {
        nearChroma: Math.max(...samples.map((sample) => sample[3]!.chroma)),
        nearBlueRedRatio: Math.max(...samples.map((sample) => sample[3]!.blueRedRatio)),
        waterSkyLuma: bands.map((_, index) =>
            Math.max(...samples.map((sample) => sample[index]!.waterSkyLuma)),
        ),
    };
    const colour = fixture.relations.s9Colour;
    return {
        cameraPitch: CAPTURE.camera.pitch,
        verticalFov: VFOV,
        slopeVariance: SLOPE_VARIANCE_RANGE,
        bands,
        reflectionDominant: bands.slice(1, 3).every((band) => band.reflectionShare > 0.5),
        clearsS9:
            ceilings.waterSkyLuma.every(
                (value, index) => value >= colour.waterSkyLumaRatios[index]!.min,
            ) &&
            ceilings.nearBlueRedRatio >= colour.nearBlueRedRatio.min &&
            ceilings.nearChroma >= colour.nearMeanChroma.min &&
            ceilings.nearChroma <= colour.nearMeanChroma.max,
        ceilings,
    };
}

export function bodyMutation(
    profile: ReachabilityProfile,
    scale: number,
): ReflectedElevationPrediction {
    const base = PROFILE_CARRIERS[profile];
    return reflectedElevationPrediction(profile, { ...base, bodyStrength: scale });
}

export function routeReachability(
    prediction: ReflectedElevationPrediction,
    reachable: (value: ReflectedElevationPrediction) => boolean = (value) =>
        value.reflectionDominant && value.clearsS9,
): ReachabilityRoute {
    return reachable(prediction) ? "reachable" : "unreachable";
}

export function printReflectedElevationPrediction(
    prediction = reflectedElevationPrediction(),
): void {
    console.log(
        `CPU reflected-elevation reach pitch=${prediction.cameraPitch.toFixed(3)} vFOV=${prediction.verticalFov.toFixed(3)} variance=${prediction.slopeVariance.join("..")}`,
    );
    for (const [index, band] of prediction.bands.entries())
        console.log(
            `  ${band.name} rows=${band.rows.join("..")} elevation=${band.elevation.toFixed(3)} chroma=${band.chroma.toFixed(2)} B/R=${band.blueRedRatio.toFixed(3)} water/sky=${band.waterSkyLuma.toFixed(3)} ceiling=${prediction.ceilings.waterSkyLuma[index]!.toFixed(3)} floor=${fixture.relations.s9Colour.waterSkyLumaRatios[index]!.min.toFixed(3)} reflection-share=${band.reflectionShare.toFixed(3)}`,
        );
    console.log(
        `  near-chroma ceiling=${prediction.ceilings.nearChroma.toFixed(2)} floor=${fixture.relations.s9Colour.nearMeanChroma.min.toFixed(2)}; near-B/R ceiling=${prediction.ceilings.nearBlueRedRatio.toFixed(3)} floor=${fixture.relations.s9Colour.nearBlueRedRatio.min.toFixed(3)}`,
    );
    console.log(
        `  reflection-dominant(mid/far)=${prediction.reflectionDominant ? "PASS" : "FAIL"}; near-share=${prediction.bands[3]!.reflectionShare.toFixed(3)} expected-body-dominant; route=${routeReachability(prediction).toUpperCase()}`,
    );
}
