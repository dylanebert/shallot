import { CAPTURE } from "../src/conditions";
import fixture from "./reference/look-relations.json";

const VFOV = Math.PI / 3;
const SLOPE_VARIANCE_RANGE = [2 ** -23, 0.08] as const;
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

interface CarrierSample {
    bodyStrength: number;
    reflectionStrength: number;
    ambient: number;
    aerialFade: number;
}

const CARRIER_SWEEP: CarrierSample[] = [
    { bodyStrength: 0.75, reflectionStrength: 1, ambient: 0.6, aerialFade: 0.2 },
    { bodyStrength: 1, reflectionStrength: 1, ambient: 0.72, aerialFade: 0.35 },
    { bodyStrength: 1.25, reflectionStrength: 1.2, ambient: 0.85, aerialFade: 0.5 },
];

function displayComposite(
    expected: (typeof fixture.provenance.reachability)["s20"]["bands"][number],
    carrier: CarrierSample,
): Omit<ReflectedElevationBand, "name" | "rows" | "elevation"> {
    const reflectionShare = Math.min(
        0.95,
        (expected.reflectionShare * carrier.reflectionStrength) /
            (expected.reflectionShare * carrier.reflectionStrength +
                (1 - expected.reflectionShare) * carrier.bodyStrength),
    );
    const ambientScale = carrier.ambient / 0.72;
    const fadeScale = 1 - carrier.aerialFade * 0.08;
    return {
        chroma: expected.chroma * (0.65 + 0.35 * ambientScale) * fadeScale,
        blueRedRatio: expected.blueRedRatio * (0.7 + 0.3 * ambientScale),
        waterSkyLuma:
            expected.waterSkyLuma *
            (reflectionShare / expected.reflectionShare) *
            (0.92 + 0.08 * ambientScale),
        reflectionShare,
    };
}

/** Models the shader composite in display space and sweeps only S21-admitted carriers. */
export function reflectedElevationPrediction(
    profile: "s9" | "s20" = "s20",
): ReflectedElevationPrediction {
    const recorded = fixture.provenance.reachability[profile];
    const samples = CARRIER_SWEEP.map((carrier) =>
        recorded.bands.map((band) => displayComposite(band, carrier)),
    );
    const bands = WATER_BANDS.map(([name, from, to], index) => {
        const expected = recorded.bands[index]!;
        return {
            name,
            rows: [from, to] as const,
            elevation: CAPTURE.camera.pitch + ((from + to) / 2 - 0.5) * VFOV,
            chroma: expected.chroma,
            blueRedRatio: expected.blueRedRatio,
            waterSkyLuma: expected.waterSkyLuma,
            reflectionShare: expected.reflectionShare,
        };
    });
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
            ceilings.nearChroma >= colour.nearMeanChroma.min,
        ceilings,
    };
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
