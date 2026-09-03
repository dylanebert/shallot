import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { decode as decodeJpeg } from "jpeg-js";
import { PNG } from "pngjs";
import fixture from "./reference/look-relations.json";

export interface Pixels {
    width: number;
    height: number;
    data: Uint8Array;
}

export interface LookReading {
    bands: Record<string, { srgb: [number, number, number]; luma: number }>;
    farWaterSkyHueDistance: number;
    lowerBandBrightSpecks: number;
    horizon: { row: number; transitionWidth: number; continuity: number; valueStep: number };
    duskBalance: {
        waterSkyLumaRatios: [number, number, number, number];
        fadeExtent: number;
        nearBlueRedRatio: number;
    };
    foam: { nearCoverage: number; maxLuma: number };
    normalResponse: {
        midRowDeviation: number;
        nearRowDeviation: number;
        nearMeanChroma: number;
        chromaWeightedWaterSkyHueDistance: number;
        brightWaterFraction: number;
    };
    lumaRange: number;
}

const DEFAULT_HORIZON = 210 / 720;
const BAND_RANGES = {
    sky: [0.08, DEFAULT_HORIZON],
    horizon: [DEFAULT_HORIZON, 0.52],
    farWater: [0.52, 0.66],
    midWater: [0.66, 0.82],
    nearWater: [0.82, 1],
} as const;

type BandRanges = Record<keyof typeof BAND_RANGES, readonly [number, number]>;

export function bandRanges(horizonRow: number, height: number): BandRanges {
    if (horizonRow / height === DEFAULT_HORIZON) return BAND_RANGES;
    const horizon = horizonRow / height;
    const map = (value: number) =>
        horizon + ((value - DEFAULT_HORIZON) * (1 - horizon)) / (1 - DEFAULT_HORIZON);
    return {
        sky: [0.08, horizon],
        horizon: [horizon, map(0.52)],
        farWater: [map(0.52), map(0.66)],
        midWater: [map(0.66), map(0.82)],
        nearWater: [map(0.82), 1],
    };
}

function luma(r: number, g: number, b: number): number {
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

function mean(
    image: Pixels,
    from: number,
    to: number,
): { srgb: [number, number, number]; luma: number } {
    const y0 = Math.floor(image.height * from);
    const y1 = Math.max(y0 + 1, Math.floor(image.height * to));
    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
    for (let y = y0; y < y1; y++) {
        for (let x = 0; x < image.width; x++) {
            const i = (y * image.width + x) * 4;
            r += image.data[i] ?? 0;
            g += image.data[i + 1] ?? 0;
            b += image.data[i + 2] ?? 0;
            count++;
        }
    }
    const srgb: [number, number, number] = [r / count, g / count, b / count];
    return { srgb, luma: luma(...srgb) };
}

function hue([r8, g8, b8]: [number, number, number]): number {
    const r = r8 / 255;
    const g = g8 / 255;
    const b = b8 / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    if (delta === 0) return 0;
    if (max === r) return (60 * ((g - b) / delta) + 360) % 360;
    if (max === g) return 60 * ((b - r) / delta + 2);
    return 60 * ((r - g) / delta + 4);
}

function brightSpeckMask(image: Pixels): Uint8Array {
    const y0 = Math.floor(image.height * 0.7);
    const values: number[] = [];
    for (let y = y0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const i = (y * image.width + x) * 4;
            values.push(luma(image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0));
        }
    }
    values.sort((a, b) => a - b);
    const threshold = values[Math.floor(values.length * 0.997)] ?? 1;
    const hot = new Uint8Array(image.width * image.height);
    for (let y = y0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const i = (y * image.width + x) * 4;
            if (
                luma(image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0) > threshold
            )
                hot[y * image.width + x] = 1;
        }
    }
    return hot;
}

function brightSpecks(image: Pixels): number {
    const hot = brightSpeckMask(image);
    const y0 = Math.floor(image.height * 0.7);
    let components = 0;
    const stack: number[] = [];
    for (let i = y0 * image.width; i < hot.length; i++) {
        if (!hot[i]) continue;
        components++;
        hot[i] = 0;
        stack.push(i);
        while (stack.length) {
            const at = stack.pop() ?? 0;
            const x = at % image.width;
            for (const next of [at - image.width, at + image.width, at - 1, at + 1]) {
                if (next < y0 * image.width || next >= hot.length || !hot[next]) continue;
                if ((next === at - 1 && x === 0) || (next === at + 1 && x === image.width - 1))
                    continue;
                hot[next] = 0;
                stack.push(next);
            }
        }
    }
    return components;
}

function horizon(image: Pixels): {
    row: number;
    transitionWidth: number;
    continuity: number;
    valueStep: number;
    means: number[];
} {
    const means: number[] = [];
    const fromX = Math.floor(image.width * 0.1);
    const toX = Math.floor(image.width * 0.9);
    for (let y = 0; y < image.height; y++) {
        let sum = 0;
        for (let x = fromX; x < toX; x++) {
            const i = (y * image.width + x) * 4;
            sum += luma(image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0);
        }
        means.push(sum / (toX - fromX));
    }
    const gradients = means
        .slice(1)
        .map((value, y) => ({ y: y + 1, value: Math.abs(value - (means[y] ?? value)) }));
    const window = gradients.filter(({ y }) => y >= image.height * 0.25 && y <= image.height * 0.7);
    const peak = window.reduce((best, item) => (item.value > best.value ? item : best), {
        y: 0,
        value: 0,
    });
    const cutoff = peak.value * 0.2;
    let lo = peak.y - 1;
    let hi = peak.y - 1;
    while (lo > 0 && (gradients[lo - 1]?.value ?? 0) >= cutoff) lo--;
    while (hi + 1 < gradients.length && (gradients[hi + 1]?.value ?? 0) >= cutoff) hi++;
    const crossings: number[] = [];
    for (let x = fromX; x < toX; x += Math.max(1, Math.floor(image.width / 80))) {
        let bestY = lo;
        let best = 0;
        for (let y = Math.max(1, lo - 3); y <= Math.min(image.height - 1, hi + 3); y++) {
            const a = (y * image.width + x) * 4;
            const b = ((y - 1) * image.width + x) * 4;
            const delta = Math.abs(
                luma(image.data[a] ?? 0, image.data[a + 1] ?? 0, image.data[a + 2] ?? 0) -
                    luma(image.data[b] ?? 0, image.data[b + 1] ?? 0, image.data[b + 2] ?? 0),
            );
            if (delta > best) {
                best = delta;
                bestY = y;
            }
        }
        crossings.push(bestY);
    }
    const crossingMean = crossings.reduce((sum, value) => sum + value, 0) / crossings.length;
    const deviation = Math.sqrt(
        crossings.reduce((sum, value) => sum + (value - crossingMean) ** 2, 0) / crossings.length,
    );
    return {
        row: peak.y,
        transitionWidth: hi - lo + 1,
        continuity: 1 / (1 + deviation),
        valueStep: peak.value,
        means,
    };
}

function duskBalance(image: Pixels, row: number, means: number[]): LookReading["duskBalance"] {
    const sky = mean(image, 0.08, row / image.height);
    const water = Array.from({ length: 4 }, (_, index) =>
        mean(
            image,
            (row + ((image.height - row) * index) / 4) / image.height,
            (row + ((image.height - row) * (index + 1)) / 4) / image.height,
        ),
    );
    const start = means[row] ?? 0;
    const end = means.at(-1) ?? start;
    const delta = Math.abs(end - start);
    let y10 = row;
    let y90 = row;
    for (let y = row; y < means.length; y++) {
        if (Math.abs((means[y] ?? start) - start) >= delta * 0.1) {
            y10 = y;
            break;
        }
    }
    for (let y = y10; y < means.length; y++) {
        if (Math.abs((means[y] ?? start) - start) >= delta * 0.9) {
            y90 = y;
            break;
        }
    }
    const near = water[3]?.srgb ?? [0, 0, 0];
    return {
        waterSkyLumaRatios: water.map((band) => band.luma / sky.luma) as [
            number,
            number,
            number,
            number,
        ],
        fadeExtent: y90 - y10,
        nearBlueRedRatio: near[2] / Math.max(near[0], 1 / 255),
    };
}

function normalResponse(
    image: Pixels,
    skyMean: number,
    ranges: BandRanges,
): LookReading["normalResponse"] {
    const deviation = (range: readonly [number, number]): number => {
        const y0 = Math.floor(image.height * range[0]);
        const y1 = Math.floor(image.height * range[1]);
        let sum = 0;
        let count = 0;
        for (let y = y0; y < y1; y++) {
            let row = 0;
            for (let x = 0; x < image.width; x++) {
                const i = (y * image.width + x) * 4;
                row += luma(image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0);
            }
            row /= image.width;
            for (let x = 0; x < image.width; x++) {
                const i = (y * image.width + x) * 4;
                const delta =
                    luma(image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0) - row;
                sum += delta * delta;
                count++;
            }
        }
        return Math.sqrt(sum / count) * 255;
    };
    const weighted = (from: number, to: number): [number, number, number] => {
        const y0 = Math.floor(image.height * from);
        const y1 = Math.floor(image.height * to);
        const sum = [0, 0, 0];
        let weight = 0;
        for (let y = y0; y < y1; y++) {
            for (let x = 0; x < image.width; x++) {
                const i = (y * image.width + x) * 4;
                const rgb = [image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0];
                const chroma = Math.max(...rgb) - Math.min(...rgb);
                for (let channel = 0; channel < 3; channel++)
                    sum[channel]! += rgb[channel]! * chroma;
                weight += chroma;
            }
        }
        return sum.map((value) => value / Math.max(weight, 1)) as [number, number, number];
    };
    const skyHue = hue(weighted(...ranges.sky));
    const waterHue = hue(weighted(ranges.horizon[0], 1));
    const hueDelta = Math.abs(skyHue - waterHue);
    const nearY = Math.floor(image.height * ranges.nearWater[0]);
    const waterY = Math.floor(image.height * ranges.horizon[0]);
    let nearChroma = 0;
    let nearCount = 0;
    let bright = 0;
    let waterCount = 0;
    for (let y = waterY; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const i = (y * image.width + x) * 4;
            const rgb = [image.data[i] ?? 0, image.data[i + 1] ?? 0, image.data[i + 2] ?? 0];
            if (luma(rgb[0]!, rgb[1]!, rgb[2]!) > 0.8 * skyMean) bright++;
            waterCount++;
            if (y >= nearY) {
                nearChroma += Math.max(...rgb) - Math.min(...rgb);
                nearCount++;
            }
        }
    }
    return {
        midRowDeviation: deviation(ranges.midWater),
        nearRowDeviation: deviation(ranges.nearWater),
        nearMeanChroma: nearChroma / nearCount,
        chromaWeightedWaterSkyHueDistance: Math.min(hueDelta, 360 - hueDelta),
        brightWaterFraction: bright / waterCount,
    };
}

function foam(
    image: Pixels,
    nearMean: number,
    ranges: BandRanges,
): { nearCoverage: number; maxLuma: number } {
    const y0 = Math.floor(image.height * ranges.nearWater[0]);
    let marked = 0;
    let count = 0;
    let maxLuma = 0;
    for (let y = y0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const i = (y * image.width + x) * 4;
            const r = image.data[i] ?? 0;
            const g = image.data[i + 1] ?? 0;
            const b = image.data[i + 2] ?? 0;
            const value = luma(r, g, b);
            if (value >= nearMean + 20 / 255 && Math.max(r, g, b) - Math.min(r, g, b) <= 48) {
                marked++;
                maxLuma = Math.max(maxLuma, value);
            }
            count++;
        }
    }
    return { nearCoverage: marked / count, maxLuma };
}

export function analyze(image: Pixels): LookReading {
    const horizonReading = horizon(image);
    const ranges = bandRanges(horizonReading.row, image.height);
    const bands = Object.fromEntries(
        Object.entries(ranges).map(([name, [from, to]]) => [name, mean(image, from, to)]),
    );
    const skyHue = hue(bands.sky?.srgb ?? [0, 0, 0]);
    const waterHue = hue(bands.farWater?.srgb ?? [0, 0, 0]);
    const hueDelta = Math.abs(skyHue - waterHue);
    const allLuma = Object.values(bands).map((band) => band.luma);
    const balance = duskBalance(image, horizonReading.row, horizonReading.means);
    return {
        bands,
        farWaterSkyHueDistance: Math.min(hueDelta, 360 - hueDelta),
        lowerBandBrightSpecks: brightSpecks(image),
        horizon: {
            row: horizonReading.row,
            transitionWidth: horizonReading.transitionWidth,
            continuity: horizonReading.continuity,
            valueStep: horizonReading.valueStep,
        },
        duskBalance: balance,
        foam: foam(image, bands.nearWater?.luma ?? 0, ranges),
        normalResponse: normalResponse(image, bands.sky?.luma ?? 0, ranges),
        lumaRange: Math.max(...allLuma) - Math.min(...allLuma),
    };
}

export async function load(path: string): Promise<Pixels> {
    const bytes = await readFile(path);
    if (extname(path).toLowerCase() === ".png") {
        const png = PNG.sync.read(bytes);
        return { width: png.width, height: png.height, data: png.data };
    }
    const jpeg = decodeJpeg(bytes, { useTArray: true });
    return { width: jpeg.width, height: jpeg.height, data: jpeg.data };
}

export function speckOverlap(a: Pixels, b: Pixels): number {
    if (a.width !== b.width || a.height !== b.height) return 0;
    const left = brightSpeckMask(a);
    const right = brightSpeckMask(b);
    let intersection = 0;
    let union = 0;
    for (let i = 0; i < left.length; i++) {
        if (left[i] || right[i]) union++;
        if (left[i] && right[i]) intersection++;
    }
    return union === 0 ? 1 : intersection / union;
}

function inside(value: number, range: { min: number; max: number }): boolean {
    return Number.isFinite(value) && value >= range.min && value <= range.max;
}

export type ReferenceRelation =
    | "horizonWidth"
    | "horizonContinuity"
    | "farWaterSkyHueDistance"
    | "lowerBandBrightSpecks"
    | "foamNearCoverage"
    | "waterSkyLumaRatio0"
    | "waterSkyLumaRatio1"
    | "waterSkyLumaRatio2"
    | "waterSkyLumaRatio3"
    | "horizonValueStep"
    | "fadeExtent"
    | "nearBlueRedRatio"
    | "nearMeanChroma";

type Interval = { min: number; max: number };

type RelationReading = {
    farWaterSkyHueDistance: number;
    lowerBandBrightSpecks: number;
    horizon: LookReading["horizon"];
    duskBalance: Omit<LookReading["duskBalance"], "waterSkyLumaRatios"> & {
        waterSkyLumaRatios: number[];
    };
    foam: LookReading["foam"];
    normalResponse: LookReading["normalResponse"];
};

export function referenceRelationResults(
    reading: RelationReading,
    relations: typeof fixture.relations = fixture.relations,
): Record<ReferenceRelation, boolean> {
    return {
        horizonWidth: inside(reading.horizon.transitionWidth, relations.horizonWidth),
        horizonContinuity: reading.horizon.continuity >= relations.horizonContinuityMin,
        farWaterSkyHueDistance: inside(
            reading.farWaterSkyHueDistance,
            relations.farWaterSkyHueDistance,
        ),
        lowerBandBrightSpecks: inside(
            reading.lowerBandBrightSpecks,
            relations.lowerBandBrightSpecks,
        ),
        foamNearCoverage: inside(reading.foam.nearCoverage, relations.foamNearCoverage),
        waterSkyLumaRatio0: inside(
            reading.duskBalance.waterSkyLumaRatios[0]!,
            relations.waterSkyLumaRatios[0]!,
        ),
        waterSkyLumaRatio1: inside(
            reading.duskBalance.waterSkyLumaRatios[1]!,
            relations.waterSkyLumaRatios[1]!,
        ),
        waterSkyLumaRatio2: inside(
            reading.duskBalance.waterSkyLumaRatios[2]!,
            relations.waterSkyLumaRatios[2]!,
        ),
        waterSkyLumaRatio3: inside(
            reading.duskBalance.waterSkyLumaRatios[3]!,
            relations.waterSkyLumaRatios[3]!,
        ),
        horizonValueStep: inside(reading.horizon.valueStep, relations.horizonValueStep),
        fadeExtent: inside(reading.duskBalance.fadeExtent, relations.fadeExtent),
        nearBlueRedRatio: inside(reading.duskBalance.nearBlueRedRatio, relations.nearBlueRedRatio),
        nearMeanChroma: inside(reading.normalResponse.nearMeanChroma, relations.nearMeanChroma),
    };
}

function distance(value: number, interval: Interval): number {
    if (value < interval.min) return interval.min - value;
    if (value > interval.max) return value - interval.max;
    return 0;
}

export function satisfiesPriorGoodFloor(fresh: number, prior: number, interval: Interval): boolean {
    const priorDistance = distance(prior, interval);
    const freshDistance = distance(fresh, interval);
    return priorDistance === 0 ? freshDistance === 0 : freshDistance < priorDistance;
}

export function satisfiesReferenceRelations(
    reading: RelationReading,
    relations: typeof fixture.relations = fixture.relations,
): boolean {
    return (
        inside(reading.horizon.transitionWidth, relations.horizonWidth) &&
        reading.horizon.continuity >= relations.horizonContinuityMin &&
        inside(reading.farWaterSkyHueDistance, relations.farWaterSkyHueDistance) &&
        inside(reading.lowerBandBrightSpecks, relations.lowerBandBrightSpecks) &&
        inside(reading.foam.nearCoverage, relations.foamNearCoverage) &&
        reading.duskBalance.waterSkyLumaRatios.every((ratio, index) =>
            inside(ratio, relations.waterSkyLumaRatios[index]!),
        ) &&
        inside(reading.horizon.valueStep, relations.horizonValueStep) &&
        inside(reading.duskBalance.fadeExtent, relations.fadeExtent) &&
        inside(reading.duskBalance.nearBlueRedRatio, relations.nearBlueRedRatio) &&
        inside(reading.normalResponse.nearMeanChroma, relations.nearMeanChroma)
    );
}

export function assertRegeneratedReferences(
    readings: Record<"t26" | "t43", RelationReading>,
): void {
    for (const name of ["t26", "t43"] as const) {
        if (JSON.stringify(readings[name]) !== JSON.stringify(fixture.references[name]))
            throw new Error(`regenerated ${name} reading differs from committed fixture`);
        if (!satisfiesReferenceRelations(readings[name]))
            throw new Error(`gold ${name} fails a derived reference relation`);
    }
}

function printBounds(): void {
    console.log(`bounds ${fixture.oracleRevision}:`);
    for (const [name, margin] of Object.entries(fixture.provenance.relationMargins))
        console.log(
            `  ${name}: source=t26,t43 rows=${fixture.provenance.absoluteRelationRows.t26},${fixture.provenance.absoluteRelationRows.t43} margin=${margin}`,
        );
}

function print(path: string, reading: LookReading): void {
    console.log(`\n${basename(path)}`);
    for (const [name, band] of Object.entries(reading.bands)) {
        console.log(
            `  ${name.padEnd(10)} sRGB ${band.srgb.map((value) => value.toFixed(1)).join(", ")}  luma ${(band.luma * 255).toFixed(1)}`,
        );
    }
    console.log(`  far-water/sky hue distance ${reading.farWaterSkyHueDistance.toFixed(1)}°`);
    console.log(`  lower-band bright specks ${reading.lowerBandBrightSpecks}`);
    console.log(
        `  horizon row ${reading.horizon.row}/${reading.bands.sky ? "image" : "?"} transition ${reading.horizon.transitionWidth}px  continuity ${reading.horizon.continuity.toFixed(3)} value-step ${(reading.horizon.valueStep * 255).toFixed(1)}`,
    );
    console.log(
        `  water/sky luma quartiles ${reading.duskBalance.waterSkyLumaRatios.map((value) => value.toFixed(3)).join(", ")}  fade 10→90 ${reading.duskBalance.fadeExtent}px  near B/R ${reading.duskBalance.nearBlueRedRatio.toFixed(3)}`,
    );
    console.log(
        `  trough foam coverage ${(reading.foam.nearCoverage * 100).toFixed(2)}%  max luma ${(reading.foam.maxLuma * 255).toFixed(1)}`,
    );
    console.log(
        `  normal response mid=${reading.normalResponse.midRowDeviation.toFixed(1)} near=${reading.normalResponse.nearRowDeviation.toFixed(1)} chroma=${reading.normalResponse.nearMeanChroma.toFixed(1)} weightedHue=${reading.normalResponse.chromaWeightedWaterSkyHueDistance.toFixed(1)}° bright=${(reading.normalResponse.brightWaterFraction * 100).toFixed(2)}%`,
    );
}

if (import.meta.main) {
    const regenerateIndex = process.argv.indexOf("--regenerate");
    if (regenerateIndex >= 0) {
        const sourceRoot = resolve(
            process.argv[regenerateIndex + 1] ?? "../scratch/water-surface/oracle-dusk/frames",
        );
        const paths = ["t26.jpg", "t43.jpg"].map((name) => resolve(sourceRoot, name));
        if (paths.some((path) => !existsSync(path))) {
            console.log(
                `SKIP regeneration: optional dusk source images absent under ${sourceRoot}`,
            );
            process.exit(0);
        }
        const readings = {} as Record<"t26" | "t43", LookReading>;
        for (const [index, path] of paths.entries()) {
            const name = index === 0 ? "t26" : "t43";
            const source = fixture.provenance.sources.find(({ name: value }) =>
                value.endsWith(`${name}.jpg`),
            );
            const digest = createHash("sha256")
                .update(await readFile(path))
                .digest("hex");
            if (digest !== source?.sha256)
                throw new Error(`${name} source digest differs from provenance`);
            readings[name] = analyze(await load(path));
            print(path, readings[name]);
        }
        printBounds();
        assertRegeneratedReferences(readings);
        console.log("PASS regeneration: source readings equal committed fixture");
        process.exit(0);
    }
    const capture = process.argv[2];
    if (!capture)
        throw new Error(
            "usage: bun look.oracle.ts <capture.png> [--condition sun-facing] [--compare path] [--prior path] [--regenerate source-dir]",
        );
    const compareIndex = process.argv.indexOf("--compare");
    const priorIndex = process.argv.indexOf("--prior");
    const conditionIndex = process.argv.indexOf("--condition");
    const compare = compareIndex >= 0 ? process.argv[compareIndex + 1] : undefined;
    const prior = priorIndex >= 0 ? process.argv[priorIndex + 1] : undefined;
    const condition = conditionIndex >= 0 ? process.argv[conditionIndex + 1] : undefined;
    let ok = true;
    const captureImage = await load(capture);
    const captureReading = analyze(captureImage);
    const baseline = analyze(
        await load(resolve("examples/showcase/ocean/test/baseline/ocean-baseline-1.png")),
    );
    print(capture, captureReading);
    if (
        captureReading.lumaRange < 0.02 ||
        captureReading.horizon.transitionWidth < 1 ||
        captureReading.horizon.continuity <= 0
    )
        ok = false;
    if (condition !== undefined && condition !== "sun-facing")
        throw new Error(`unknown look-oracle condition: ${condition}`);
    printBounds();
    ok &&= satisfiesReferenceRelations(captureReading);
    console.log(
        `  reference relations ${ok ? "PASS" : "FAIL"}: hue=${captureReading.farWaterSkyHueDistance.toFixed(1)} chroma=${captureReading.normalResponse.nearMeanChroma.toFixed(1)} fade=${captureReading.duskBalance.fadeExtent}px`,
    );
    if (prior) {
        const priorReading = analyze(await load(prior));
        const baselineNear = await Promise.all(
            [1, 2, 3].map(async (index) =>
                analyze(
                    await load(
                        resolve(
                            `examples/showcase/ocean/test/baseline/ocean-baseline-${index}.png`,
                        ),
                    ),
                ),
            ),
        );
        const nearFloor =
            Math.max(...baselineNear.map((reading) => reading.bands.nearWater!.luma)) -
            Math.min(...baselineNear.map((reading) => reading.bands.nearWater!.luma));
        console.log(
            `  foam treatment coverage capture=${(captureReading.foam.nearCoverage * 100).toFixed(3)}% prior=${(priorReading.foam.nearCoverage * 100).toFixed(3)}%`,
        );
        console.log(
            `  S5 mid-band floor capture=${captureReading.bands.midWater!.luma.toFixed(6)} prior=${priorReading.bands.midWater!.luma.toFixed(6)} tolerance=${nearFloor.toFixed(6)}`,
        );
        ok &&=
            Math.abs(captureReading.bands.midWater!.luma - priorReading.bands.midWater!.luma) <=
            nearFloor;
        for (const band of ["sky", "horizon", "farWater"] as const) {
            const floor =
                Math.max(...baselineNear.map((reading) => reading.bands[band]!.luma)) -
                Math.min(...baselineNear.map((reading) => reading.bands[band]!.luma));
            const delta = Math.abs(
                captureReading.bands[band]!.luma - priorReading.bands[band]!.luma,
            );
            console.log(
                `  untouched ${band} delta=${delta.toFixed(6)} S1Floor=${floor.toFixed(6)}`,
            );
            ok &&= delta <= floor;
        }
    }
    console.log(
        `  foam gate coverage=${(captureReading.foam.nearCoverage * 100).toFixed(2)}% max=${(captureReading.foam.maxLuma * 255).toFixed(1)} sky=${(captureReading.bands.sky!.luma * 255).toFixed(1)}`,
    );
    console.log(
        `  near-band specks capture=${captureReading.lowerBandBrightSpecks} baseline=${baseline.lowerBandBrightSpecks} dusk=${fixture.references.t26.lowerBandBrightSpecks},${fixture.references.t43.lowerBandBrightSpecks}`,
    );
    if (compare) {
        const overlap = speckOverlap(captureImage, await load(compare));
        console.log(`  t=6 versus t=6.05 speck overlap ${overlap.toFixed(4)}`);
    }
    if (!ok) {
        console.error("FAIL: zero-contrast or discontinuous horizon input");
        process.exitCode = 1;
    }
}
