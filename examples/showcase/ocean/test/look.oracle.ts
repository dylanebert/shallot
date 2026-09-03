import { readFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { decode as decodeJpeg } from "jpeg-js";
import { PNG } from "pngjs";

export interface Pixels {
    width: number;
    height: number;
    data: Uint8Array;
}

export interface LookReading {
    bands: Record<string, { srgb: [number, number, number]; luma: number }>;
    farWaterSkyHueDistance: number;
    lowerBandBrightSpecks: number;
    horizon: { transitionWidth: number; continuity: number };
    lumaRange: number;
}

const BAND_RANGES = {
    sky: [0.08, 0.3],
    horizon: [0.38, 0.52],
    farWater: [0.52, 0.66],
    midWater: [0.66, 0.82],
    nearWater: [0.82, 1],
} as const;

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

function horizon(image: Pixels): { transitionWidth: number; continuity: number } {
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
    let lo = peak.y;
    let hi = peak.y;
    while (lo > 1 && (gradients[lo - 1]?.value ?? 0) >= cutoff) lo--;
    while (hi < gradients.length && (gradients[hi]?.value ?? 0) >= cutoff) hi++;
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
    return { transitionWidth: hi - lo + 1, continuity: 1 / (1 + deviation) };
}

export function analyze(image: Pixels): LookReading {
    const bands = Object.fromEntries(
        Object.entries(BAND_RANGES).map(([name, [from, to]]) => [name, mean(image, from, to)]),
    );
    const skyHue = hue(bands.sky?.srgb ?? [0, 0, 0]);
    const waterHue = hue(bands.farWater?.srgb ?? [0, 0, 0]);
    const hueDelta = Math.abs(skyHue - waterHue);
    const allLuma = Object.values(bands).map((band) => band.luma);
    return {
        bands,
        farWaterSkyHueDistance: Math.min(hueDelta, 360 - hueDelta),
        lowerBandBrightSpecks: brightSpecks(image),
        horizon: horizon(image),
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
        `  horizon transition ${reading.horizon.transitionWidth}px  continuity ${reading.horizon.continuity.toFixed(3)}`,
    );
}

if (import.meta.main) {
    const capture = process.argv[2];
    if (!capture) throw new Error("usage: bun look.oracle.ts <capture.png> [t26.jpg] [t43.jpg]");
    const compareIndex = process.argv.indexOf("--compare");
    const priorIndex = process.argv.indexOf("--prior");
    const compare = compareIndex >= 0 ? process.argv[compareIndex + 1] : undefined;
    const prior = priorIndex >= 0 ? process.argv[priorIndex + 1] : undefined;
    const references = [
        resolve("../research/water-surface/oracle-dusk/frames/t26.jpg"),
        resolve("../research/water-surface/oracle-dusk/frames/t43.jpg"),
    ];
    let ok = true;
    const captureImage = await load(capture);
    const captureReading = analyze(captureImage);
    const baseline = analyze(
        await load(resolve("examples/showcase/ocean/test/baseline/ocean-baseline-1.png")),
    );
    const referenceReadings = await Promise.all(
        references.map(async (path) => analyze(await load(path))),
    );
    for (const [path, reading] of [
        [capture, captureReading] as const,
        ...references.map((path, i) => [path, referenceReadings[i]!] as const),
    ]) {
        print(path, reading);
        if (
            reading.lumaRange < 0.02 ||
            reading.horizon.transitionWidth < 1 ||
            reading.horizon.continuity <= 0
        )
            ok = false;
    }
    const referenceSpecks = referenceReadings.map((reading) => reading.lowerBandBrightSpecks);
    const referenceWidths = referenceReadings.map((reading) => reading.horizon.transitionWidth);
    const referenceContinuity = referenceReadings.map((reading) => reading.horizon.continuity);
    const referenceHueDistance = referenceReadings.map((reading) => reading.farWaterSkyHueDistance);
    const widthMin = Math.min(...referenceWidths) * 0.5;
    const widthMax = Math.max(...referenceWidths) * 1.5;
    console.log(
        `  aerial gate width=${captureReading.horizon.transitionWidth} referenceRange=${widthMin}..${widthMax} continuity=${captureReading.horizon.continuity.toFixed(3)} referenceFloor=${Math.min(...referenceContinuity).toFixed(3)} hue=${captureReading.farWaterSkyHueDistance.toFixed(1)} referenceMax=${Math.max(...referenceHueDistance).toFixed(1)}`,
    );
    ok &&=
        captureReading.horizon.transitionWidth >= widthMin &&
        captureReading.horizon.transitionWidth <= widthMax;
    ok &&= captureReading.horizon.continuity >= Math.min(...referenceContinuity);
    ok &&= captureReading.farWaterSkyHueDistance <= Math.max(...referenceHueDistance);
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
            `  S4 near-band floor capture=${captureReading.bands.nearWater!.luma.toFixed(6)} prior=${priorReading.bands.nearWater!.luma.toFixed(6)} tolerance=${nearFloor.toFixed(6)}`,
        );
        ok &&=
            captureReading.bands.nearWater!.luma >= priorReading.bands.nearWater!.luma - nearFloor;
    }
    const referenceMin = Math.min(...referenceSpecks) / 10;
    const referenceMax = Math.max(...referenceSpecks) * 10;
    console.log(
        `  near-band speck gate capture=${captureReading.lowerBandBrightSpecks} baseline=${baseline.lowerBandBrightSpecks} dusk=${referenceSpecks.join(",")}`,
    );
    ok &&= captureReading.lowerBandBrightSpecks > baseline.lowerBandBrightSpecks;
    ok &&=
        captureReading.lowerBandBrightSpecks >= referenceMin &&
        captureReading.lowerBandBrightSpecks <= referenceMax;
    if (compare) {
        const overlap = speckOverlap(captureImage, await load(compare));
        console.log(`  t=6 versus t=6.05 speck overlap ${overlap.toFixed(4)}`);
    }
    if (!ok) {
        console.error("FAIL: zero-contrast or discontinuous horizon input");
        process.exitCode = 1;
    }
}
