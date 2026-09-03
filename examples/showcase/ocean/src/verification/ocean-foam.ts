import type { State } from "@dylanebert/shallot";
import { declaredFoamStrength, writeFoamStrength } from "../ocean/surface";

interface Check {
    name: string;
    pass: boolean;
    detail?: string;
}

const HORIZON_ROW = 210;
const REFERENCE_COVERAGE = [0.0001, 0.1] as const;
const BAND_BUDGETS = {
    horizon: 0.00006670126553096067,
    far: 0.000015761223548821368,
    mid: 0.00011517122762150223,
} as const;

async function pixels(canvas: HTMLCanvasElement): Promise<Uint8ClampedArray> {
    const bitmap = await createImageBitmap(canvas);
    const copy = new OffscreenCanvas(canvas.width, canvas.height);
    const context = copy.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("2d capture context unavailable");
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    return context.getImageData(0, 0, canvas.width, canvas.height).data;
}

function luma(data: Uint8ClampedArray, index: number): number {
    return (
        (0.2126 * (data[index] ?? 0) +
            0.7152 * (data[index + 1] ?? 0) +
            0.0722 * (data[index + 2] ?? 0)) /
        255
    );
}

function bandDelta(
    declared: Uint8ClampedArray,
    zero: Uint8ClampedArray,
    width: number,
    from: number,
    to: number,
): number {
    let sum = 0;
    let count = 0;
    for (let y = from; y < to; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            sum += Math.abs(luma(declared, index) - luma(zero, index));
            count++;
        }
    }
    return sum / count;
}

export async function runDeviceClaim(state: State): Promise<Check[]> {
    state.pause();
    const canvas = document.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
        return [{ name: "foam ablation canvas is available", pass: false }];
    }
    const elapsed = state.time.elapsed;
    const strength = declaredFoamStrength();
    writeFoamStrength(strength);
    state.step(0);
    const declared = await pixels(canvas);
    writeFoamStrength(0);
    state.step(0);
    const zero = await pixels(canvas);
    writeFoamStrength(strength);

    let changed = 0;
    let changedAboveHorizon = 0;
    let brighterThanSky = 0;
    let skyLuma = 0;
    let skyCount = 0;
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const index = (y * canvas.width + x) * 4;
            if (y < HORIZON_ROW) {
                skyLuma += luma(declared, index);
                skyCount++;
            }
            const differs =
                declared[index] !== zero[index] ||
                declared[index + 1] !== zero[index + 1] ||
                declared[index + 2] !== zero[index + 2] ||
                declared[index + 3] !== zero[index + 3];
            if (!differs) continue;
            changed++;
            if (y < HORIZON_ROW) changedAboveHorizon++;
        }
    }
    const skyMean = skyLuma / skyCount;
    for (let index = 0; index < declared.length; index += 4) {
        const differs =
            declared[index] !== zero[index] ||
            declared[index + 1] !== zero[index + 1] ||
            declared[index + 2] !== zero[index + 2];
        if (differs && luma(declared, index) > skyMean) brighterThanSky++;
    }
    const waterPixels = canvas.width * (canvas.height - HORIZON_ROW);
    const coverage = changed / waterPixels;
    const horizon = bandDelta(
        declared,
        zero,
        canvas.width,
        HORIZON_ROW,
        Math.floor(0.52 * canvas.height),
    );
    const far = bandDelta(
        declared,
        zero,
        canvas.width,
        Math.floor(0.52 * canvas.height),
        Math.floor(0.66 * canvas.height),
    );
    const mid = bandDelta(
        declared,
        zero,
        canvas.width,
        Math.floor(0.66 * canvas.height),
        Math.floor(0.82 * canvas.height),
    );
    return [
        {
            name: "declared foam strength is nonzero",
            pass: strength > 0,
            detail: `strength=${strength}`,
        },
        {
            name: "same-state same-clock foam delta has reference-derived coverage",
            pass: coverage >= REFERENCE_COVERAGE[0] && coverage <= REFERENCE_COVERAGE[1],
            detail: `coverage=${coverage} reference=${REFERENCE_COVERAGE.join("..")} elapsed=${elapsed} after=${state.time.elapsed}`,
        },
        {
            name: "foam ablation leaves sky rows bit-identical",
            pass: changedAboveHorizon === 0,
            detail: `changedSkyPixels=${changedAboveHorizon} horizonRow=${HORIZON_ROW}`,
        },
        {
            name: "foam delta pixels do not exceed the sky band",
            pass: brighterThanSky === 0,
            detail: `brighter=${brighterThanSky} skyMean=${skyMean}`,
        },
        {
            name: "foam ablation preserves horizon far and mid budgets",
            pass:
                horizon <= BAND_BUDGETS.horizon &&
                far <= BAND_BUDGETS.far &&
                mid <= BAND_BUDGETS.mid,
            detail: `horizon=${horizon}/${BAND_BUDGETS.horizon} far=${far}/${BAND_BUDGETS.far} mid=${mid}/${BAND_BUDGETS.mid}`,
        },
    ];
}
