import type { State } from "@dylanebert/shallot";
import { Sky } from "../sky";
import { SOLAR_ANGULAR_RADIUS } from "../sky/shader";

interface Check {
    name: string;
    pass: boolean;
    detail?: string;
}

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

function horizon(data: Uint8ClampedArray, width: number, height: number): number {
    const means: number[] = [];
    const x0 = Math.floor(width * 0.1);
    const x1 = Math.floor(width * 0.9);
    for (let y = 0; y < height; y++) {
        let sum = 0;
        for (let x = x0; x < x1; x++) sum += luma(data, (y * width + x) * 4);
        means.push(sum / (x1 - x0));
    }
    let row = Math.floor(height * 0.25);
    let peak = 0;
    for (let y = row + 1; y <= height * 0.7; y++) {
        const gradient = Math.abs((means[y] ?? 0) - (means[y - 1] ?? 0));
        if (gradient > peak) {
            peak = gradient;
            row = y;
        }
    }
    return row;
}

export async function runDeviceClaim(state: State): Promise<Check[]> {
    state.pause();
    const canvas = document.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement))
        return [{ name: "solar ablation canvas is available", pass: false }];
    const sky = state.only([Sky]);
    const strength = Sky.sunStrength.get(sky);
    Sky.sunStrength.set(sky, strength);
    state.step(0);
    const declared = await pixels(canvas);
    Sky.sunStrength.set(sky, 0);
    state.step(0);
    const zero = await pixels(canvas);
    Sky.sunStrength.set(sky, strength);

    const row = horizon(zero, canvas.width, canvas.height);
    const delta = new Float64Array(canvas.width * canvas.height);
    let peak = 0;
    let peakIndex = 0;
    let waterDelta = 0;
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const pixel = y * canvas.width + x;
            const value = Math.max(0, luma(declared, pixel * 4) - luma(zero, pixel * 4)) * 255;
            delta[pixel] = value;
            if (y < row && value > peak) {
                peak = value;
                peakIndex = pixel;
            }
            if (y >= row) waterDelta += value;
        }
    }
    const active = new Uint8Array(delta.length);
    for (let i = 0; i < row * canvas.width; i++) if ((delta[i] ?? 0) >= 1 / 255) active[i] = 1;
    const stack = [peakIndex];
    active[peakIndex] = 0;
    let minX = canvas.width;
    let maxX = 0;
    let minY = canvas.height;
    let maxY = 0;
    const radii: number[][] = [[], [], []];
    const cx = peakIndex % canvas.width;
    const cy = Math.floor(peakIndex / canvas.width);
    while (stack.length) {
        const at = stack.pop()!;
        const x = at % canvas.width;
        const y = Math.floor(at / canvas.width);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        const radius = Math.hypot(x - cx, y - cy);
        radii[Math.min(2, Math.floor(radius))]!.push(delta[at] ?? 0);
        for (const next of [at - canvas.width, at + canvas.width, at - 1, at + 1]) {
            if (next < 0 || next >= row * canvas.width || !active[next]) continue;
            if ((next === at - 1 && x === 0) || (next === at + 1 && x === canvas.width - 1))
                continue;
            active[next] = 0;
            stack.push(next);
        }
    }
    const coreWidth = maxX - minX + 1;
    const coreHeight = maxY - minY + 1;
    const expected = (2 * SOLAR_ANGULAR_RADIUS * canvas.height * 180) / (Math.PI * 60);
    const radial = radii.map((ring) => ring.reduce((sum, value) => sum + value, 0) / ring.length);
    const monotone = radial.every((value, index) => index === 0 || value <= radial[index - 1]!);
    return [
        {
            name: "declared solar strength is nonzero",
            pass: strength > 0,
            detail: `strength=${strength}`,
        },
        {
            name: "solar sky delta is one angularly bounded monotone core",
            pass:
                Math.abs(coreWidth - expected) <= 1 &&
                Math.abs(coreHeight - expected) <= 1 &&
                monotone,
            detail: `core=${coreWidth}x${coreHeight} expected=${expected.toFixed(2)} radial=${radial.join(",")} horizon=${row}`,
        },
        {
            name: "solar disk is brighter than the sky column gradient",
            pass: peak > 3.8,
            detail: `peakDelta=${peak} gradient=3.8`,
        },
        {
            name: "reflected sky disk contributes positive water energy",
            pass: waterDelta > 0,
            detail: `waterDelta=${waterDelta} elapsed=${state.time.elapsed}`,
        },
    ];
}
