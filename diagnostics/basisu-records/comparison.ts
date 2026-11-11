import { openBasisImage, type BasisTarget } from "../../src/extras/gltf/basisu";
import type { ReferenceImage } from "./reference";

export interface Reference {
    KTX2File: new (bytes: Uint8Array) => ReferenceImage;
}
export interface Input {
    path: string;
    bytes: Uint8Array<ArrayBuffer>;
    plain?: Uint8Array<ArrayBuffer>;
}
export function modelFiles(reference: Reference, inputs: Input[], model: string) {
    return inputs.filter((file) => file.path.startsWith(`${model}/`)).map((file) => {
        const own = openBasisImage(file.bytes);
        const ref = new reference.KTX2File(file.bytes);
        try {
            const target: BasisTarget = own.model === "uastc" ? "astc" : ref.getHasAlpha() ? "etc2-rgba" : "etc1";
            if (own.levels !== ref.getLevels()) throw new Error("level count mismatch");
            return { ...file, target, levels: own.levels, format: target === "astc" ? 10 : target === "etc2-rgba" ? 1 : 0 };
        } finally { own.dispose(); ref.close(); ref.delete(); }
    });
}
export function equal(a: Uint8Array, b: Uint8Array): boolean {
    return a.length === b.length && a.every((value, index) => value === b[index]);
}
// meshoptimizer tools/codecbench.cpp keeps maximum throughput over attempts.
// With fixed work, that is minimum elapsed time, not a contention-heavy median.
export const SAMPLE_BATCHES = 256;

export function timing(reference: Reference, inputs: Input[], first: number) {
    for (const model of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
        const files = modelFiles(reference, inputs, model);
        for (const file of files) {
            const own = openBasisImage(file.bytes);
            const ref = new reference.KTX2File(file.bytes);
            if (!ref.startTranscoding()) throw new Error("reference startTranscoding failed");
            try {
                for (let level = 0; level < file.levels; level++) {
                    const out = new Uint8Array(ref.getImageTranscodedSizeInBytes(level, 0, 0, file.format));
                    if (!ref.transcodeImage(out, level, 0, 0, file.format, 0, -1, -1)) throw new Error("reference transcode failed");
                    if (!equal(own.transcode(level, file.target), out)) throw new Error(`${model} level ${level}: WASM reference mismatch`);
                }
            } finally { own.dispose(); ref.close(); ref.delete(); }
        }
        let observed = 0;
        const own = () => {
            for (const file of files) {
                const image = openBasisImage(file.bytes);
                try { for (let level = 0; level < file.levels; level++) observed ^= image.transcode(level, file.target)[0]; }
                finally { image.dispose(); }
            }
        };
        const ref = () => {
            for (const file of files) {
                const image = new reference.KTX2File(file.bytes);
                try {
                    if (!image.startTranscoding()) throw new Error("reference startTranscoding failed");
                    for (let level = 0; level < file.levels; level++) {
                        const out = new Uint8Array(image.getImageTranscodedSizeInBytes(level, 0, 0, file.format));
                        if (!image.transcodeImage(out, level, 0, 0, file.format, 0, -1, -1)) throw new Error("reference transcode failed");
                        observed ^= out[0];
                    }
                } finally { image.close(); image.delete(); }
            }
        };
        // Full-model batches include input/output copies, KTX2 setup and Zstandard.
        const functions = [own, ref];
        for (let round = 0; round < 4; round++) { functions[(first + round) & 1](); functions[(first + round + 1) & 1](); }
        function batch(index: number, iterations: number) {
            const start = performance.now();
            for (let i = 0; i < iterations; i++) functions[index]();
            return (performance.now() - start) / iterations;
        }
        let iterations = 1;
        for (let round = 0; ; round++) {
            const a = (first + round) & 1;
            const one = batch(a, iterations);
            const two = batch(1 - a, iterations);
            if (Math.min(one, two) * iterations >= 100) break;
            iterations *= 2;
        }
        const samples: number[][] = [[], []];
        for (let round = 0; round < SAMPLE_BATCHES; round++) {
            const a = (first + round) & 1;
            samples[a].push(batch(a, iterations)); samples[1-a].push(batch(1-a, iterations));
        }
        const minima = samples.map((values) => Math.min(...values));
        const firstHalf = samples.map((values) => Math.min(...values.slice(0, SAMPLE_BATCHES / 2)));
        console.log(JSON.stringify({ process: first+1, model, images: files.length, levels: files.reduce((n,f) => n+f.levels,0), statistic: "minimum", batches: SAMPLE_BATCHES, iterations, ownFirstHalfMs: firstHalf[0], referenceFirstHalfMs: firstHalf[1], ownMs: minima[0], referenceMs: minima[1], ratio: minima[0]/minima[1], observed }));
    }
}
