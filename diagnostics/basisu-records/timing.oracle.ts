import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { bunChild } from "../../scripts/bun-child";
import { openBasisImage, type BasisTarget } from "../../src/extras/gltf/basisu";
import inputs from "../../crates/basisu/tests/records/inputs.json";
import { loadReference, referenceDirectory } from "./reference";

const root = resolve(import.meta.dir, "../..");
referenceDirectory();
if (!process.argv.includes("--child")) {
    const sync = Bun.spawnSync({ ...bunChild([resolve(import.meta.dir, "records.ts")]), cwd: root, stdout: "pipe", stderr: "pipe" });
    if (sync.exitCode !== 0) throw new Error(new TextDecoder().decode(sync.stderr));
    console.log("Reference: Binomial webgl/transcoder/build/basis_transcoder.{js,wasm}; 32c7b74 built 3784d52; transcoder inputs unchanged at 073eb6a0");
    for (let i = 0; i < 3; i++) {
        const result = Bun.spawnSync({ ...bunChild([import.meta.path, "--child", String(i)]), cwd: root, stdout: "inherit", stderr: "inherit" });
        if (result.exitCode !== 0) throw new Error(`timing process ${i + 1} failed`);
    }
} else {
    const module = await loadReference();
    const first = Number(process.argv.at(-1));
    for (const model of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
        const files = inputs.files.filter((f) => f.path.startsWith(`${model}/`)).map((file) => {
            const bytes = readFileSync(resolve(root, "test-results/basisu-assets", file.path));
            if (createHash("sha256").update(bytes).digest("hex") !== file.sha256) throw new Error(`${file.path}: digest mismatch`);
            const own = openBasisImage(bytes);
            const ref = new module.KTX2File(bytes);
            const target: BasisTarget = own.model === "uastc" ? "astc" : ref.getHasAlpha() ? "etc2-rgba" : "etc1";
            const levels = own.levels;
            if (levels !== ref.getLevels()) throw new Error("level count mismatch");
            own.dispose(); ref.close(); ref.delete();
            return { bytes, target, levels, format: target === "astc" ? 10 : target === "etc2-rgba" ? 1 : 0 };
        });
        const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
        for (const file of files) {
            const own = openBasisImage(file.bytes);
            const ref = new module.KTX2File(file.bytes);
            if (!ref.startTranscoding()) throw new Error("reference startTranscoding failed");
            try {
                for (let level = 0; level < file.levels; level++) {
                    const out = new Uint8Array(ref.getImageTranscodedSizeInBytes(level, 0, 0, file.format));
                    if (!ref.transcodeImage(out, level, 0, 0, file.format, 0, -1, -1)) throw new Error("reference transcode failed");
                    if (digest(own.transcode(level, file.target)) !== digest(out)) throw new Error(`${model} level ${level}: WASM reference mismatch`);
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
                const image = new module.KTX2File(file.bytes);
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
        // Like meshopt-records/timing.ts, alternate warmup/calibration/sampling order.
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
        for (let round = 0; round < 8; round++) {
            const a = (first + round) & 1;
            samples[a].push(batch(a, iterations)); samples[1-a].push(batch(1-a, iterations));
        }
        const medians = samples.map((values) => { values.sort((a,b) => a-b); return (values[3]+values[4])/2; });
        console.log(JSON.stringify({ process: first+1, model, images: files.length, levels: files.reduce((n,f) => n+f.levels,0), ownMs: medians[0], referenceMs: medians[1], ratio: medians[0]/medians[1], observed }));
    }
}
