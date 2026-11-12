// bun diagnostics/gltf-pose/ab.ts <rounds|-> <label=gltf.wasm.ts> <label=gltf.wasm.ts> ...
// Each sample is a fresh Node process; paired rounds alternate variant order.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { shallot } from "@dylanebert/shallot/bun";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";

const [roundArg, ...args] = process.argv.slice(2);
const rounds = roundArg === "-" ? 12 : Number(roundArg);
if (!Number.isInteger(rounds) || rounds < 2 || args.length < 2 || args.some(a => !a.includes("=")))
    throw new Error("usage: ab.ts <rounds >=2|-> <label=gltf.wasm.ts> <label=gltf.wasm.ts> ... (- selects 12 rounds)");
const root = resolve(import.meta.dir, "../..");
const assets = join(root, "test-results/skinning-assets");
const file = {
    path: "Models/CesiumMan/glTF-Binary/CesiumMan.glb",
    sha256: "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9",
};
await syncFiles([file], new URL(`${manifest.source}${manifest.commit}/`), assets);
const asset = join(assets, file.path);
const median = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted.length % 2 ? sorted[sorted.length >> 1] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
};
// Exact paired sign test, as box3d-parity/ab.ts: 2% effect band, alpha=.025 per direction.
function verdict(a: number[], b: number[]): string {
    const ratios = a.map((x, i) => b[i] / x);
    const tail = (wins: number) => {
        let mass = 2 ** -ratios.length, probability = 0;
        for (let k = 0; k <= ratios.length; k++) {
            if (k >= wins) probability += mass;
            mass *= (ratios.length - k) / (k + 1);
        }
        return probability;
    };
    if (tail(ratios.filter(x => x < 0.98).length) <= 0.025) return "faster";
    if (tail(ratios.filter(x => x > 1.02).length) <= 0.025) return "slower";
    return "no difference";
}
const dir = mkdtempSync(join(root, "test-results/gltf-pose-ab-"));
const start = performance.now();
try {
    const variants = await Promise.all(args.map(async (arg, i) => {
        const equals = arg.indexOf("=");
        const label = arg.slice(0, equals), path = resolve(arg.slice(equals + 1));
        if (!label || !existsSync(path) || !path.endsWith("gltf.wasm.ts")) throw new Error(`invalid artifact variant: ${arg}`);
        const outdir = join(dir, String(i));
        mkdirSync(outdir);
        const built = await Bun.build({
            entrypoints: [join(import.meta.dir, "subject.ts")], outdir, target: "node", format: "esm", naming: "subject.mjs", external: ["webgpu"],
            plugins: [
                { name: "gltf-variant", setup(build) {
                    build.onResolve({ filter: /gltf\.wasm$/ }, () => ({ path }));
                } },
                shallot({ root }),
            ],
        });
        if (!built.success) throw new Error(built.logs.join("\n"));
        return { label, outdir, bundle: join(outdir, "subject.mjs") };
    }));
    if (new Set(variants.map(v => v.label)).size !== variants.length) throw new Error("variant labels must be unique");
    const run = (i: number, mode: "check" | "time") => {
        const v = variants[i];
        const sampleStart = performance.now();
        const p = Bun.spawnSync(["node", v.bundle, asset, mode, v.outdir]);
        if (p.exitCode !== 0) throw new Error(p.stderr.toString());
        return { lines: p.stdout.toString().trim().split("\n"), wall: performance.now() - sampleStart };
    };
    const checks = variants.map((_, i) => run(i, "check").lines.filter(l => l.startsWith("C ")));
    if (!checks[0].length || checks.some(c => c.join("\n") !== checks[0].join("\n"))) throw new Error("missing or mismatched conformance rows");
    let lanes = 0, maxError = 0;
    for (const line of checks[0]) {
        const [, frame, count] = line.split(" ");
        const arrays = variants.map(v => {
            const bytes = readFileSync(join(v.outdir, `${frame}.bin`));
            const array = new Float64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
            if (array.length !== Number(count)) throw new Error("invalid conformance output length");
            return array;
        });
        for (let i = 0; i < arrays[0].length; i++) {
            const expected = arrays[0][i];
            for (const [v, array] of arrays.entries()) {
                const error = Math.abs(array[i] - expected) / Math.max(1, Math.abs(expected));
                // examples/gltf/tests/animation.parity.node.ts's record comparator.
                if (!Number.isFinite(error) || error > 3e-5) throw new Error(`${variants[v].label} differs at frame ${frame}, lane ${i}: ${array[i]} vs ${expected}`);
                maxError = Math.max(maxError, error);
            }
        }
        lanes += arrays[0].length;
    }
    console.log(`CesiumMan 2560: globals, local poses/weights, palettes, morph weights and cull bounds agree; ${lanes} lanes per variant; max normalized error=${maxError}; tolerance=3e-5`);
    const samples = variants.map(() => [] as number[]), walls = variants.map(() => [] as number[]);
    for (let round = 0; round < rounds; round++) {
        const order = variants.map((_, i) => i);
        if (round % 2) order.reverse();
        for (const i of order) {
            const result = run(i, "time");
            const rows = result.lines.filter(l => l.startsWith("T "));
            if (rows.length !== 1) throw new Error("missing timing row");
            const row = JSON.parse(rows[0].slice(2)) as { median: number; calls: number };
            if (!Number.isFinite(row.median) || row.median <= 0 || row.calls !== 240 * 2560) throw new Error("invalid per-call timing");
            samples[i].push(row.median);
            walls[i].push(result.wall);
            console.log(`round=${round + 1} variant=${variants[i].label} per-call-us=${row.median.toFixed(3)} sample-wall-ms=${result.wall.toFixed(1)}`);
        }
    }
    for (const [i, v] of variants.entries())
        console.log(`${v.label}: per-call-us median=${median(samples[i]).toFixed(3)} min=${Math.min(...samples[i]).toFixed(3)} max=${Math.max(...samples[i]).toFixed(3)}; sample-wall-ms median=${median(walls[i]).toFixed(1)}`);
    for (let i = 0; i < variants.length; i++)
        for (let j = i + 1; j < variants.length; j++)
            console.log(`${variants[j].label} vs ${variants[i].label}: ${verdict(samples[i], samples[j])}; paired median ratio=${median(samples[i].map((x, k) => samples[j][k] / x)).toFixed(3)} (${rounds} rounds)`);
} finally {
    rmSync(dir, { recursive: true, force: true });
    console.log(`whole run wall: ${((performance.now() - start) / 1000).toFixed(3)} s (includes builds, setup and conformance checks)`);
}
