// Interleaved kernel comparison. Each timing sample is a fresh Node process.
// bun diagnostics/box3d-parity/ab.ts <scene> <threads comma list> <from> <to exclusive> <rounds> <label=path> <label=path> ...
// Paths are built JS bundles, kernel.wasm.ts artifacts, or directories containing both kernel artifacts.
// Hashes are checked over the entire replay once per variant/thread cell; timings never hash.
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { PROFILE_FIELDS } from "../../src/standard/physics/world/profile";

const [scene, threadArg, fromArg, toArg, roundsArg, ...variantArgs] = process.argv.slice(2);
const from = Number(fromArg), to = Number(toArg), rounds = Number(roundsArg);
const threads = (threadArg ?? "").split(",").map(Number);
if (!scene || !Number.isInteger(from) || from < 0 || !Number.isInteger(to) || to <= from ||
    !Number.isInteger(rounds) || rounds < 2 || threads.some(t => !Number.isInteger(t) || t < 1) ||
    variantArgs.length < 2 || variantArgs.some(v => !v.includes("="))) {
    throw new Error("usage: ab.ts <scene> <threads comma list> <from> <to exclusive> <rounds >=2> <label=bundle.js|artifact-directory> ...");
}
const median = (values: number[]) => {
    const s = [...values].sort((a, b) => a - b);
    return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const range = (values: number[]) => ({ median: median(values), lo: Math.min(...values), hi: Math.max(...values) });
const display = (values: number[]) => {
    const r = range(values);
    return `${r.median.toFixed(3)} [${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}]`;
};
const dir = mkdtempSync(join(tmpdir(), "box3d-ab-"));
const start = performance.now();
try {
    const variants = await Promise.all(variantArgs.map(async (arg, i) => {
        const equals = arg.indexOf("=");
        const label = arg.slice(0, equals), path = resolve(arg.slice(equals + 1));
        if (!label || !existsSync(path)) throw new Error(`invalid variant: ${arg}`);
        if (/\.m?js$/.test(path)) return { label, bundle: path };
        const artifacts = path.endsWith("kernel.wasm.ts") ? dirname(path) : path;
        for (const file of ["kernel.wasm.ts", "kernel.shared.wasm.ts"])
            if (!existsSync(join(artifacts, file))) throw new Error(`missing artifact: ${join(artifacts, file)}`);
        const outdir = join(dir, String(i));
        const built = await Bun.build({
            entrypoints: [join(import.meta.dir, "scenes.ts")], outdir, target: "node", format: "esm",
            plugins: [{ name: "kernel-variant", setup(build) {
                build.onResolve({ filter: /kernel\.(shared\.)?wasm$/ }, args => ({ path: join(artifacts, `${args.path.split("/").at(-1)}.ts`) }));
            } }],
        });
        if (!built.success) throw new Error(built.logs.join("\n"));
        return { label, bundle: join(outdir, "scenes.js") };
    }));
    if (new Set(variants.map(v => v.label)).size !== variants.length) throw new Error("variant labels must be unique");
    for (const thread of threads) {
        const run = (bundle: string, hashSteps: boolean) => {
            const sampleStart = performance.now();
            const p = Bun.spawnSync(["node", bundle, scene, String(thread), String(to)], {
                env: { ...process.env, PROFILE: String(from), COUNTERS: "-1", CPU: "-1", WALL: "-1", HASH_STEPS: hashSteps ? "1" : "0", SAMPLE_WALL: "1" },
            });
            const wall = performance.now() - sampleStart;
            if (p.exitCode !== 0) throw new Error(p.stderr.toString());
            const lines = p.stdout.toString().trim().split("\n");
            const rows = lines.filter(l => l.startsWith("F ")).map(l => l.split(" ").slice(2).map(Number));
            if (rows.length !== to - from || rows.some(r => r.length !== PROFILE_FIELDS.length || r.some(x => !Number.isFinite(x))))
                throw new Error("missing or invalid phase rows");
            const measurement = lines.find(l => l.startsWith("M "));
            if (!measurement) throw new Error("bundle must support SAMPLE_WALL and HASH_STEPS; rebuild with current scenes.ts");
            const m = JSON.parse(measurement.slice(2)) as Record<string, number>;
            return { wall, hashes: lines.filter(l => /^\d+ 0x/.test(l)), phases: PROFILE_FIELDS.map((_, k) => median(rows.map(r => r[k]))), costs: { startup: m.setup + wall - Object.values(m).reduce((a, b) => a + b, 0), before: m.before, hashing: m.hashing, window: m.window, other: m.other } };
        };
        const hashes = variants.map(v => run(v.bundle, true).hashes);
        if (hashes.some(h => h.length !== to)) throw new Error("missing replay hashes");
        if (hashes.some(h => h.join("\n") !== hashes[0].join("\n"))) throw new Error(`${scene} ${thread}t hashes DIFFER`);
        console.log(`${scene} ${thread}t [${from}, ${to}) ${rounds} rounds: hashes equal; ms median [min, max] of sample medians`);
        const samples = variants.map(() => [] as ReturnType<typeof run>[]);
        for (let round = 0; round < rounds; ++round) {
            const order = variants.map((_, i) => i);
            if (round % 2) order.reverse();
            for (const i of order) {
                const result = run(variants[i].bundle, false);
                if (result.hashes.length) throw new Error("timing bundle did not disable hashing");
                samples[i].push(result);
            }
        }
        for (let k = 0; k < PROFILE_FIELDS.length; ++k) {
            const values = samples.map(s => s.map(r => r.phases[k]));
            const ranges = values.map(range);
            const pairs = variants.flatMap((v, i) => variants.slice(i + 1).map((w, j) => {
                const a = ranges[i], b = ranges[i + j + 1];
                return `${v.label}/${w.label} ${a.hi < b.lo || b.hi < a.lo ? "disjoint" : "overlap"}`;
            }));
            console.log(`${PROFILE_FIELDS[k]}: ${variants.map((v, i) => `${v.label} ${display(values[i])}`).join("; ")}; ${pairs.join("; ")}`);
        }
        for (const [i, v] of variants.entries()) {
            console.log(`${v.label} sample wall ms: ${display(samples[i].map(s => s.wall))}`);
            for (const key of ["startup", "before", "hashing", "window", "other"])
                console.log(`  ${key}: ${display(samples[i].map(s => s.costs[key]))}`);
        }
        console.log("startup includes imports, initialization, scene construction and process exit; before/window include scene callbacks and stepping; other includes profile formatting.");
    }
} finally {
    rmSync(dir, { recursive: true, force: true });
    console.log(`whole run wall: ${((performance.now() - start) / 1000).toFixed(3)} s (includes builds and hash checks)`);
}
