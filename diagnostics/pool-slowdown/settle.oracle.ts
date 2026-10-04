// Manual oracle for the pool's step-count slowdown (strategy unit pool-slowdown, stage 1). Run by path:
//
//     bun test ./diagnostics/pool-slowdown/settle.oracle.ts
//
// Bundles pyramid.entry.ts once per variant, each with `kernel/pool.ts` patched in the bundle only, and
// runs the variants interleaved as child processes on the same host. Reports per-variant early and late
// step times; timings are reported, never asserted. Knobs: POOL_STEPS (2000), POOL_REPEATS (2),
// POOL_THREADS (4), POOL_VARIANTS (comma list of the names below), POOL_RUNTIME (bun or node).
import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const JOIN = "            while (Atomics.load(ctlView, CTL_DONE) < count) {}";
const ACK = "        Atomics.add(ctl, ${CTL_DONE}, 1);\n    }";
const WORDS = "const CTL_WORDS = 4;";

/** Each variant rewrites pool.ts source text, `[from, to]`, and nothing else. `single` runs the
 * unchanged pool at threads 0. */
const VARIANTS: Record<string, [string, string][]> = {
    unchanged: [],
    single: [],
    // The TC39 spin-wait hint each pass.
    pause: [[JOIN, "            while (Atomics.load(ctlView, CTL_DONE) < count) Atomics.pause();"]],
    // A clock read each pass, as Emscripten's main-thread futex wait does; the comparison keeps it live.
    now: [[JOIN, "            while (Atomics.load(ctlView, CTL_DONE) < count) if (performance.now() < 0) break;"]],
    // About the clock read's ~30 ns per pass (Bun 1.4.2, M4 Max) of register-only arithmetic, no call and
    // no clock: separates the clock from the spacing between loads. The result is kept live after the loop.
    spaced: [
        [
            JOIN,
            "            let spacer = 0;\n            while (Atomics.load(ctlView, CTL_DONE) < count) for (let i = 0; i < 5; i++) spacer = (spacer * 31 + i) | 0;\n            if (spacer === 1) Atomics.store(ctlView, CTL_OP, OP_SOLVE);",
        ],
    ],
    // Workers ack with a plain atomic store to a cache line of their own instead of an `Atomics.add` on
    // the word the orchestrator spins on; the orchestrator spins on each line in turn.
    store: [
        [WORDS, "const CTL_WORDS = 16 * 9;"],
        [ACK, "        Atomics.store(ctl, 16 * (d.index + 1), seen);\n    }"],
        [JOIN, "            for (let i = 1; i <= count; i++) while (Atomics.load(ctlView, 16 * i) !== seq) {}"],
    ],
};

const steps = Number(process.env.POOL_STEPS ?? 2000);
const repeats = Number(process.env.POOL_REPEATS ?? 2);
const threads = Number(process.env.POOL_THREADS ?? 4);
const runtime = process.env.POOL_RUNTIME ?? "bun";
const names = (process.env.POOL_VARIANTS ?? "unchanged,pause,now,spaced,store").split(",");

const median = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
const quantile = (a: number[], q: number) => [...a].sort((x, y) => x - y)[Math.floor(a.length * q)];
const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;

test("report step times per join variant, interleaved", async () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-pool-slowdown-"));
    try {
        for (const name of names) {
            const patches = VARIANTS[name];
            if (patches === undefined) throw new Error(`unknown variant ${name}`);
            const built = await Bun.build({
                entrypoints: [resolve(import.meta.dir, "pyramid.entry.ts")],
                outdir: dir,
                target: "node",
                format: "esm",
                naming: `${name}.mjs`,
                plugins: [
                    {
                        name: "pool-join",
                        setup(build) {
                            build.onLoad({ filter: /kernel\/pool\.ts$/ }, async ({ path }) => {
                                let source = await Bun.file(path).text();
                                for (const [from, to] of patches) {
                                    if (!source.includes(from)) throw new Error(`pool.ts moved: ${from}`);
                                    source = source.replace(from, to);
                                }
                                return { contents: source, loader: "ts" };
                            });
                        },
                    },
                ],
            });
            expect(built.success).toBe(true);
        }
        const rows: string[] = [];
        for (let r = 0; r < repeats; r++)
            for (const name of names) {
                const proc = Bun.spawn(
                    [runtime, join(dir, `${name}.mjs`), String(name === "single" ? 0 : threads), String(steps)],
                    { stdout: "pipe", stderr: "inherit" },
                );
                const out = await new Response(proc.stdout).text();
                expect(await proc.exited).toBe(0);
                const { step, collide } = JSON.parse(out.trim().split("\n").at(-1) as string) as {
                    step: number[];
                    collide: number[];
                };
                const early = step.slice(100, 400);
                const late = step.slice(800);
                const lateCollide = collide.slice(800);
                rows.push(
                    `| ${name} | ${r + 1} | ${median(early).toFixed(2)} | ${median(late).toFixed(2)} | ${mean(late).toFixed(2)} | ${quantile(late, 0.9).toFixed(2)} | ${Math.round((100 * lateCollide.filter((c) => c > 1).length) / lateCollide.length)}% |`,
                );
            }
        console.info(
            `[pool-slowdown] ${runtime} ${runtime === "bun" ? Bun.version : ""}, ${threads} threads, ${steps} steps; early = steps 100-399, late = 800 on (ms)\n` +
                "| join | run | early median | late median | late mean | late p90 | late collide > 1 ms |\n|---|---|---|---|---|---|---|\n" +
                rows.join("\n"),
        );
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}, 0);
