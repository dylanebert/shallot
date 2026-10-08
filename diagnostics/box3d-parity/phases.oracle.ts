// Manual oracle comparing Shallot's step phases with native Box3D's b3Profile on joint_grid, rain and
// junkyard at full size (strategy unit box3d-parity). Run by path, with a Box3D checkout at 47d7f7cc:
//
//     BOX3D=/path/to/box3d bun test ./diagnostics/box3d-parity/phases.oracle.ts
//
// Every step's hash and each measured step's contact, awake and joint counts must equal native.
// Cases run concurrently, each native then Shallot. PHASE_TIMING=1 runs sequentially with phase and
// main-thread CPU profiling (cpu.ts) and reports timings, never asserts them.
// PHASE_THREADS (comma list, "1,4") picks the thread counts.
import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary } from "./native";
import { WINDOWS } from "./windows";

setDefaultTimeout(300_000);

// b3Profile's fields in order (box3d/types.h), as both sides print them.
const FIELDS = [
    "step",
    "pairs",
    "collide",
    "solve",
    "solverSetup",
    "constraints",
    "prepareConstraints",
    "integrateVelocities",
    "warmStart",
    "solveImpulses",
    "integratePositions",
    "relaxImpulses",
    "applyRestitution",
    "storeImpulses",
    "splitIslands",
    "transforms",
    "sensorHits",
    "jointEvents",
    "hitEvents",
    "refit",
    "bullets",
    "sleepIslands",
    "sensors",
];
const SHOWN = FIELDS;
const threadCounts = (process.env.PHASE_THREADS ?? "1,4").split(",").map(Number);
const timing = process.env.PHASE_TIMING === "1";

const native = nativeBinary();
const dir = mkdtempSync(join(tmpdir(), "box3d-parity-phases-"));
const built = await Bun.build({
    entrypoints: [join(import.meta.dir, "scenes.ts")],
    outdir: dir,
    target: "node",
    format: "esm",
});
if (!built.success) throw new Error(built.logs.join("\n"));
const bundle = join(dir, "scenes.js");
process.on("exit", () => rmSync(dir, { recursive: true, force: true }));

const median = (a: number[]) => {
    const s = [...a].sort((x, y) => x - y);
    return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const profile = (lines: string[]) =>
    lines.filter((l) => l.startsWith("F ")).map((l) => l.split(" ").slice(2).map(Number));
const hashes = (lines: string[]) => lines.filter((l) => /^\d+ 0x/.test(l));

async function run(cmd: string[], from: number): Promise<string[]> {
    const proc = Bun.spawn(cmd, {
        env: {
            ...process.env,
            PROFILE: timing ? String(from) : "-1",
            CPU: timing ? String(from) : "-1",
            COUNTERS: String(from),
        },
        stdout: "pipe",
        stderr: "pipe",
    });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (exitCode !== 0) throw new Error(`${cmd.join(" ")} failed:\n${stderr}`);
    return stdout.trim().split("\n");
}

async function runCase(scene: string, threads: number, from: number, to: number) {
    const args = [scene, String(threads), String(to)];
    const n = await run([native, ...args], from);
    const s = await run(["node", bundle, ...args], from);
    return { n, s };
}

const cases = new Map<string, ReturnType<typeof runCase>>();
if (!timing) {
    for (const [scene, [from, to]] of Object.entries(WINDOWS)) {
        for (const threads of threadCounts) {
            const result = runCase(scene, threads, from, to);
            // Tests consume the failure even if a child exits before its test starts.
            void result.catch(() => {});
            cases.set(`${scene}:${threads}`, result);
        }
    }
}

for (const [scene, [from, to]] of Object.entries(WINDOWS)) {
    test(`${scene} hashes and counts equal native over steps ${from}-${to - 1}`, async () => {
        const report: string[] = [];
        for (const threads of threadCounts) {
            const { n, s } = await (cases.get(`${scene}:${threads}`) ?? runCase(scene, threads, from, to));
            expect(hashes(s)).toEqual(hashes(n));
            const counters = n.filter((l) => l.startsWith("N ")).map((l) => l.split(" "));
            const worldCounts = s.filter((l) => l.startsWith("W ")).map((l) => l.split(" "));
            expect(counters).toHaveLength(to - from);
            expect(worldCounts).toHaveLength(counters.length);
            for (let i = 0; i < counters.length; ++i) {
                for (const name of ["contacts", "awake", "joints"]) {
                    expect(Number(worldCounts[i][worldCounts[i].indexOf(name) + 1])).toBe(
                        Number(counters[i][counters[i].indexOf(name) + 1]),
                    );
                }
            }
            if (!timing) continue;
            const np = profile(n),
                sp = profile(s);
            report.push(
                `${scene}, ${threads} thread${threads > 1 ? "s" : ""}: median ms per step over steps ${from}-${to - 1}`,
                "| phase | native | shallot | ratio |",
                "|---|---|---|---|",
            );
            for (const field of SHOWN) {
                const k = FIELDS.indexOf(field);
                const a = median(np.map((r) => r[k])),
                    b = median(sp.map((r) => r[k]));
                report.push(
                    `| ${field} | ${a.toFixed(2)} | ${b.toFixed(2)} | ${a >= 0.01 ? (b / a).toFixed(1) : "-"} |`,
                );
            }
            // The step ratio's median over each half of the window: equal halves mean the window suffices.
            const ratio = np.map((r, i) => sp[i][0] / r[0]);
            const half = ratio.length >> 1;
            report.push(
                `step ratio median, first and second half: ${median(ratio.slice(0, half)).toFixed(1)}, ${median(ratio.slice(half)).toFixed(1)}`,
            );
            const count = (name: string) =>
                median(counters.map((c) => Number(c[c.indexOf(name) + 1])));
            report.push(
                `native counters (median): contacts ${count("contacts")}, awake ${count("awake")}, with a manifold ${count("manifolds")}, recycled ${count("recycled")}, sat calls ${count("sat")}, sat cache hits ${count("satHit")}, joints ${count("joints")}`,
                "Shallot contacts, awake contacts and joints equal native at every measured step; SAT calls/cache hits are native denominators, not Shallot instrumentation.",
                "shallot main thread, CPU profile ms per step (J phase total wasm join ts; JC inclusive callees; wasm includes internal stage waits, join is only the JS pool tail):",
                ...s.filter((l) => l.startsWith("J")),
                "",
            );
        }
        if (timing) console.info(report.join("\n"));
    });
}
