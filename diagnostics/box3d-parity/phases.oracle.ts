// Manual oracle comparing Shallot's step phases with native Box3D's b3Profile on joint_grid, rain and
// junkyard at full size (strategy unit box3d-parity). Run by path, with a Box3D checkout at 47d7f7cc:
//
//     BOX3D=/path/to/box3d bun test ./diagnostics/box3d-parity/phases.oracle.ts
//
// For each scene at 1 and 4 threads, native.c and scenes.ts (bundled for Node) step to the end of a window
// of steps; both print every step's phase times inside it, and Shallot also its main thread's CPU profile
// over it (cpu.ts). A test asserts only that the two hash equal at every step, so both time the same
// world; timings are reported, never asserted. PHASE_THREADS (comma list, "1,4") picks the thread counts.
import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary, run } from "./native";

setDefaultTimeout(300_000);

// Windows sized to the fewest steps whose median the run repeats: joint_grid is steady after its first
// steps; rain's last column spawns before 280; junkyard's rocks have landed and its contacts climb.
const WINDOWS = {
    // biome-ignore lint/style/useNamingConvention: Box3D's benchmark name, as native.c takes it.
    joint_grid: [20, 40],
    rain: [280, 320],
    junkyard: [180, 200],
} as const;
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

for (const [scene, [from, to]] of Object.entries(WINDOWS)) {
    test(`${scene} hashes equal native over steps ${from}-${to - 1}; its phase times are reported against b3Profile`, () => {
        const report: string[] = [];
        for (const threads of threadCounts) {
            const env = { PROFILE: String(from), CPU: String(from) };
            const args = [scene, String(threads), String(to)];
            const n = run([native, ...args], env)
                .trim()
                .split("\n");
            const s = run(["node", bundle, ...args], env)
                .trim()
                .split("\n");
            expect(hashes(s)).toEqual(hashes(n));
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
            const counters = n.filter((l) => l.startsWith("N ")).map((l) => l.split(" "));
            const worldCounts = s.filter((l) => l.startsWith("W ")).map((l) => l.split(" "));
            for (let i = 0; i < counters.length; ++i) {
                for (const name of ["contacts", "awake", "joints"]) {
                    expect(Number(worldCounts[i][worldCounts[i].indexOf(name) + 1])).toBe(
                        Number(counters[i][counters[i].indexOf(name) + 1]),
                    );
                }
            }
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
        console.info(report.join("\n"));
    });
}
