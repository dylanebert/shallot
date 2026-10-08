import { expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
    type AllocationSample,
    allocatesNothing,
    allocationFailure,
    measuredIsolateTrace,
    windowBytes,
    tracedSample,
    TIER_FLAGS,
} from "./allocation";

test("the sampler keeps lazy feedback and Chromium's existing timing flags", () => {
    expect(TIER_FLAGS).not.toContain("--no-lazy-feedback-allocation");
    expect(TIER_FLAGS).toEqual([
        "--no-concurrent-recompilation",
        "--invocation-count-for-maglev=10",
        "--invocation-count-for-turbofan=10",
    ]);
});

const ENTRY = resolve(import.meta.dir, "../../examples/first-person/src/allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");

test("a Node allocation import does not load the display-only oracle or Hyprland instrument", async () => {
    const built = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation.ts")],
        metafile: true,
        external: ["playwright", "webgpu", "chromium-bidi"],
        target: "bun",
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`allocation import graph failed: ${built.logs.map(String).join("\\n")}`);
    const inputs = Object.keys(built.metafile.inputs).map((path) => resolve(path));
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display.ts"))).toBe(false);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display-seat.ts"))).toBe(false);

    const node = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation-sampler.mjs")],
        metafile: true,
        target: "node",
        external: ["webgpu"],
    });
    if (!node.success || node.metafile === undefined)
        throw new Error(`Node sampler graph failed: ${node.logs.map(String).join("\\n")}`);
    const nodeInputs = Object.keys(node.metafile.inputs).map((path) => resolve(path));
    expect(nodeInputs.some((path) => path.endsWith("/allocation-sampler.mjs"))).toBe(true);
    expect(nodeInputs.some((path) => path.endsWith("/display.ts"))).toBe(false);
    const nodeImports = Object.values(node.metafile.inputs).flatMap((input) => input.imports);
    expect(nodeImports.some((edge) => edge.external && edge.path === "webgpu")).toBe(true);
    expect(nodeImports.some((edge) => edge.external && /^(?:bun:|playwright)/.test(edge.path))).toBe(
        false,
    );
});

// Profiler modules: the `profile` extra, which owns the physics step's timing clock.
const PROFILER = [/^src\/extras\/profile\//];

test("the allocation-gated first-person composition carries no timing or profiling module, so its default step does no diagnostics work", async () => {
    const built = await Bun.build({
        entrypoints: [ENTRY],
        target: "node",
        format: "esm",
        metafile: true,
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`gated bundle failed: ${built.logs.map(String).join("\n")}`);
    const inputs = built.metafile.inputs;
    const paths = Object.keys(inputs).map((path) => resolve(ROOT, path).slice(ROOT.length + 1));
    // Non-vacuity: the graph reaches the physics step whose timers this row is about.
    if (!paths.includes("src/standard/physics/solver/step.ts"))
        throw new Error(
            `inconclusive: gated bundle graph lacks the physics step (${paths.length} modules)`,
        );
    const importers = (module: string) =>
        Object.entries(inputs)
            .filter(([, input]) =>
                input.imports.some((edge) => resolve(ROOT, edge.path) === resolve(ROOT, module)),
            )
            .map(([path]) => path);
    const found = Object.keys(inputs).filter((path) =>
        PROFILER.some((pattern) => pattern.test(resolve(ROOT, path).slice(ROOT.length + 1))),
    );
    if (found.length !== 0)
        throw new Error(
            `gated bundle imports profiler modules:\n${found.map((path) => `  ${path} <- ${importers(path).join(", ")}`).join("\n")}`,
        );
});

test("optimization traces refuse even a zero-byte window, including anonymous OSR and repeated compiles", () => {
    const sample = { runtime: "planted trace", warm: 120, frames: 120, control: [], windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites: [], frames: 120, framesAtMost: 120 },
    ] };
    const lines = [
        'SHALLOT_SAMPLE_BEGIN "after warm 120"',
        '[compiling method 0x1 <JSFunction late (sfi = 0x2)> (target MAGLEV), mode: ConcurrencyMode::kSynchronous]',
        '[completed compiling 0x1 <JSFunction late (sfi = 0x2)> (target MAGLEV)]',
        '[compiling method 0x1 <JSFunction (sfi = 0x3)> (target TURBOFAN_JS) OSR, mode: ConcurrencyMode::kSynchronous]',
        'SHALLOT_SAMPLE_END',
        'SHALLOT_SAMPLE_BEGIN "after warm 240"',
        'SHALLOT_SAMPLE_END',
        'SHALLOT_SAMPLE_BEGIN "A/A repeat"',
        'SHALLOT_SAMPLE_END',
        JSON.stringify(sample),
    ];
    const result = tracedSample(lines.join("\n"));
    expect(result.windows[0].optimizations).toEqual(["late", "(anonymous)"]);
    expect(allocatesNothing(result)).toBe(false);
    expect(allocationFailure(result)).toContain("after warm 120; optimized late, (anonymous); no byte reading");
    expect(() => tracedSample(JSON.stringify(sample))).toThrow("missing allocation trace markers");
    expect(() => tracedSample('SHALLOT_SAMPLE_BEGIN "A/A repeat"')).toThrow("incomplete allocation trace");
});

test("a worker isolate's optimization inside a window does not refuse it, and the measured isolate's still does", async () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-isolate-trace-"));
    try {
        const sampler = pathToFileURL(resolve(import.meta.dir, "allocation-sampler.mjs")).href;
        writeFileSync(
            join(dir, "subject.mjs"),
            `import { openSync, writeSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { measuredTrace } from ${JSON.stringify(sampler)};
const trace = measuredTrace();
const fd = openSync(trace, "a");
const mark = (text) => writeSync(fd, text);
// The worker boots before the window and heats \`now\` inside it while this thread blocks, so the
// window's only optimization is the worker's.
const flag = new Int32Array(new SharedArrayBuffer(8));
const worker = new Worker(\`const { workerData: flag, parentPort } = require("node:worker_threads");
function now() { let s = 0; for (let i = 0; i < 100; i++) s += i; return s; }
parentPort.postMessage(0);
Atomics.wait(flag, 0, 0);
for (let i = 0; i < 100000; i++) now();
Atomics.store(flag, 1, 1);
Atomics.notify(flag, 1);\`, { eval: true, workerData: flag });
await new Promise((done) => worker.once("message", done));
for (let i = 0; i < 1000; i++) mark("");
const windows = ["after warm 120", "after warm 240", "A/A repeat"];
mark(\`SHALLOT_SAMPLE_BEGIN \${JSON.stringify(windows[0])}\\n\`);
Atomics.store(flag, 0, 1);
Atomics.notify(flag, 0);
Atomics.wait(flag, 1, 0);
mark("SHALLOT_SAMPLE_END\\n");
await new Promise((done) => worker.on("exit", done));
mark(\`SHALLOT_SAMPLE_BEGIN \${JSON.stringify(windows[1])}\\n\`);
function late() { let s = 0; for (let i = 0; i < 100; i++) s += i; return s; }
for (let i = 0; i < 100000; i++) late();
mark("SHALLOT_SAMPLE_END\\n");
mark(\`SHALLOT_SAMPLE_BEGIN \${JSON.stringify(windows[2])}\\n\`);
mark("SHALLOT_SAMPLE_END\\n");
console.log(JSON.stringify({ runtime: "isolates", warm: 120, frames: 120, control: [],
    windows: windows.map((label) => ({ label, sites: [], frames: 120, framesAtMost: 120 })) }));
`,
        );
        const proc = Bun.spawn(
            ["node", ...TIER_FLAGS, "--trace-opt", "--redirect-code-traces", "--allow-natives-syntax", "subject.mjs"],
            { cwd: dir, stdout: "pipe", stderr: "pipe" },
        );
        const [stdout, stderr, code] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        if (code !== 0) throw new Error(`isolate trace subject exited ${code}: ${stderr}`);
        // Non-vacuity: the worker's own trace shows `now` optimizing.
        const traces = readdirSync(dir).filter((file) => file.endsWith(".asm"));
        expect(traces.some((file) => readFileSync(join(dir, file), "utf8").includes("<JSFunction now "))).toBe(true);
        const result = tracedSample(`${measuredIsolateTrace(dir)}\n${stdout}`);
        expect(result.windows[0].optimizations).toBeUndefined();
        expect(result.windows[1].optimizations).toContain("late");
        expect(result.windows[1].optimizations).not.toContain("now");
        expect(allocationFailure(result)).toContain("after warm 240; optimized late");
        expect(allocationFailure(result)).not.toContain("after warm 120");
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

const steadySample = (sites: AllocationSample["windows"][number]["sites"]) => ({
    warm: 120,
    windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites, frames: 120, framesAtMost: 120 },
    ],
});

test("the allocation gate passes when all three expected steady windows sample zero bytes at zero sites", () => {
    const sample = steadySample([]);
    expect(allocatesNothing(sample)).toBe(true);
    expect(allocationFailure(sample)).toBeUndefined();
});

test("the allocation gate reds with the names of all expected windows when a sampler returns no steady windows", () => {
    const sample = { warm: 120, windows: [] as AllocationSample["windows"] };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 120, after warm 240, A/A repeat",
    );
});

test("the allocation gate reds with the names of expected steady windows omitted by an incomplete sampler result", () => {
    const sample = {
        warm: 120,
        windows: [{ label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 }],
    };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 240, A/A repeat",
    );
});

test("the allocation gate reds on any sampled steady allocation and prints its site only for diagnosis", () => {
    const sample = steadySample([
        { site: "stepChunk src/transitional/character/sweep.ts:42", bytes: 96, count: 3 },
    ]);
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady play allocated JavaScript heap; sampler sites are diagnosis only:\n  A/A repeat: 96 B at stepChunk src/transitional/character/sweep.ts:42",
    );
});
