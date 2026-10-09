import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { CEILING } from "../../scripts/test-tiers";
import { allocationFailure, measuredIsolateTrace, tracedSample, TIER_FLAGS } from "./allocation";

setDefaultTimeout(CEILING.node);

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
            [
                "node",
                ...TIER_FLAGS,
                "--trace-opt",
                "--redirect-code-traces",
                "--allow-natives-syntax",
                "subject.mjs",
            ],
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
        expect(
            traces.some((file) =>
                readFileSync(join(dir, file), "utf8").includes("<JSFunction now "),
            ),
        ).toBe(true);
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
