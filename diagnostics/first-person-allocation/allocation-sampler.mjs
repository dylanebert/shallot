// Node's V8 sampling heap profiler and allocation attribution. Its source-map helpers are also used by the
// first-person display diagnostic; the rest runs only when Node executes this file, never in the Bun host.
// JSC's statistics hold still between collections.
// Needs node --expose-gc --allow-natives-syntax.
// argv: <bundle.mjs> <warm frames> <window frames> <input file> [transition]. The bundle's default export
// takes the input text and resolves to { step(), dispose() }, plus { spawn(), despawn() } for a transition;
// its `control` export allocates one known literal per call. Needs --trace-opt --redirect-code-traces;
// writes window markers into this isolate's trace file and one JSON sample on stdout.
import { openSync, readdirSync, readFileSync, realpathSync, writeSync } from "node:fs";
import { Session } from "node:inspector/promises";
import { findSourceMap } from "node:module";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/**
 * A profile frame's original source and zero-based line through `map`, a `node:module` SourceMap whose
 * relative sources resolve from `base`; undefined where the map has no entry.
 * @param {{ lineNumber: number, columnNumber?: number }} frame
 * @param {import("node:module").SourceMap | undefined} map
 * @param {string} base
 * @returns {{ source: string, line: number, column: number } | undefined}
 */
export function originalPosition(frame, map, base) {
    if (!Number.isInteger(frame.columnNumber)) return undefined;
    const entry = map?.findEntry(frame.lineNumber, frame.columnNumber);
    if (entry === undefined || !("originalSource" in entry)) return undefined;
    const source = entry.originalSource.startsWith("file://")
        ? new URL(entry.originalSource).pathname
        : resolve(base, entry.originalSource);
    return { source, line: entry.originalLine, column: entry.originalColumn };
}

/**
 * A frame's exact source position, including its column, or `fallback` and its generated position where
 * the map has no entry.
 * @param {{ functionName: string, lineNumber: number, columnNumber?: number }} frame
 * @param {import("node:module").SourceMap | undefined} map
 * @param {string} base
 * @param {string} fallback
 * @param {string} root
 * @returns {string}
 */
export function subjectSite(frame, map, base, fallback, root = process.cwd()) {
    const name = frame.functionName || "(anonymous)";
    const at = originalPosition(frame, map, base);
    return at === undefined
        ? `${name} ${fallback}:${frame.lineNumber + 1}:${Number.isInteger(frame.columnNumber) ? frame.columnNumber + 1 : "?"}`
        : `${name} ${relative(root.replace(/^\/private/, ""), at.source.replace(/^\/private/, ""))}:${at.line + 1}:${at.column + 1}`;
}

/**
 * @typedef {{ functionName: string, url: string, scriptId: string, lineNumber: number, columnNumber?: number }} CallFrame
 * @typedef {{ id: number, callFrame: CallFrame, children: ProfileNode[] }} ProfileNode
 */

/**
 * Collect and, for a subject that opts in, drain one unprofiled step before allocation sampling.
 * @param {{ step(): void, wait?: () => Promise<void>, warmSamplingSessions?: boolean }} subject
 * @param {() => void} collect
 */
export async function prepareSamplingSession(subject, collect) {
    collect();
    if (!subject.warmSamplingSessions) return;
    subject.step();
    await subject.wait?.();
}

/**
 * Preserve the ordered allocation stack beneath each run marker. V8 reports a wrapper declaration position;
 * source-checked single-expression boundaries identify the WebGPU calls admitted by the gate.
 * @param {{ head: ProfileNode, samples: { nodeId: number, size: number }[] }} profile
 * @param {(frame: CallFrame) => string | undefined} runSite names a run frame, else undefined
 * @param {(frame: CallFrame) => string | undefined} siteOf names a source-mapped frame, else undefined
 * @param {{ functionName: string, wrapperSite: string, expression: string }[]} expressions
 * @returns {{ site: string, bytes: number, count: number, stack: { functionName: string, site: string, native: boolean, boundary?: string }[], boundary?: string }[]} most bytes first
 */
export function attribute(profile, runSite, siteOf, expressions = []) {
    const boundaryByFunction = new Map(expressions.map((entry) => [entry.functionName, entry]));
    const paths = new Map();
    /** @param {ProfileNode} node @param {CallFrame[]} path */
    const visit = (node, path) => {
        const current = [...path, node.callFrame];
        paths.set(node.id, current);
        for (const child of node.children) visit(child, current);
    };
    visit(profile.head, []);
    const rows = new Map();
    for (const { nodeId, size } of profile.samples) {
        const path = paths.get(nodeId);
        if (!path) continue;
        const runIndex = path.findIndex((frame) => runSite(frame) !== undefined);
        if (runIndex < 0) continue;
        const stack = path.slice(runIndex + 1).map((frame) => {
            const native = frame.url === "" || frame.url === undefined || frame.url.startsWith("node:");
            return {
                functionName: frame.functionName || "(anonymous)",
                site: native
                    ? `${frame.functionName || "(native)"} <native>`
                    : siteOf(frame) ?? `${frame.functionName || "(anonymous)"} ${frame.url}:${frame.lineNumber + 1}:${Number.isInteger(frame.columnNumber) ? frame.columnNumber + 1 : "?"}`,
                native,
            };
        });
        let topJavaScript = -1;
        for (let i = stack.length - 1; i >= 0; i--) {
            if (!stack[i].native) {
                topJavaScript = i;
                break;
            }
        }
        const top = stack[topJavaScript];
        const topFrame = path[runIndex + 1 + topJavaScript];
        const candidate = top && boundaryByFunction.get(top.functionName);
        const boundary =
            candidate &&
            Number.isInteger(topFrame?.columnNumber) &&
            top.site === `${candidate.functionName} ${candidate.wrapperSite}`
                ? candidate
                : undefined;
        if (boundary && topJavaScript >= 0) stack[topJavaScript].boundary = boundary.wrapperSite;
        const site = boundary?.wrapperSite ?? top?.site ?? "<no JavaScript frame>";
        const key = JSON.stringify({ site, stack });
        const row = rows.get(key) ?? { site, bytes: 0, count: 0, stack, boundary: boundary?.wrapperSite };
        row.bytes += size;
        row.count++;
        rows.set(key, row);
    }
    return [...rows.values()].sort((a, b) => b.bytes - a.bytes);
}

/**
 * This isolate's code-trace file under `--redirect-code-traces`, which gives each isolate its own
 * `code-<pid>-<isolate id>.asm` in the working directory (V8 src/diagnostics/code-tracer.h). Found by
 * optimizing a probe only this isolate defines. Needs --allow-natives-syntax.
 */
export function measuredTrace() {
    function shallotMeasuredIsolate() {
        return 1;
    }
    const optimize = new Function(
        "fn",
        "%PrepareFunctionForOptimization(fn); fn(); fn(); %OptimizeFunctionOnNextCall(fn); fn();",
    );
    optimize(shallotMeasuredIsolate);
    const files = readdirSync(".").filter(
        (file) =>
            file.startsWith(`code-${process.pid}-`) &&
            file.endsWith(".asm") &&
            readFileSync(file, "utf8").includes("shallotMeasuredIsolate"),
    );
    if (files.length !== 1)
        throw new Error(`allocation sampler: expected one measured-isolate trace, found ${files.length}`);
    return resolve(files[0]);
}

async function main() {
    const [bundle, warmArg, framesArg, inputFile, mode] = process.argv.slice(2);
    const warm = Number(warmArg);
    const frames = Number(framesArg);
    if (!bundle || !inputFile || !Number.isInteger(frames) || frames <= 0 || !Number.isInteger(warm) || warm < frames)
        throw new Error("usage: allocation-sampler.mjs <bundle> <warm >= frames> <frames> <input>");
    const collect = globalThis.gc;
    if (typeof collect !== "function") throw new Error("allocation sampler needs node --expose-gc");

    const session = new Session();
    let gpu;
    let subject;
    let connected = false;
    try {
        const { create: createGpu, globals } = await import("webgpu");
        Object.assign(globalThis, globals);
        gpu = createGpu([]);
        navigator.gpu = gpu;

        const bundlePath = realpathSync(bundle);
        const projectRoot = process.env.SHALLOT_PROJECT_ROOT ?? process.cwd();
        const bundleUrl = pathToFileURL(bundlePath).href;
        const samplerUrl = import.meta.url;

        // Short chunks leave GPU drains outside the attributed run. Every sampled byte under a chunk
        // frame is attributed to the subject, including builtin bytes above its own call frames.
        const CHUNK = 60;
        const RUN_FRAMES = new Set(["stepChunk", "controlChunk"]);

        const runSite = (frame) =>
            frame.url === samplerUrl && RUN_FRAMES.has(frame.functionName)
                ? `${frame.functionName} ${relative(process.cwd(), fileURLToPath(samplerUrl))}:${frame.lineNumber + 1}`
                : undefined;
        const bundleSite = (frame) =>
            frame.url === bundleUrl
                ? subjectSite(frame, findSourceMap(bundlePath), dirname(bundlePath), "bundle", projectRoot)
                : undefined;
        let expressions = [];
        const sites = (profile) => attribute(profile, runSite, bundleSite, expressions);

        session.connect();
        connected = true;
        await session.post("HeapProfiler.enable");

        async function sample(run, n, label = "transition") {
            await prepareSamplingSession(subject, collect);
            await session.post("HeapProfiler.startSampling", {
                samplingInterval: 1,
                includeObjectsCollectedByMajorGC: true,
                includeObjectsCollectedByMinorGC: true,
            });
            // These synchronous markers share this isolate's trace file and bracket the profiled run,
            // excluding inspector and attribution work, just as the heap attribution does. V8 appends
            // each trace as it happens, and so do these, so they interleave in order.
            mark(`SHALLOT_SAMPLE_BEGIN ${JSON.stringify(label)}\n`);
            const running = run(n);
            if (running && typeof running.then === "function") await running;
            mark("SHALLOT_SAMPLE_END\n");
            const { profile } = await session.post("HeapProfiler.stopSampling");
            return sites(profile);
        }

        // Warm marker writes separately so their first compiles do not land in subject runs. A
        // descriptor, not appendFileSync, whose path handling tiers up late inside windows.
        const trace = measuredTrace();
        const traceFd = openSync(trace, "a");
        const mark = (text) => writeSync(traceFd, text);
        for (let i = 0; i < 10000; i++) mark("");

        const { default: create, control } = await import(bundleUrl);
        if (typeof control !== "function")
            throw new Error("allocation sampler: the bundle must export a `control` function");
        if (warm % CHUNK !== 0 || frames % CHUNK !== 0)
            throw new Error(`allocation sampler: warm and frames must be multiples of ${CHUNK}`);
        subject = await create(readFileSync(inputFile, "utf8"));
        expressions = (subject.allocationExpressions ?? []).map((entry) => ({
            functionName: entry.fn.name,
            wrapperSite: entry.wrapperSite,
            expression: entry.expression,
        }));
        // Preserve only the named attribution boundaries as stack frames; subject systems still tier normally.
        const neverOptimizeFunction = new Function("fn", "%NeverOptimizeFunction(fn)");
        for (const entry of subject.allocationExpressions ?? []) neverOptimizeFunction(entry.fn);
        for (const fn of subject.allocationFunctions ?? []) neverOptimizeFunction(fn);
        const stepChunk = () => {
            for (let i = 0; i < CHUNK; i++) subject.step();
        };
        const controlChunk = () => {
            for (let i = 0; i < CHUNK; i++) control();
        };
        // Keep no optimized harness code for GC to invalidate; the subject still tiers normally.
        // V8's CPU-profiler harness does this to `start` while optimizing its callees:
        // test/cctest/test-cpu-profiler.cc, inlining_test_source2.
        const neverOptimize = new Function("fn", "%NeverOptimizeFunction(fn)");
        neverOptimize(stepChunk);
        neverOptimize(controlChunk);
        const steps = async (n) => {
            for (let i = 0; i < n; i += CHUNK) {
                // Match the sampled chunks' collections during warm-up too: a full collection can
                // clear weak code dependencies and cause a recompile on the next subject call.
                collect();
                stepChunk();
                await subject.wait?.();
            }
        };
        const sampleSteps = async (n, label) => {
            const totals = new Map();
            for (let i = 0; i < n; i += CHUNK) {
                for (const row of await sample(stepChunk, CHUNK, label)) {
                    const key = JSON.stringify([row.site, row.stack]);
                    const current = totals.get(key) ?? { ...row, bytes: 0, count: 0 };
                    current.bytes += row.bytes;
                    current.count += row.count;
                    totals.set(key, current);
                }
                await subject.wait?.();
            }
            return [...totals.values()].sort((a, b) => b.bytes - a.bytes);
        };
        const controls = (n) => {
            for (let i = 0; i < n; i += CHUNK) controlChunk();
        };
        // A transition's event frames: the subject's `spawn` and `despawn` each mutate the scene and step one
        // frame. They share the chunk rule, so their samples count under a run frame too.
        const spawnFrame = () => subject.spawn();
        const despawnFrame = () => subject.despawn();
        RUN_FRAMES.add("spawnFrame").add("despawnFrame");

        // Each window must read zero independently and have no optimization during its runs.
        async function steadyWindows() {
            await steps(warm);
            const atWarm = await sampleSteps(frames, `after warm ${warm}`);
            await steps(warm - frames);
            const atDoubleWarm = await sampleSteps(frames, `after warm ${2 * warm}`);
            const repeat = await sampleSteps(frames, "A/A repeat");
            // Node steps its own frames, so each window's frame count is exact by construction; the page
            // sampler has to measure its windows, because a page window overshoots what it was asked for.
            return [
                { label: `after warm ${warm}`, sites: atWarm, frames, framesAtMost: frames },
                { label: `after warm ${2 * warm}`, sites: atDoubleWarm, frames, framesAtMost: frames },
                { label: "A/A repeat", sites: repeat, frames, framesAtMost: frames },
            ];
        }

        // The transition: a warm of paired cycles so the spawn and despawn paths tier as play would reach them,
        // then two sampled cycles at the same high-water mark whose event frames and the chunk after each count
        // every byte, the steady windows after them, and a third cycle sampled live-only: after despawn and a full
        // collection, what the cycle allocated and still holds.
        async function transition() {
            for (let i = 0; i < warm; i += CHUNK) {
                spawnFrame();
                stepChunk();
                despawnFrame();
                await subject.wait?.();
            }
            const drainedSample = async (run) => {
                const result = await sample(run, 1, "transition");
                await subject.wait?.();
                return result;
            };
            const spawn = await drainedSample(spawnFrame);
            const afterSpawn = await sample(steps, CHUNK, "transition");
            const despawn = await drainedSample(despawnFrame);
            const afterDespawn = await sample(steps, CHUNK, "transition");
            const spawnAgain = await drainedSample(spawnFrame);
            const afterSpawnAgain = await sample(steps, CHUNK, "transition");
            const despawnAgain = await drainedSample(despawnFrame);
            const afterDespawnAgain = await sample(steps, CHUNK, "transition");
            const afterEvents = [
                { label: `${CHUNK} frames after spawn`, sites: afterSpawn, frames: CHUNK, framesAtMost: CHUNK },
                { label: `${CHUNK} frames after despawn`, sites: afterDespawn, frames: CHUNK, framesAtMost: CHUNK },
                { label: `${CHUNK} frames after second spawn`, sites: afterSpawnAgain, frames: CHUNK, framesAtMost: CHUNK },
                {
                    label: `${CHUNK} frames after second despawn`,
                    sites: afterDespawnAgain,
                    frames: CHUNK, framesAtMost: CHUNK },
            ];
            const windows = await steadyWindows();
            await prepareSamplingSession(subject, collect);
            await session.post("HeapProfiler.startSampling", { samplingInterval: 1 });
            spawnFrame();
            stepChunk();
            despawnFrame();
            collect();
            const { profile } = await session.post("HeapProfiler.stopSampling");
            return {
                spawn,
                despawn,
                spawnAgain,
                despawnAgain,
                afterEvents,
                windows,
                survivors: sites(profile),
            };
        }

        const measured =
            mode === "transition" ? await transition() : { windows: await steadyWindows() };

        // Control: the bundle's known per-call literal, run and attributed exactly as the windows are, so an
        // empty site set is not a dead probe. It runs last, so it never reaches the windows' code.
        const controlSites = await sample(controls, frames, "control");

        process.stdout.write(
            `${JSON.stringify({
                runtime: `node ${process.version} ${process.execArgv.join(" ")}`,
                warm,
                frames,
                ...measured,
                control: controlSites,
            })}\n`,
        );
    } finally {
        try {
            subject?.dispose();
        } finally {
            try {
                if (connected) session.disconnect();
            } finally {
                navigator.gpu = undefined;
                gpu = undefined;
            }
        }
    }
}

if (import.meta.main) await main();
