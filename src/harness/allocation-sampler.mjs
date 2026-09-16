// Allocation attribution for both samplers, and the Node sampler itself. `attribute`, `subjectSite` and
// `originalPosition` are shared with the page sampler in `./allocation.ts`. The rest runs only when Node runs
// this file, never Bun: V8's sampling heap profiler counts allocation exactly, where JSC's statistics hold
// still between collections.
// argv: <bundle.mjs> <warm frames> <window frames> <input file> [transition]. The bundle's default export
// takes the input text and resolves to { step(), dispose() }, plus { spawn(), despawn() } for a transition;
// its `control` export allocates one known literal per call. Prints one JSON sample on stdout.
import { readFileSync } from "node:fs";
import { Session } from "node:inspector/promises";
import { findSourceMap } from "node:module";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/**
 * A profile frame's original source and zero-based line through `map`, a `node:module` SourceMap whose
 * relative sources resolve from `base`; undefined where the map has no entry.
 * @param {{ lineNumber: number, columnNumber: number }} frame
 * @param {import("node:module").SourceMap | undefined} map
 * @param {string} base
 * @returns {{ source: string, line: number } | undefined}
 */
export function originalPosition(frame, map, base) {
    const entry = map?.findEntry(frame.lineNumber, frame.columnNumber);
    if (entry === undefined || !("originalSource" in entry)) return undefined;
    const source = entry.originalSource.startsWith("file://")
        ? new URL(entry.originalSource).pathname
        : resolve(base, entry.originalSource);
    return { source, line: entry.originalLine };
}

/**
 * A subject frame's site: its function name and original source line, or `fallback` and its generated line
 * where the map has no entry.
 * @param {{ functionName: string, lineNumber: number, columnNumber: number }} frame
 * @param {import("node:module").SourceMap | undefined} map
 * @param {string} base
 * @param {string} fallback
 * @returns {string}
 */
export function subjectSite(frame, map, base, fallback) {
    const name = frame.functionName || "(anonymous)";
    const at = originalPosition(frame, map, base);
    return at === undefined
        ? `${name} ${fallback}:${frame.lineNumber + 1}`
        : `${name} ${relative(process.cwd(), at.source)}:${at.line + 1}`;
}

/**
 * @typedef {{ functionName: string, url: string, scriptId: string, lineNumber: number, columnNumber: number }} CallFrame
 * @typedef {{ id: number, callFrame: CallFrame, children: ProfileNode[] }} ProfileNode
 */

/**
 * Sum a sampling heap profile's bytes under a run frame by the nearest subject frame, or by the run frame
 * itself where the subject was inlined into it. A frame outside the subject (a builtin such as a typed-array
 * constructor, an iterator `next` or `Set.prototype.clear`, or the browser's own) credits the subject frame
 * above it. Samples outside every run frame, the profiler's and the harness's own, are not the subject's.
 * @param {{ head: ProfileNode, samples: { nodeId: number, size: number }[] }} profile
 * @param {(frame: CallFrame) => string | undefined} runSite names a run frame, else undefined
 * @param {(frame: CallFrame) => string | undefined} siteOf names a subject frame, else undefined
 * At a one-byte sampling interval every allocation is sampled, so the number of samples at a site is the
 * number of allocations there: a sanctioned site's per-frame count is read from it, never from bytes.
 * @returns {{ site: string, bytes: number, count: number }[]} most bytes first
 */
export function attribute(profile, runSite, siteOf) {
    const owner = new Map();
    /** @param {ProfileNode} node @param {string | undefined} nearest */
    const visit = (node, nearest) => {
        const frame = node.callFrame;
        const here = runSite(frame) ?? (nearest === undefined ? undefined : (siteOf(frame) ?? nearest));
        if (here !== undefined) owner.set(node.id, here);
        for (const child of node.children) visit(child, here);
    };
    visit(profile.head, undefined);
    const bytes = new Map();
    const counts = new Map();
    for (const { nodeId, size } of profile.samples) {
        const key = owner.get(nodeId);
        if (key === undefined) continue;
        bytes.set(key, (bytes.get(key) ?? 0) + size);
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...bytes]
        .map(([name, size]) => ({ site: name, bytes: size, count: counts.get(name) ?? 0 }))
        .sort((x, y) => y.bytes - x.bytes);
}

async function main() {
    const [bundle, warmArg, framesArg, inputFile, mode] = process.argv.slice(2);
    const warm = Number(warmArg);
    const frames = Number(framesArg);
    if (!bundle || !inputFile || !Number.isInteger(frames) || frames <= 0 || !Number.isInteger(warm) || warm < frames)
        throw new Error("usage: allocation-sampler.mjs <bundle> <warm >= frames> <frames> <input>");
    const collect = globalThis.gc;
    if (typeof collect !== "function") throw new Error("allocation sampler needs node --expose-gc");

    const bundleUrl = pathToFileURL(bundle).href;
    const samplerUrl = import.meta.url;

    // Frames step in short calls, so the warm compiles this loop for an ordinary entry. One long call would
    // tier it only by on-stack replacement, and each window's fresh entry would install new code, whose
    // allocation lands on whichever subject frame is running. Each chunk closes over its one callee, so its
    // call site stays monomorphic. Every sampled byte under a chunk frame is the subject's, including what
    // TurboFan inlines up into it.
    const CHUNK = 60;
    const RUN_FRAMES = new Set(["stepChunk", "controlChunk"]);

    const runSite = (frame) =>
        frame.url === samplerUrl && RUN_FRAMES.has(frame.functionName)
            ? `${frame.functionName} ${relative(process.cwd(), fileURLToPath(samplerUrl))}:${frame.lineNumber + 1}`
            : undefined;
    const bundleSite = (frame) =>
        frame.url === bundleUrl
            ? subjectSite(frame, findSourceMap(bundle), dirname(bundle), "bundle")
            : undefined;
    const sites = (profile) => attribute(profile, runSite, bundleSite);

    const session = new Session();
    session.connect();
    await session.post("HeapProfiler.enable");

    async function sample(run, n) {
        collect();
        await session.post("HeapProfiler.startSampling", {
            samplingInterval: 1,
            includeObjectsCollectedByMajorGC: true,
            includeObjectsCollectedByMinorGC: true,
        });
        run(n);
        const { profile } = await session.post("HeapProfiler.stopSampling");
        return sites(profile);
    }

    const { default: create, control } = await import(bundleUrl);
    if (typeof control !== "function")
        throw new Error("allocation sampler: the bundle must export a `control` function");
    if (warm % CHUNK !== 0 || frames % CHUNK !== 0)
        throw new Error(`allocation sampler: warm and frames must be multiples of ${CHUNK}`);
    const subject = await create(readFileSync(inputFile, "utf8"));
    const stepChunk = () => {
        for (let i = 0; i < CHUNK; i++) subject.step();
    };
    const controlChunk = () => {
        for (let i = 0; i < CHUNK; i++) control();
    };
    const steps = (n) => {
        for (let i = 0; i < n; i += CHUNK) stepChunk();
    };
    const controls = (n) => {
        for (let i = 0; i < n; i += CHUNK) controlChunk();
    };
    // A transition's event frames: the subject's `spawn` and `despawn` each mutate the scene and step one
    // frame. They share the chunk rule, so their samples count under a run frame too.
    const spawnFrame = () => subject.spawn();
    const despawnFrame = () => subject.despawn();
    RUN_FRAMES.add("spawnFrame").add("despawnFrame");

    // Three windows: after `warm` frames, after twice that, and an A/A repeat. Tiering only adds
    // allocation, so each must read zero on its own.
    async function steadyWindows() {
        steps(warm);
        const atWarm = await sample(steps, frames);
        steps(warm - frames);
        const atDoubleWarm = await sample(steps, frames);
        const repeat = await sample(steps, frames);
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
        }
        const spawn = await sample(spawnFrame);
        const afterSpawn = await sample(steps, CHUNK);
        const despawn = await sample(despawnFrame);
        const afterDespawn = await sample(steps, CHUNK);
        const spawnAgain = await sample(spawnFrame);
        const afterSpawnAgain = await sample(steps, CHUNK);
        const despawnAgain = await sample(despawnFrame);
        const afterDespawnAgain = await sample(steps, CHUNK);
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
        collect();
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

    try {
        const measured =
            mode === "transition" ? await transition() : { windows: await steadyWindows() };

        // Control: the bundle's known per-call literal, run and attributed exactly as the windows are, so an
        // empty site set is not a dead probe. It runs last, so it never reaches the windows' code.
        const controlSites = await sample(controls, frames);

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
        subject.dispose();
        session.disconnect();
    }
}

if (import.meta.main) await main();
