// Runs under Node, never Bun: V8's sampling heap profiler counts allocation exactly, where JSC's
// statistics hold still between collections.
// argv: <bundle.mjs> <warm frames> <window frames> <input file> [transition]. The bundle's default export
// takes the input text and resolves to { step(), dispose() }, plus { spawn(), despawn() } for a transition;
// its `control` export allocates one known literal per call. Prints one JSON sample on stdout.
import { readFileSync } from "node:fs";
import { findSourceMap } from "node:module";
import { Session } from "node:inspector/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

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

function bundleSite(frame) {
    const name = frame.functionName || "(anonymous)";
    const entry = findSourceMap(bundle)?.findEntry(frame.lineNumber, frame.columnNumber);
    if (entry?.originalSource === undefined) return `${name} bundle:${frame.lineNumber + 1}`;
    const source = entry.originalSource.startsWith("file://")
        ? new URL(entry.originalSource).pathname
        : resolve(dirname(bundle), entry.originalSource);
    return `${name} ${relative(process.cwd(), source)}:${entry.originalLine + 1}`;
}

const harnessSite = (frame) =>
    `${frame.functionName} ${relative(process.cwd(), fileURLToPath(samplerUrl))}:${frame.lineNumber + 1}`;

// Sum sampled bytes under the run frame by the nearest bundle frame, or by the run frame itself where the
// subject was inlined into it. A builtin frame (a typed-array constructor, an iterator `next`,
// `Set.prototype.clear`) has an empty url and credits the frame above it. Samples outside the run frame,
// the inspector's own bookkeeping, are not the subject's.
function sites(profile) {
    const owner = new Map();
    const visit = (node, nearest) => {
        const frame = node.callFrame;
        let here = nearest;
        if (frame.url === samplerUrl && RUN_FRAMES.has(frame.functionName)) here = harnessSite(frame);
        else if (nearest !== undefined && frame.url === bundleUrl) here = bundleSite(frame);
        if (here !== undefined) owner.set(node.id, here);
        for (const child of node.children) visit(child, here);
    };
    visit(profile.head, undefined);
    const bytes = new Map();
    for (const { nodeId, size } of profile.samples) {
        const key = owner.get(nodeId);
        if (key !== undefined) bytes.set(key, (bytes.get(key) ?? 0) + size);
    }
    return [...bytes]
        .map(([name, size]) => ({ site: name, bytes: size }))
        .sort((x, y) => y.bytes - x.bytes);
}

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
    return [
        { label: `after warm ${warm}`, sites: atWarm },
        { label: `after warm ${2 * warm}`, sites: atDoubleWarm },
        { label: "A/A repeat", sites: repeat },
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
        { label: `${CHUNK} frames after spawn`, sites: afterSpawn },
        { label: `${CHUNK} frames after despawn`, sites: afterDespawn },
        { label: `${CHUNK} frames after second spawn`, sites: afterSpawnAgain },
        { label: `${CHUNK} frames after second despawn`, sites: afterDespawnAgain },
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
