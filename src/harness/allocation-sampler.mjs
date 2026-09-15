// Runs under Node, never Bun: V8's sampling heap profiler counts allocation exactly, where JSC's
// statistics hold still between collections.
// argv: <bundle.mjs> <warm frames> <window frames> <input file>. The bundle's default export takes
// the input text and resolves to { step(), dispose() }. Prints one JSON sample on stdout.
import { readFileSync } from "node:fs";
import { findSourceMap } from "node:module";
import { Session } from "node:inspector/promises";
import { dirname, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [bundle, warmArg, framesArg, inputFile] = process.argv.slice(2);
const warm = Number(warmArg);
const frames = Number(framesArg);
if (!bundle || !inputFile || !Number.isInteger(frames) || frames <= 0 || !Number.isInteger(warm) || warm < frames)
    throw new Error("usage: allocation-sampler.mjs <bundle> <warm >= frames> <frames> <input>");
const collect = globalThis.gc;
if (typeof collect !== "function") throw new Error("allocation sampler needs node --expose-gc");

const bundleUrl = pathToFileURL(bundle).href;

function site(frame) {
    const name = frame.functionName || "(anonymous)";
    const entry = findSourceMap(bundle)?.findEntry(frame.lineNumber, frame.columnNumber);
    if (entry?.originalSource === undefined) return `${name} bundle:${frame.lineNumber + 1}`;
    const source = entry.originalSource.startsWith("file://")
        ? new URL(entry.originalSource).pathname
        : resolve(dirname(bundle), entry.originalSource);
    return `${name} ${relative(process.cwd(), source)}:${entry.originalLine + 1}`;
}

// Sum sampled bytes by the nearest frame in `url`. A builtin frame (a typed-array constructor, an
// iterator `next`, `Set.prototype.clear`) has an empty url; its bytes belong to the subject frame above
// it. Only samples with no such ancestor, the inspector's own bookkeeping, are not the subject's.
function sites(profile, url) {
    const owner = new Map();
    const visit = (node, nearest) => {
        const here = node.callFrame.url === url ? site(node.callFrame) : nearest;
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

async function sample(url, body) {
    await session.post("HeapProfiler.startSampling", {
        samplingInterval: 1,
        includeObjectsCollectedByMajorGC: true,
        includeObjectsCollectedByMinorGC: true,
    });
    body();
    const { profile } = await session.post("HeapProfiler.stopSampling");
    return sites(profile, url);
}

const { default: create } = await import(bundleUrl);
const subject = await create(readFileSync(inputFile, "utf8"));
const run = (n) => {
    for (let i = 0; i < n; i++) subject.step();
};
const window = () => {
    collect();
    return sample(bundleUrl, () => run(frames));
};
try {
    // Three windows: after `warm` frames, after twice that, and an A/A repeat. Tiering only adds
    // allocation, so each must read zero on its own.
    run(warm);
    const atWarm = await window();
    run(warm - frames);
    const atDoubleWarm = await window();
    const repeat = await window();

    // Control: the sampler attributes a known per-frame literal, so an empty site set is not a dead probe.
    let sink;
    const control = await sample(import.meta.url, () => {
        for (let i = 0; i < frames; i++) sink = { frame: i };
    });
    void sink;

    process.stdout.write(
        `${JSON.stringify({
            runtime: `node ${process.version} ${process.execArgv.join(" ")}`,
            warm,
            frames,
            windows: [
                { label: `after warm ${warm}`, sites: atWarm },
                { label: `after warm ${2 * warm}`, sites: atDoubleWarm },
                { label: "A/A repeat", sites: repeat },
            ],
            controlBytes: control.reduce((sum, row) => sum + row.bytes, 0),
        })}\n`,
    );
} finally {
    subject.dispose();
    session.disconnect();
}
