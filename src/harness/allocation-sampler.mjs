// Runs under Node, never Bun: V8's heap statistics and sampling heap profiler count allocation
// exactly, where JSC's statistics hold still between collections.
// argv: <bundle.mjs> <warm frames> <measured frames> <input file>. The bundle's default export takes
// the input text and resolves to { step(), dispose() }. Prints one JSON sample on stdout.
import { readFileSync } from "node:fs";
import { findSourceMap } from "node:module";
import { Session } from "node:inspector/promises";
import { getHeapStatistics } from "node:v8";
import { dirname, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [bundle, warmArg, framesArg, inputFile] = process.argv.slice(2);
const warm = Number(warmArg);
const frames = Number(framesArg);
if (!bundle || !inputFile || !Number.isInteger(warm) || !Number.isInteger(frames) || frames <= 0)
    throw new Error("usage: allocation-sampler.mjs <bundle> <warm> <frames> <input>");
const collect = globalThis.gc;
if (typeof collect !== "function") throw new Error("allocation sampler needs node --expose-gc");

const bundleUrl = pathToFileURL(bundle).href;
const used = () => getHeapStatistics().used_heap_size;

function site(frame) {
    const name = frame.functionName || "(anonymous)";
    const entry = findSourceMap(bundle)?.findEntry(frame.lineNumber, frame.columnNumber);
    if (entry?.originalSource === undefined) return `${name} bundle:${frame.lineNumber + 1}`;
    const source = entry.originalSource.startsWith("file://")
        ? new URL(entry.originalSource).pathname
        : resolve(dirname(bundle), entry.originalSource);
    return `${name} ${relative(process.cwd(), source)}:${entry.originalLine + 1}`;
}

// Sum sampled bytes by site, keeping only frames in `url`; the inspector's own bookkeeping lives
// elsewhere and is not the subject's allocation.
function sites(profile, url) {
    const bytes = new Map();
    const visit = (node) => {
        if (node.selfSize > 0 && node.callFrame.url === url) {
            const key = site(node.callFrame);
            bytes.set(key, (bytes.get(key) ?? 0) + node.selfSize);
        }
        for (const child of node.children) visit(child);
    };
    visit(profile.head);
    return [...bytes].map(([name, size]) => ({ site: name, bytes: size }));
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
try {
    for (let i = 0; i < warm; i++) subject.step();
    collect();

    // Controls: a read costs a stable amount, an empty window moves the heap by exactly that read,
    // and the sampler attributes a known per-frame literal, so an empty site set is not a dead probe.
    used();
    const a = used();
    const warmReadBytes = used() - a;
    const n0 = used();
    for (let i = 0; i < frames; i++);
    const nullHeapDelta = used() - n0;
    let sink;
    const control = await sample(import.meta.url, () => {
        for (let i = 0; i < frames; i++) sink = { frame: i };
    });
    const controlBytes = control.reduce((sum, row) => sum + row.bytes, 0);
    void sink;

    collect();
    const measured = await sample(bundleUrl, () => {
        for (let i = 0; i < frames; i++) subject.step();
    });
    measured.sort((x, y) => y.bytes - x.bytes);
    process.stdout.write(
        `${JSON.stringify({
            runtime: `node ${process.version}`,
            warm,
            frames,
            totalBytes: measured.reduce((sum, row) => sum + row.bytes, 0),
            sites: measured,
            warmReadBytes,
            nullHeapDelta,
            controlBytes,
        })}\n`,
    );
} finally {
    subject.dispose();
    session.disconnect();
}
