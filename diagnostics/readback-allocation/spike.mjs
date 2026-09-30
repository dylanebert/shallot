// Manual WebGPU readback allocation measurement. Run with node --expose-gc by path.
// Measures two result shapes over the same reused staging buffer; not an engine implementation.
import { Session } from "node:inspector/promises";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await bounded("readback adapter", gpu.requestAdapter());
if (!adapter) throw new Error("readback adapter unavailable");
const device = await bounded("readback device", adapter.requestDevice());
let validation;
device.addEventListener("uncapturederror", (event) => { validation ??= event.error; });
const session = new Session();
session.connect();
await session.post("HeapProfiler.enable");
const frames = 600;
const url = import.meta.url;

async function bounded(label, promise) {
    let timer;
    try {
        return await Promise.race([promise, new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error(`${label} exceeded 750 ms`)), 750);
        })]);
    } finally { clearTimeout(timer); }
}

async function measure(size, copy) {
    const source = device.createBuffer({ size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bytes = new Uint8Array(size);
    let checksum = 0;
    let pending;
    // Persistent callback identities: no fresh closures or application promises per frame.
    function consume() {
        const view = new Uint8Array(staging.getMappedRange());
        if (copy) bytes.set(view);
        checksum ^= view[0];
        staging.unmap();
    }
    function fail(error) { throw error; }
    function encode() {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, staging, 0, size);
        device.queue.submit([encoder.finish()]);
    }
    function map() {
        pending = staging.mapAsync(GPUMapMode.READ).then(consume, fail);
    }
    async function run(n) {
        for (let i = 0; i < n; i++) {
            encode();
            map();
            await bounded("staging map and result consumption", pending);
            if (validation) throw validation;
        }
    }
    try {
        await run(1200);
        globalThis.gc();
        await session.post("HeapProfiler.startSampling", {
            samplingInterval: 1,
            includeObjectsCollectedByMajorGC: true,
            includeObjectsCollectedByMinorGC: true,
        });
        await run(frames);
        const { profile } = await session.post("HeapProfiler.stopSampling");
        const start = performance.now();
        await run(frames);
        const ms = performance.now() - start;
        const owners = new Map();
        function visit(node, owner) {
            const frame = node.callFrame;
            if (frame.url === url && ["encode", "map", "consume"].includes(frame.functionName)) owner = frame.functionName;
            if (owner) owners.set(node.id, owner);
            for (const child of node.children) visit(child, owner);
        }
        visit(profile.head);
        const totals = { encode: 0, map: 0, consume: 0 };
        for (const sample of profile.samples) {
            const owner = owners.get(sample.nodeId);
            if (owner) totals[owner] += sample.size;
        }
        return { size, shape: copy ? "persistent byte destination" : "mapped event consumption", frames, msPerFrame: ms / frames,
            sampledBytesPerFrame: Object.fromEntries(Object.entries(totals).map(([key, value]) => [key, value / frames])), checksum };
    } finally { source.destroy(); staging.destroy(); }
}

try {
    console.log(JSON.stringify({ adapter: {
        vendor: adapter.info.vendor, architecture: adapter.info.architecture,
        device: adapter.info.device, description: adapter.info.description,
    }, node: process.version }));
    for (const size of [4, 65536]) {
        for (const copy of [false, true]) console.log(JSON.stringify(await measure(size, copy)));
    }
} finally {
    session.disconnect();
    device.destroy();
}
