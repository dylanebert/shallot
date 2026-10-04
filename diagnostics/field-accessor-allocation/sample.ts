// bun diagnostics/field-accessor-allocation/sample.ts
// Node sampler bytes per window for each row, then JSC ns per entity (median and spread of 7 runs).
import { resolve } from "node:path";
import { sampleAllocation, windowBytes } from "../first-person-allocation/allocation";
import create, { ENTITIES, ROWS } from "./subject.entry";

const entry = resolve(import.meta.dir, "subject.entry.ts");
for (const row of ROWS) {
    const sample = await sampleAllocation(entry, { warm: 600, frames: 600, input: row });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const sites = [...new Set(sample.windows.flatMap((w) => w.sites.map((s) => s.site)))];
    console.log(
        JSON.stringify({ row, bytes: sample.windows.map(windowBytes), frames: sample.windows[0].frames, sites }),
    );
}
console.log(JSON.stringify({ node: (await Bun.$`node --version`.text()).trim(), bun: Bun.version }));

const STEPS = 20000;
for (const row of ROWS) {
    const subject = await create(row);
    for (let i = 0; i < STEPS; i++) subject.step();
    const runs: number[] = [];
    for (let r = 0; r < 7; r++) {
        const start = performance.now();
        for (let i = 0; i < STEPS; i++) subject.step();
        runs.push(((performance.now() - start) * 1e6) / STEPS / ENTITIES);
    }
    subject.dispose();
    runs.sort((a, b) => a - b);
    console.log(
        JSON.stringify({ row, jscNsPerEntity: { median: +runs[3].toFixed(2), min: +runs[0].toFixed(2), max: +runs[6].toFixed(2) } }),
    );
}
