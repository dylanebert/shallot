import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { sampleAllocation, siteTable, windowBytes } from "../first-person-allocation/allocation";

// Manual diagnosis of the engine's new Node-tier red. Keep the engine allocation check active.
test("isolate native GPU-copy handle allocation against the allocation-free upload control", async () => {
    const entry = resolve(import.meta.dir, "copy.entry.ts");
    const upload = await sampleAllocation(entry, { warm: 6000, frames: 600, input: "upload" });
    const copy = await sampleAllocation(entry, { warm: 6000, frames: 600, input: "copy" });
    expect(windowBytes({ sites: upload.control })).toBeGreaterThan(0);
    expect(windowBytes({ sites: copy.control })).toBeGreaterThan(0);
    console.info("[gpu-copy-allocation] writeBuffer control\n" + siteTable(upload));
    console.info("[gpu-copy-allocation] copyBufferToBuffer\n" + siteTable(copy));
    for (const window of upload.windows) expect(windowBytes(window)).toBe(0);
    for (const window of copy.windows) expect(windowBytes(window)).toBeGreaterThan(0);
}, 0);
