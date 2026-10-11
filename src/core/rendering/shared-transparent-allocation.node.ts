import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    sampleAllocation,
    windowBytes,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("composed transparent World.step allocation sites are attributed after warm-up", async () => {
    const entry = resolve(
        import.meta.dir,
        "../../../diagnostics/first-person-allocation/shared-transparent.entry.ts",
    );
    // Preserve attribution through JIT changes; this diagnoses existing whole-frame sites, not zero allocations.
    const sample = await sampleAllocation(entry, {
        warm: 12000,
        frames: 600,
        retainOptimizedSites: true,
    });
    const controlBytes = windowBytes({ sites: sample.control });
    if (controlBytes === 0) throw new Error("allocation control was invisible");
    const expected = ["after warm 12000", "after warm 24000", "A/A repeat"];
    if (sample.windows.length !== expected.length)
        throw new Error(`expected ${expected.length} steady windows, got ${sample.windows.length}`);
    for (let i = 0; i < expected.length; i++) {
        const window = sample.windows[i]!;
        if (window.label !== expected[i] || window.frames !== 600)
            throw new Error(
                `unexpected allocation window ${window.label} (${window.frames} frames)`,
            );
        if (!windowBytes(window) || !window.sites.length)
            throw new Error(`${window.label} did not attribute the composed frame's sampled work`);
        if (window.sites.some((site) => site.site.includes("shared-transparent.entry.ts")))
            throw new Error(
                `${window.label} sampled an allocation site in the new composition entry`,
            );
        if (window.sites.some((site) => site.site.includes("expandRetained")))
            throw new Error(
                `${window.label} sampled the retained-line query site changed in this follow-up`,
            );
    }
    const inventory = new Map<string, { bytes: number[]; count: number[] }>();
    for (let index = 0; index < sample.windows.length; index++) {
        for (const site of sample.windows[index]!.sites) {
            let totals = inventory.get(site.site);
            if (!totals) {
                totals = {
                    bytes: Array(sample.windows.length).fill(0),
                    count: Array(sample.windows.length).fill(0),
                };
                inventory.set(site.site, totals);
            }
            totals.bytes[index] = site.bytes;
            totals.count[index] = site.count;
        }
    }
    const sites = Array.from(inventory, ([site, windows]) => ({ site, ...windows })).sort(
        (a, b) => b.bytes[0]! - a.bytes[0]!,
    );
    console.info(
        "composed transparent steady heap profile (nonzero frame-wide sites are attributed below)",
        JSON.stringify({
            controlBytes,
            windows: sample.windows.map((window) => ({
                label: window.label,
                frames: window.frames,
                bytes: windowBytes(window),
                optimizations: window.optimizations,
            })),
            sites,
        }),
    );
});
