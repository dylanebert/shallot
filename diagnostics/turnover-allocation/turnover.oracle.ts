import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, type AllocationSite } from "../first-person-allocation/allocation";

setDefaultTimeout(240_000);
const owners = /^(writeMeshBounds|registerDraws|publishMeshInstanceDraws|publishProducerDraws|rebuildMeshCounts|syncBuffers) src\/standard\/rendering\//;
const membership = (row: AllocationSite) => row.stack?.some((frame) => owners.test(frame.site)) ?? false;

test("warm voxel turnover allocates nothing in standard membership refresh", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "turnover.entry.ts"), {
        warm: 2_400,
        frames: 60,
        retainOptimizedSites: true,
    });
    expect(sample.control.some(membership), "growth control must attribute allocation to a named standard membership site").toBe(true);
    for (const window of sample.windows) {
        const totals = new Map<string, number>();
        for (const row of window.sites.filter((row) => !membership(row)))
            totals.set(row.site, (totals.get(row.site) ?? 0) + row.count);
        console.log(window.label, "other owners (sample counts)", Object.fromEntries(totals));
        const refresh = new Map<string, number>();
        for (const row of window.sites.filter(membership))
            refresh.set(row.site, (refresh.get(row.site) ?? 0) + row.count);
        console.log(window.label, "membership (sample counts)", Object.fromEntries(refresh));
    }
    const failure = allocationFailure({ ...sample, windows: sample.windows.map((window) => ({ ...window, sites: window.sites.filter(membership) })) });
    if (failure) throw new Error(failure);
});
