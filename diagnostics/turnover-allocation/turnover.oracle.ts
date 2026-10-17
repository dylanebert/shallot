import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation } from "../first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../first-person-allocation/allocation-admissions";

setDefaultTimeout(240_000);

const membershipOwners = /^(writeMeshBounds|registerDraws|publishMeshInstanceDraws|publishProducerDraws|rebuildMeshCounts|syncBuffers) src\/standard\/rendering\//;

test("warm voxel turnover allocates nothing outside the admitted wrappers", async () => {
    const result = await sampleAllocation(resolve(import.meta.dir, "turnover.entry.ts"), {
        warm: 2_400,
        frames: 60,
        retainOptimizedSites: true,
    });
    expect(result.control.some((row) => row.stack?.some((frame) => membershipOwners.test(frame.site))), "growth control must attribute allocation to a named standard membership site").toBe(true);
    const failure = allocationFailure({
        ...result,
        windows: result.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((row) => !isAdmittedAllocationSite(row)),
        })),
    });
    if (failure) throw new Error(failure);
});
