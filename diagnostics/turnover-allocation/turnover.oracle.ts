import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { type AllocationFrame, allocationFailure, type AllocationSample, type AllocationSite, sampleAllocation } from "../first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../first-person-allocation/allocation-admissions";

setDefaultTimeout(240_000);

const membershipOwners = /^(writeMeshBounds|registerDraws|publishMeshInstanceDraws|publishProducerDraws|rebuildMeshCounts|syncBuffers) src\/standard\/rendering\//;
const membership = (row: AllocationSite) => row.stack?.some((frame) => membershipOwners.test(frame.site)) ?? false;
const topJavaScript = (row: AllocationSite): AllocationFrame | undefined => row.stack?.findLast((frame) => !frame.native);
const through = (pattern: RegExp) => (row: AllocationSite) => row.stack?.some((frame) => pattern.test(frame.site)) ?? false;

// Each open owner group names its sites; probe.ts reproduces each cause. A group's todo turns red
// once its fix lands, and its rows then join the assertion. Standard's membership refresh is never a group.
const groups = [
    {
        name: "ECS and mesh registry turnover",
        sites: through(/^((add|remove) src\/(engine\/ecs\/world|core\/mesh\/mesh)\.ts|attachComponent src\/engine\/ecs\/table\.ts):/),
    },
    {
        name: "forward's background path",
        // backdrop's fresh pick is inlined through the opaque callback, so recordMainPass is its top frame.
        sites: (row: AllocationSite) =>
            through(/^(backgroundGroup|layoutResources|validateMeshBindingOverrides|backdrop) src\/standard\/rendering\/forward\.ts:/)(row) ||
            /^recordMainPass src\/core\/rendering\/phases\.ts:/.test(topJavaScript(row)?.site ?? ""),
    },
    { name: "orbit and sky", sites: through(/^update src\/extras\/(orbit|sky)\/index\.ts:/) },
    {
        name: "voxel's own passes",
        sites: (row: AllocationSite) => /^\S+ examples\/voxel\/src\/(terrain|placement)\.ts:/.test(topJavaScript(row)?.site ?? ""),
    },
];
const group = (row: AllocationSite) => (membership(row) ? undefined : groups.find((owner) => owner.sites(row)));

let sampled: Promise<AllocationSample> | undefined;
const sample = () =>
    (sampled ??= sampleAllocation(resolve(import.meta.dir, "turnover.entry.ts"), {
        warm: 2_400,
        frames: 60,
        retainOptimizedSites: true,
    }));
const only = (sample: AllocationSample, keep: (row: AllocationSite) => boolean) => ({
    ...sample,
    windows: sample.windows.map((window) => ({ ...window, sites: window.sites.filter(keep) })),
});

test("warm voxel turnover allocates nothing outside the admitted wrappers and the open owner groups", async () => {
    const result = await sample();
    expect(result.control.some(membership), "growth control must attribute allocation to a named standard membership site").toBe(true);
    const failure = allocationFailure(only(result, (row) => !isAdmittedAllocationSite(row) && !group(row)));
    if (failure) throw new Error(failure);
});

for (const owner of groups)
    test.todo(`warm voxel turnover allocates nothing in ${owner.name}`, async () => {
        const result = await sample();
        const failure = allocationFailure(only(result, (row) => !isAdmittedAllocationSite(row) && group(row) === owner));
        if (failure) throw new Error(failure);
    });
