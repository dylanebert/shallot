import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { type AllocationFrame, allocationFailure, type AllocationSample, type AllocationSite, sampleAllocation } from "../first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../first-person-allocation/allocation-admissions";

setDefaultTimeout(240_000);

const membershipOwners = /^(writeMeshBounds|registerDraws|publishMeshInstanceDraws|publishProducerDraws|rebuildMeshCounts|syncBuffers) src\/standard\/rendering\//;
const membership = (row: AllocationSite) => row.stack?.some((frame) => membershipOwners.test(frame.site)) ?? false;

// A claim is one diagnosed site: its top JavaScript frame, the frames that call it nearest last, the native
// frames above it, and for an inlined leaf the sampled size that pins it.
type Claim = { callers: string[]; leaf: string; natives?: string[]; bytesPerSample?: number };
const claims = (row: AllocationSite, claim: Claim) => {
    const sites = row.stack?.map((frame) => frame.site) ?? [];
    const top = row.stack?.findLastIndex((frame) => !frame.native) ?? -1;
    const natives = claim.natives ?? [];
    return (
        top >= claim.callers.length &&
        sites[top] === claim.leaf &&
        sites.slice(top + 1).join("\n") === natives.join("\n") &&
        sites.slice(top - claim.callers.length, top).join("\n") === claim.callers.join("\n") &&
        (claim.bytesPerSample === undefined || row.bytes === claim.bytesPerSample * row.count)
    );
};

const place = "place examples/voxel/src/terrain.ts:99:15";
const worldAdd = "add src/engine/ecs/world.ts:777:11";
const backgroundGroup = "backgroundGroup src/standard/rendering/forward.ts:692:25";
const layoutResources = "layoutResources src/standard/rendering/forward.ts:332:25";
const orbit = "update src/extras/orbit/index.ts:152:11";
const sky = "update src/extras/sky/index.ts:102:11";
const runGroup = "runGroup src/engine/ecs/scheduler.ts:240:21";

// Each open owner group claims exactly its diagnosed sites; probe.ts reproduces each cause. A group's todo
// turns red once its fix lands, and its rows then join the assertion. Standard's membership refresh is never a group.
const groups: { name: string; claims: Claim[] }[] = [
    {
        name: "ECS and mesh registry turnover",
        claims: [
            { callers: [place], leaf: worldAdd },
            { callers: [place], leaf: "remove src/engine/ecs/world.ts:821:11" },
            { callers: [place], leaf: "remove src/core/mesh/mesh.ts:148:11" },
            { callers: [place, worldAdd], leaf: "attachComponent src/engine/ecs/table.ts:398:20", natives: ["set <native>"] },
            { callers: [place], leaf: "add src/core/mesh/mesh.ts:117:8", natives: ["set <native>"] },
        ],
    },
    {
        name: "forward's background path",
        claims: [
            { callers: [backgroundGroup], leaf: layoutResources },
            { callers: [backgroundGroup], leaf: layoutResources, natives: ["entries <native>"] },
            { callers: [layoutResources], leaf: "validateMeshBindingOverrides src/standard/rendering/forward.ts:251:45", natives: ["entries <native>"] },
            { callers: ["renderColor src/standard/rendering/forward.ts:747:21"], leaf: backgroundGroup, natives: ["keys <native>"] },
            // backdrop's `{ bg, ct }` is inlined through the opaque callback, leaving no forward frame;
            // the main pass's own frames and the pick's 40 B pin it.
            {
                callers: ["update src/core/rendering/phases.ts:372:11", "forEach <native>"],
                leaf: "recordMainPass src/core/rendering/phases.ts:306:24",
                bytesPerSample: 40,
            },
        ],
    },
    {
        name: "orbit and sky",
        claims: [
            { callers: [runGroup], leaf: orbit },
            { callers: [orbit], leaf: "(anonymous) src/engine/ecs/query.ts:26:12" },
            { callers: [orbit], leaf: "lookAtRotation src/engine/utils/math.ts:617:31" },
            { callers: [runGroup], leaf: sky },
            { callers: [sky, "packSky src/extras/sky/pack.ts:10:24"], leaf: "unpackColor src/engine/utils/color.ts:26:28" },
        ],
    },
    {
        name: "voxel's own passes",
        claims: [
            { callers: [runGroup], leaf: "update examples/voxel/src/terrain.ts:137:15" },
            { callers: [runGroup], leaf: "update examples/voxel/src/terrain.ts:296:23" },
        ],
    },
];
const claimants = (row: AllocationSite) =>
    membership(row) ? [] : groups.filter((owner) => owner.claims.some((claim) => claims(row, claim)));

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
    const shared = result.windows.flatMap((window) =>
        window.sites
            .filter((row) => claimants(row).length > 1)
            .map((row) => `${window.label}: ${row.site} is claimed by ${claimants(row).map((owner) => owner.name).join(" and ")}`),
    );
    if (shared.length > 0) throw new Error(`rows claimed by more than one group:\n${shared.join("\n")}`);
    const failure = allocationFailure(only(result, (row) => !isAdmittedAllocationSite(row) && claimants(row).length === 0));
    if (failure) throw new Error(failure);
});

for (const owner of groups)
    test.todo(`warm voxel turnover allocates nothing in ${owner.name}`, async () => {
        const result = await sample();
        const failure = allocationFailure(only(result, (row) => !isAdmittedAllocationSite(row) && claimants(row).includes(owner)));
        if (failure) throw new Error(failure);
    });

// Synthetic rows: a stack from the scheduler's runGroup to the leaf, with `bytes` per sample.
const frame = (site: string): AllocationFrame => ({ functionName: site.split(" ")[0]!, site, native: site.endsWith(" <native>") });
const row = (sites: string[], bytes: number): AllocationSite => ({
    site: sites.findLast((site) => !site.endsWith(" <native>"))!,
    bytes: bytes * 60,
    count: 60,
    stack: [runGroup, ...sites].map(frame),
});
const placing = ["update examples/voxel/src/terrain.ts:285:23", place];
const background = ["update src/core/rendering/phases.ts:372:11", "forEach <native>", "recordMainPass src/core/rendering/phases.ts:306:24", "renderColor src/standard/rendering/forward.ts:747:21", backgroundGroup];

test("each group claims exactly its diagnosed rows", () => {
    const diagnosed: [string, AllocationSite][] = [
        ["ECS and mesh registry turnover", row([...placing, worldAdd], 64)],
        ["ECS and mesh registry turnover", row([...placing, "remove src/engine/ecs/world.ts:821:11"], 52)],
        ["ECS and mesh registry turnover", row([...placing, "remove src/core/mesh/mesh.ts:148:11"], 48)],
        ["ECS and mesh registry turnover", row([...placing, worldAdd, "attachComponent src/engine/ecs/table.ts:398:20", "set <native>"], 488)],
        ["ECS and mesh registry turnover", row([...placing, "add src/core/mesh/mesh.ts:117:8", "set <native>"], 936)],
        ["forward's background path", row([...background, layoutResources], 112)],
        ["forward's background path", row([...background, layoutResources, "entries <native>"], 40)],
        ["forward's background path", row([...background, layoutResources, "validateMeshBindingOverrides src/standard/rendering/forward.ts:251:45", "entries <native>"], 40)],
        ["forward's background path", row([...background, "keys <native>"], 56)],
        ["forward's background path", row(background.slice(0, 3), 40)],
        ["orbit and sky", row([orbit], 35)],
        ["orbit and sky", row([orbit, "(anonymous) src/engine/ecs/query.ts:26:12"], 28)],
        ["orbit and sky", row([orbit, "lookAtRotation src/engine/utils/math.ts:617:31"], 80)],
        ["orbit and sky", row([sky], 56)],
        ["orbit and sky", row([sky, "packSky src/extras/sky/pack.ts:10:24", "unpackColor src/engine/utils/color.ts:26:28"], 96)],
        ["voxel's own passes", row(["update examples/voxel/src/terrain.ts:137:15"], 34)],
        ["voxel's own passes", row(["update examples/voxel/src/terrain.ts:296:23"], 34)],
    ];
    const stack = (claimed: AllocationSite) => claimed.stack!.map((f) => f.site).join(" <- ");
    const misclaimed = diagnosed
        .filter(([name, diagnosedRow]) => claimants(diagnosedRow).map((owner) => owner.name).join() !== name)
        .map(([name, diagnosedRow]) => `${name} does not alone claim ${stack(diagnosedRow)}`);
    const unclaimed = [
        // another allocation made directly in recordMainPass, of a size the backdrop pick is not
        row(background.slice(0, 3), 32),
        // another function in voxel's terrain.ts: the scroll system's own body
        row(["update examples/voxel/src/terrain.ts:285:23"], 34),
        // orbit's OrbitSmooth add, under World.add but not voxel's place
        row([orbit, worldAdd], 64),
    ];
    for (const unclaimedRow of unclaimed) {
        const names = claimants(unclaimedRow).map((owner) => owner.name);
        if (names.length > 0) misclaimed.push(`${names.join(" and ")} claimed ${stack(unclaimedRow)}`);
    }
    expect(misclaimed).toEqual([]);
});
