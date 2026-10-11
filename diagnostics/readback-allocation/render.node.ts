import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

/**
 * Node attributes native WebGPU binding allocations to the nearest project call site. These are
 * frame/pass objects or their native command wrappers, not byte allowances. World submission
 * cardinality is asserted in frame-encoder.gpu.test.ts; any other source site remains a failure.
 */
const singleUseObjectSites = [
    ["frameEncoder", "src/engine/ecs/world.ts:205"], // createCommandEncoder
    ["step", "src/engine/ecs/scheduler.ts:200"], // finish, submit, and the submission fence
    ["recordMainPass", "src/core/rendering/phases.ts:300"], // main render-pass encoder
    ["update", "src/core/rendering/substrate.ts:280"], // current surface texture view
    ["update", "src/standard/rendering/cluster.ts:1374"], // four light compute-pass encoders
    ["encodeLightRaster", "src/standard/rendering/cluster.ts:1480"], // two light render-pass encoders
    ["update", "src/standard/rendering/preprocess.ts:212"], // mesh compute-pass encoder
    ["update", "src/core/rendering/tonemapping-state.ts:133"], // tonemapping render-pass encoder
    ["setBound", "src/standard/rendering/preprocess.ts:279"], // native compute-pass binding calls
    ["renderColor", "src/standard/rendering/forward.ts:744"], // native render-pass draw calls
] as const;

function isSingleUseObjectSite(site: string): boolean {
    return singleUseObjectSites.some(
        ([name, source]) => site.startsWith(`${name} `) && site.endsWith(source),
    );
}

test("the WebGPU allocation exception leaves ordinary renderer sites red", () => {
    const sample = {
        warm: 120,
        windows: [
            {
                label: "after warm 120",
                sites: [
                    {
                        site: "queueTransparent src/standard/rendering/forward.ts:867",
                        bytes: 16,
                        count: 1,
                    },
                ],
                frames: 120,
                framesAtMost: 120,
            },
            { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
            { label: "A/A repeat", sites: [], frames: 120, framesAtMost: 120 },
        ],
    };
    expect(isSingleUseObjectSite(sample.windows[0]!.sites[0]!.site)).toBe(false);
    expect(allocationFailure(sample)).toContain("queueTransparent src/standard/rendering/forward.ts:867");
});

test("rendering allocates no JavaScript heap beyond its required single-use GPU objects", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), {
        warm: 6000,
        frames: 600,
    });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((row) => !isSingleUseObjectSite(row.site)),
        })),
    });
    if (failure) throw new Error(failure);
});
