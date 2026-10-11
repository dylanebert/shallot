import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";

import { CEILING } from "../../scripts/test-tiers";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";

setDefaultTimeout(CEILING.node);

/**
 * Only an exact WebGPU expression with no application JavaScript beneath its wrapper may admit a
 * sampled allocation. Submission cardinality is asserted separately in frame-encoder.gpu.test.ts.
 */
const exactGpuExpressions = new Map(
    gpuAllocationExpressions.map((entry) => [
        entry.site,
        { functionName: entry.fn.name, frameSource: "src/engine/runtime/allocation-site.ts" },
    ]),
);

function isExactGpuApiSite(row: {
    expression?: string;
    stack?: readonly {
        functionName: string;
        site: string;
        expression?: string;
        native: boolean;
    }[];
}): boolean {
    if (!row.expression || !row.stack) return false;
    const expected = exactGpuExpressions.get(row.expression);
    if (!expected) return false;
    let topJavaScript = -1;
    for (let i = row.stack.length - 1; i >= 0; i--) {
        if (!row.stack[i]!.native) {
            topJavaScript = i;
            break;
        }
    }
    const top = row.stack[topJavaScript];
    return (
        topJavaScript >= 0 &&
        top !== undefined &&
        top.functionName === expected.functionName &&
        top.site.startsWith(`${expected.functionName} ${expected.frameSource}:`) &&
        top.expression === row.expression &&
        row.stack.slice(topJavaScript + 1).every((frame) => frame.native)
    );
}

test("GPU exceptions require exact API expressions, not their containing functions", () => {
    const source = readFileSync(
        resolve(import.meta.dir, "../../src/engine/runtime/allocation-site.ts"),
        "utf8",
    ).split("\n");
    for (const entry of gpuAllocationExpressions) {
        const match = /^(.*):(\d+):(\d+)$/.exec(entry.site);
        if (!match) throw new Error(`invalid GPU allocation expression site: ${entry.site}`);
        const line = Number(match[2]);
        const column = Number(match[3]);
        expect(source[line - 1]?.slice(column - 1).startsWith(entry.expression)).toBe(true);
    }
    const encoder = gpuAllocationExpressions[0]!;
    expect(
        isExactGpuApiSite({
            expression: encoder.site,
            stack: [
                {
                    functionName: "frameEncoder",
                    site: "frameEncoder src/engine/ecs/world.ts:210:17",
                    native: false,
                },
                {
                    functionName: "createFrameEncoder",
                    site: "createFrameEncoder src/engine/runtime/allocation-site.ts:2:35",
                    native: false,
                    expression: encoder.site,
                },
                { functionName: "WebGPU", site: "WebGPU <native>", native: true },
            ],
        }),
    ).toBe(true);
    expect(
        isExactGpuApiSite({
            expression: encoder.site,
            stack: [
                {
                    functionName: "frameEncoder",
                    site: "frameEncoder src/engine/ecs/world.ts:210:17",
                    native: false,
                },
            ],
        }),
    ).toBe(false);
    expect(
        isExactGpuApiSite({
            expression: encoder.site,
            stack: [
                {
                    functionName: "createFrameEncoder",
                    site: "createFrameEncoder src/engine/runtime/allocation-site.ts:2:35",
                    native: false,
                    expression: encoder.site,
                },
                { functionName: "webgpuBinding", site: "webgpuBinding gpu:binding", native: false },
                {
                    functionName: "node:internal/perf/performance",
                    site: "node:internal/perf/performance:137:6",
                    native: true,
                },
            ],
        }),
    ).toBe(false);
});

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
    expect(isExactGpuApiSite(sample.windows[0]!.sites[0]!)).toBe(false);
    expect(allocationFailure(sample)).toContain("queueTransparent src/standard/rendering/forward.ts:867");
});

test("rendering allocates no JavaScript heap beyond exact WebGPU API sites", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), {
        warm: 6000,
        frames: 600,
    });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((row) => !isExactGpuApiSite(row)),
        })),
    });
    if (failure) throw new Error(failure);
});
