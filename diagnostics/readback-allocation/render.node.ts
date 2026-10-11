import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";
import { attribute } from "../first-person-allocation/allocation-sampler.mjs";

import { CEILING } from "../../scripts/test-tiers";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";

setDefaultTimeout(CEILING.node);

/**
 * Only exact WebGPU expressions and the owned submission-fence reaction may admit sampled bytes,
 * with no application JavaScript beneath their wrappers. Submission counts are tested separately.
 */
const exactAllocationExpressions = new Map(
    gpuAllocationExpressions.map((entry) => [entry.site, { functionName: entry.fn.name }]),
);

function isExactAllocationSite(row: {
    expression?: string;
    stack?: readonly {
        functionName: string;
        site: string;
        expression?: string;
        native: boolean;
    }[];
}): boolean {
    if (!row.expression || !row.stack) return false;
    const expected = exactAllocationExpressions.get(row.expression);
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
        top.site === `${expected.functionName} ${row.expression}` &&
        top.expression === row.expression &&
        row.stack.slice(topJavaScript + 1).every((frame) => frame.native)
    );
}

test("allocation exceptions require exact call expressions, not containing functions", () => {
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
        isExactAllocationSite({
            expression: encoder.site,
            stack: [
                {
                    functionName: "frameEncoder",
                    site: "frameEncoder src/engine/ecs/world.ts:210:17",
                    native: false,
                },
                {
                    functionName: "createFrameEncoder",
                    site: `createFrameEncoder ${encoder.site}`,
                    native: false,
                    expression: encoder.site,
                },
                { functionName: "WebGPU", site: "WebGPU <native>", native: true },
            ],
        }),
    ).toBe(true);
    expect(
        isExactAllocationSite({
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
        isExactAllocationSite({
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

    expect(gpuAllocationExpressions.map((entry) => entry.expression)).toEqual([
        "device.createCommandEncoder(descriptor)",
        "encoder.finish()",
        "queue.submit(buffers)",
        "queue.onSubmittedWorkDone()",
        "fence.then(complete, settle)",
        "encoder.beginRenderPass(descriptor)",
        "encoder.beginComputePass(descriptor)",
        "pass.end()",
        "pass.end()",
    ]);
    const passEnds = gpuAllocationExpressions.filter((entry) => entry.expression === "pass.end()");
    expect(passEnds.map((entry) => entry.fn.name)).toEqual(["endRenderPass", "endComputePass"]);
    for (const passEnd of passEnds) {
        const frame = {
            functionName: passEnd.fn.name,
            site: `${passEnd.fn.name} src/engine/runtime/allocation-site.ts:${passEnd.fn.name === "endRenderPass" ? "111:30" : "115:31"}`,
            native: false,
            expression: passEnd.site,
        };
        expect(
            isExactAllocationSite({
                expression: passEnd.site,
                stack: [
                    { ...frame, site: `${passEnd.fn.name} ${passEnd.site}` },
                    { functionName: "end", site: "end <native>", native: true },
                ],
            }),
        ).toBe(true);
        expect(
            isExactAllocationSite({
                expression: passEnd.site,
                stack: [
                    frame,
                    { functionName: "applicationCallback", site: "callback app.ts:1:1", native: false },
                ],
            }),
        ).toBe(false);
    }
});

test("exact allocation exceptions leave ordinary renderer sites red", () => {
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
                    {
                        site: "endRenderPass src/engine/runtime/allocation-site.ts:111:30",
                        bytes: 16,
                        count: 1,
                        stack: [
                            {
                                functionName: "endRenderPass",
                                site: "endRenderPass src/engine/runtime/allocation-site.ts:111:30",
                                native: false,
                            },
                        ],
                    },
                ],
                frames: 120,
                framesAtMost: 120,
            },
            { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
            { label: "A/A repeat", sites: [], frames: 120, framesAtMost: 120 },
        ],
    };
    expect(isExactAllocationSite(sample.windows[0]!.sites[0]!)).toBe(false);
    expect(isExactAllocationSite(sample.windows[0]!.sites[1]!)).toBe(false);
    expect(allocationFailure(sample)).toContain("queueTransparent src/standard/rendering/forward.ts:867");
    expect(allocationFailure(sample)).toContain("endRenderPass src/engine/runtime/allocation-site.ts:111:30");
});

test("the sampler maps an admitted expression only at its exact source column", () => {
    const entry = gpuAllocationExpressions[0]!;
    const [, line, exactColumn] = /:(\d+):(\d+)$/.exec(entry.site)!;
    const attributeAt = (columnNumber?: number) => {
        const run = {
            functionName: "stepChunk",
            url: "sampler.mjs",
            scriptId: "1",
            lineNumber: 0,
            columnNumber: 0,
        };
        const expression = {
            functionName: entry.fn.name,
            url: "subject.mjs",
            scriptId: "2",
            lineNumber: Number(line) - 1,
            columnNumber,
        };
        return attribute(
            {
                head: {
                    id: 0,
                    callFrame: run,
                    children: [{ id: 1, callFrame: expression, children: [] }],
                },
                samples: [{ nodeId: 1, size: 8 }],
            },
            (frame) => (frame.functionName === "stepChunk" ? "stepChunk sampler.mjs:1" : undefined),
            (frame) =>
                `${frame.functionName} src/engine/runtime/allocation-site.ts:${frame.lineNumber + 1}:${Number.isInteger(frame.columnNumber) ? frame.columnNumber + 1 : "?"}`,
            [{ functionName: entry.fn.name, expressionSite: entry.site, expression: entry.expression }],
        )[0];
    };
    const expectedColumn = Number(exactColumn);
    expect(attributeAt(expectedColumn - 1)?.expression).toBe(entry.site);
    expect(attributeAt(expectedColumn)?.expression).toBeUndefined();
    expect(attributeAt()?.expression).toBeUndefined();
});

test("rendering allocates no JavaScript heap beyond exact GPU and fence expressions", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), {
        warm: 6000,
        frames: 600,
    });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((row) => !isExactAllocationSite(row)),
        })),
    });
    if (failure) throw new Error(failure);
});
