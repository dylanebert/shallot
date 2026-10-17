import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";
import {
    allocationWrapperFailure,
    assertAllocationWrappers,
    isAdmittedAllocationSite,
} from "../first-person-allocation/allocation-admissions";
import { attribute } from "../first-person-allocation/allocation-sampler.mjs";

import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";

// The qualified 12,000/600-frame sample takes about 11 s on an Apple M4 Max.
setDefaultTimeout(120_000);

const allocationSourcePath = resolve(import.meta.dir, "../../src/engine/runtime/allocation-site.ts");
const allocationSource = () => readFileSync(allocationSourcePath, "utf8");
const bindingWrappers = [
    "setComputePipeline",
    "setComputeBindGroup",
    "executeRenderBundles",
    "dispatchWorkgroups",
    "dispatchWorkgroups2",
    "setIndexBuffer",
    "setViewport",
    "drawIndexedIndirect",
    "setRenderPipeline",
    "setRenderBindGroup",
    "draw",
];

test("allocation exceptions require each named wrapper to keep its complete signature and API expression", () => {
    const source = allocationSource();
    assertAllocationWrappers(source);
    for (const entry of gpuAllocationExpressions)
        expect(allocationWrapperFailure(source, entry)).toBeUndefined();

    const submit = gpuAllocationExpressions.find((entry) => entry.fn.name === "submitFrame")!;
    const expression = "    queue.submit(buffers);";
    const array = source.replace(expression, `${expression} const witness = new Array(16);`);
    const closure = source.replace(expression, `${expression} const witness = () => buffers.length;`);
    const submitSignature = "submitFrame(queue: GPUQueue, buffers: readonly GPUCommandBuffer[]): void";
    const rest = source.replace(
        submitSignature,
        "submitFrame(queue: GPUQueue, buffers: readonly GPUCommandBuffer[], ...spare: unknown[]): void",
    );
    const defaultParameter = source.replace(
        submitSignature,
        "submitFrame(queue: GPUQueue, buffers: readonly GPUCommandBuffer[] = []): void",
    );
    const bodyFailure = "allocation exception wrapper submitFrame must contain only queue.submit(buffers)";
    const parameterFailure = `allocation exception wrapper submitFrame must have exactly parameters (${submit.parameters})`;
    expect(allocationWrapperFailure(array, submit)).toBe(bodyFailure);
    expect(allocationWrapperFailure(closure, submit)).toBe(bodyFailure);
    expect(allocationWrapperFailure(rest, submit)).toBe(parameterFailure);
    expect(allocationWrapperFailure(defaultParameter, submit)).toBe(parameterFailure);
    expect(() => assertAllocationWrappers(rest)).toThrow(parameterFailure);
    expect(() => assertAllocationWrappers(defaultParameter)).toThrow(parameterFailure);
    const encoder = gpuAllocationExpressions[0]!;
    expect(
        isAdmittedAllocationSite({
            boundary: encoder.wrapperSite,
            stack: [
                {
                    functionName: "frameEncoder",
                    site: "frameEncoder src/engine/ecs/world.ts:210:17",
                    native: false,
                },
                {
                    functionName: "createFrameEncoder",
                    site: `createFrameEncoder ${encoder.wrapperSite}`,
                    native: false,
                    boundary: encoder.wrapperSite,
                },
                { functionName: "WebGPU", site: "WebGPU <native>", native: true },
            ],
        }),
    ).toBe(true);
    expect(
        isAdmittedAllocationSite({
            boundary: encoder.wrapperSite,
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
        isAdmittedAllocationSite({
            boundary: encoder.wrapperSite,
            stack: [
                {
                    functionName: "createFrameEncoder",
                    site: `createFrameEncoder ${encoder.wrapperSite}`,
                    native: false,
                    boundary: encoder.wrapperSite,
                },
                { functionName: "webgpuBinding", site: "webgpuBinding gpu:binding", native: false },
            ],
        }),
    ).toBe(false);

    expect(gpuAllocationExpressions.map((entry) => entry.fn.name)).toEqual([
        "createFrameEncoder",
        "finishFrameEncoder",
        "submitFrame",
        "submissionFence",
        "continueSubmissionFence",
        "beginRenderPass",
        "beginComputePass",
        "endRenderPass",
        "endComputePass",
    ]);
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
            site: `${passEnd.fn.name} ${passEnd.wrapperSite}`,
            native: false,
            boundary: passEnd.wrapperSite,
        };
        expect(
            isAdmittedAllocationSite({
                boundary: passEnd.wrapperSite,
                stack: [
                    frame,
                    { functionName: "end", site: "end <native>", native: true },
                ],
            }),
        ).toBe(true);
        expect(
            isAdmittedAllocationSite({
                boundary: passEnd.wrapperSite,
                stack: [
                    frame,
                    { functionName: "applicationCallback", site: "callback app.ts:1:1", native: false },
                ],
            }),
        ).toBe(false);
    }
});

test("wrapper admission leaves renderer and caller-closure samples red", () => {
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
                        site: "step src/engine/ecs/scheduler.ts:200:9",
                        bytes: 96,
                        count: 1,
                        stack: [
                            {
                                functionName: "step",
                                site: "step diagnostics/readback-allocation/render.entry.ts:76:15",
                                native: false,
                            },
                            {
                                functionName: "step",
                                site: "step src/engine/ecs/scheduler.ts:200:9",
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
    expect(isAdmittedAllocationSite(sample.windows[0]!.sites[0]!)).toBe(false);
    expect(isAdmittedAllocationSite(sample.windows[0]!.sites[1]!)).toBe(false);
    expect(allocationFailure(sample)).toContain("queueTransparent src/standard/rendering/forward.ts:867");
    expect(allocationFailure(sample)).toContain("step src/engine/ecs/scheduler.ts:200:9");
});

test("the sampler admits a wrapper only at its declaration position", () => {
    const entry = gpuAllocationExpressions[0]!;
    const [, line, exactColumn] = /:(\d+):(\d+)$/.exec(entry.wrapperSite)!;
    const attributeAt = (columnNumber?: number) => {
        const run = {
            functionName: "stepChunk",
            url: "sampler.mjs",
            scriptId: "1",
            lineNumber: 0,
            columnNumber: 0,
        };
        const wrapper = {
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
                    children: [{ id: 1, callFrame: wrapper, children: [] }],
                },
                samples: [{ nodeId: 1, size: 8 }],
            },
            (frame) => (frame.functionName === "stepChunk" ? "stepChunk sampler.mjs:1" : undefined),
            (frame) =>
                `${frame.functionName} src/engine/runtime/allocation-site.ts:${frame.lineNumber + 1}:${Number.isInteger(frame.columnNumber) ? frame.columnNumber + 1 : "?"}`,
            [{ functionName: entry.fn.name, wrapperSite: entry.wrapperSite, expression: entry.expression }],
        )[0];
    };
    const expectedColumn = Number(exactColumn);
    expect(attributeAt(expectedColumn - 1)?.boundary).toBe(entry.wrapperSite);
    expect(attributeAt(expectedColumn)?.boundary).toBeUndefined();
    expect(attributeAt()?.boundary).toBeUndefined();
});

test.each(["render", "profile"])("steady %s allocates no JavaScript heap beyond source-checked GPU and fence wrappers", async (composition) => {
    // The first frame after each profiler-session idle gap is collected but unprofiled.
    // Session setup and post-idle work land together on it, so timing cannot separate them;
    // one bounded allocation per gap is not the per-frame allocation that drives GC hitches.
    assertAllocationWrappers(allocationSource());
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), {
        // The composed gizmo and shadow paths finish optimizing after 12,000 frames.
        warm: 24_000,
        frames: 600,
        input: composition,
    });
    if (windowBytes({ sites: sample.control }) === 0) throw new Error("allocation control was invisible");
    for (const window of sample.windows) {
        if (window.optimizations?.length) continue;
        for (const functionName of bindingWrappers) {
            const bytes = window.sites
                .filter((row) => row.stack?.some((frame) => frame.functionName === functionName))
                .reduce((total, row) => total + row.bytes, 0);
            if (bytes !== 0)
                throw new Error(`${window.label}: unadmitted binding wrapper ${functionName} sampled ${bytes} B`);
        }
    }
    const failure = allocationFailure({
        ...sample,
        windows: sample.windows.map((window) => ({
            ...window,
            sites: window.sites.filter((row) => !isAdmittedAllocationSite(row)),
        })),
    });
    console.info(`${composition} allocation`, {
        controlBytes: windowBytes({ sites: sample.control }),
        windows: sample.windows.map((window) => ({
            label: window.label,
            frames: window.frames,
            optimizations: window.optimizations ?? [],
            unadmittedBytes: windowBytes({ sites: window.sites.filter((row) => !isAdmittedAllocationSite(row)) }),
        })),
    });
    if (failure) throw new Error(failure);
});
