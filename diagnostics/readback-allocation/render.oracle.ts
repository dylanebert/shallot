import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { allocationFailure, sampleAllocation, windowBytes } from "../first-person-allocation/allocation";
import { attribute } from "../first-person-allocation/allocation-sampler.mjs";

import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";

// The qualified 6,000/600-frame sample takes about 33 s on Mesa lavapipe.
setDefaultTimeout(70_000);

const allocationSourcePath = resolve(import.meta.dir, "../../src/engine/runtime/allocation-site.ts");
const allocationSource = () => readFileSync(allocationSourcePath, "utf8");
const parserMetadata = new Set([
    "start", "end", "loc", "extra", "comments", "leadingComments", "trailingComments", "innerComments", "tokens", "errors",
]);
const admittedAllocationWrappers = new Map(
    gpuAllocationExpressions.map((entry) => [entry.wrapperSite, { functionName: entry.fn.name }]),
);
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

function syntaxShape(node: any): unknown {
    if (Array.isArray(node)) return node.map(syntaxShape);
    if (!node || typeof node !== "object") return node;
    return Object.fromEntries(
        Object.entries(node)
            .filter(([key]) => !["start", "end", "loc", "extra", "comments", "leadingComments", "trailingComments", "innerComments"].includes(key))
            .map(([key, value]) => [key, syntaxShape(value)]),
    );
}

function mentionsArguments(node: any): boolean {
    if (Array.isArray(node)) return node.some(mentionsArguments);
    if (!node || typeof node !== "object") return false;
    if (node.type === "Identifier" && node.name === "arguments") return true;
    return Object.entries(node)
        .filter(([key]) => !parserMetadata.has(key))
        .some(([, value]) => mentionsArguments(value));
}

function allocationWrapperFailure(
    sourceText: string,
    entry: (typeof gpuAllocationExpressions)[number],
): string | undefined {
    const functionName = entry.fn.name;
    let source;
    try {
        source = parse(sourceText, { sourceType: "module", plugins: ["typescript", "decorators-legacy"] });
    } catch {
        return `allocation exception wrapper ${functionName} source does not parse`;
    }
    const declarations: any[] = [];
    const visit = (node: any) => {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        if (node.type === "FunctionDeclaration" && node.id?.name === functionName) declarations.push(node);
        for (const [key, value] of Object.entries(node))
            if (!parserMetadata.has(key)) visit(value);
    };
    visit(source.program);
    if (declarations.length !== 1)
        return `allocation exception wrapper ${functionName} is missing or ambiguous`;
    const declaration = declarations[0]!;
    if (!declaration.id || !declaration.body)
        return `allocation exception wrapper ${functionName} has no body`;
    const namePosition = declaration.id.loc.end;
    const wrapperSite = `src/engine/runtime/allocation-site.ts:${namePosition.line}:${namePosition.column + 1}`;
    if (wrapperSite !== entry.wrapperSite)
        return `allocation exception wrapper ${functionName} moved from ${entry.wrapperSite} to ${wrapperSite}`;
    if (declaration.decorators?.length || declaration.params.some((parameter: any) => parameter.decorators?.length))
        return `allocation exception wrapper ${functionName} must not have decorators`;
    if (mentionsArguments(declaration.body))
        return `allocation exception wrapper ${functionName} must not use arguments`;
    let expectedDeclaration;
    try {
        const returnType = entry.returnType === undefined ? "" : `: ${entry.returnType}`;
        expectedDeclaration = parse(
            `function ${functionName}(${entry.parameters})${returnType} { ${entry.returns ? "return " : ""}${entry.expression}; }`,
            { sourceType: "module", plugins: ["typescript", "decorators-legacy"] },
        ).program.body[0];
    } catch {
        return `allocation exception wrapper ${functionName} has an invalid expected declaration`;
    }
    if (JSON.stringify(syntaxShape(declaration.params)) !== JSON.stringify(syntaxShape(expectedDeclaration?.params)))
        return `allocation exception wrapper ${functionName} must have exactly parameters (${entry.parameters})`;
    if (declaration.body.body.length !== 1)
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    const statement = declaration.body.body[0];
    const actual = entry.returns
        ? statement.type === "ReturnStatement" ? statement.argument : undefined
        : statement.type === "ExpressionStatement" ? statement.expression : undefined;
    if (!actual || actual.type !== "CallExpression")
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    let expected;
    try {
        expected = parse(entry.expression, { sourceType: "module", plugins: ["typescript", "decorators-legacy"] }).program.body[0];
    } catch {
        return `allocation exception wrapper ${functionName} has invalid expected expression`;
    }
    if (
        expected?.type !== "ExpressionStatement" ||
        expected.expression.type !== "CallExpression" ||
        JSON.stringify(syntaxShape(actual)) !== JSON.stringify(syntaxShape(expected.expression))
    )
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    if (JSON.stringify(syntaxShape(declaration)) !== JSON.stringify(syntaxShape(expectedDeclaration)))
        return `allocation exception wrapper ${functionName} must be a plain declaration containing only ${entry.returns ? "return " : ""}${entry.expression}`;
    return undefined;
}

function assertAllocationWrappers(sourceText: string): void {
    const failures = gpuAllocationExpressions
        .map((entry) => allocationWrapperFailure(sourceText, entry))
        .filter((failure): failure is string => failure !== undefined);
    if (failures.length > 0) throw new Error(failures.join("\n"));
}

function isAdmittedAllocationSite(row: {
    boundary?: string;
    stack?: readonly {
        functionName: string;
        site: string;
        boundary?: string;
        native: boolean;
    }[];
}): boolean {
    if (!row.boundary || !row.stack) return false;
    const expected = admittedAllocationWrappers.get(row.boundary);
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
        top.site === `${expected.functionName} ${row.boundary}` &&
        top.boundary === row.boundary &&
        row.stack.slice(topJavaScript + 1).every((frame) => frame.native)
    );
}

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

test("steady rendering allocates no JavaScript heap beyond source-checked GPU and fence wrappers", async () => {
    // The first frame after each profiler-session idle gap is collected but unprofiled.
    // Session setup and post-idle work land together on it, so timing cannot separate them;
    // one bounded allocation per gap is not the per-frame allocation that drives GC hitches.
    assertAllocationWrappers(allocationSource());
    const sample = await sampleAllocation(resolve(import.meta.dir, "render.entry.ts"), {
        warm: 6000,
        frames: 600,
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
    if (failure) throw new Error(failure);
});
