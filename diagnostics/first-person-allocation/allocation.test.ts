import { expect, test } from "bun:test";
import { dirname, resolve, win32 } from "node:path";
import { fileURLToPath } from "node:url";
import { attribute, originalPosition, prepareSamplingSession, subjectSite } from "./allocation-sampler.mjs";
import { isAdmittedAllocationSite } from "./allocation-admissions";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";
import {
    type AllocationSample,
    allocatesNothing,
    allocationFailure,
    windowBytes,
    tracedSample,
    TIER_FLAGS,
} from "./allocation";
import { normalizePath } from "../../src/project/paths";

test("a sampling session starts after one collected and drained subject step", async () => {
    const order: string[] = [];
    await prepareSamplingSession(
        {
            warmSamplingSessions: true,
            step: () => order.push("step"),
            wait: async () => { order.push("wait"); },
        },
        () => order.push("collect"),
    );
    order.push("startSampling");
    expect(order).toEqual(["collect", "step", "wait", "startSampling"]);
    const otherSubject: string[] = [];
    await prepareSamplingSession({ warmSamplingSessions: false, step: () => otherSubject.push("step") }, () => otherSubject.push("collect"));
    expect(otherSubject).toEqual(["collect"]);
});

test("the sampler keeps lazy feedback but disables allocation-site pretenuring for every window", () => {
    expect(TIER_FLAGS).not.toContain("--no-lazy-feedback-allocation");
    expect(TIER_FLAGS).toEqual([
        "--no-concurrent-recompilation",
        "--no-allocation-site-pretenuring",
        "--invocation-count-for-maglev=10",
        "--invocation-count-for-turbofan=10",
        "--no-wasm-inlining",
    ]);
});

test("source-map sites resolve Windows-shaped file URLs as filesystem paths", () => {
    const url = "file:///D:/a%20b/shallot/src/engine/runtime/allocation-site.ts";
    const root = fileURLToPath("file:///D:/a%20b/shallot");
    const map = {
        findEntry: () => ({
            originalSource: url,
            originalLine: 8,
            originalColumn: 34,
        }),
    };
    const frame = { lineNumber: 0, columnNumber: 0 };

    expect(originalPosition(frame, map, dirname(root), win32)).toEqual({
        source: fileURLToPath(url),
        line: 8,
        column: 34,
    });
    const anonymousSite = subjectSite(frame, map, dirname(root), "bundle", root, win32);
    expect(anonymousSite).toBe("(anonymous) src/engine/runtime/allocation-site.ts:9:35");

    const wrapper = gpuAllocationExpressions.find((entry) => entry.fn.name === "finishFrameEncoder")!;
    const site = subjectSite(
        { ...frame, functionName: wrapper.fn.name },
        map,
        dirname(root),
        "bundle",
        root,
        win32,
    );
    expect(site).toBe(`${wrapper.fn.name} ${wrapper.wrapperSite}`);
    expect(
        isAdmittedAllocationSite({
            boundary: wrapper.wrapperSite,
            stack: [
                {
                    functionName: wrapper.fn.name,
                    site,
                    native: false,
                    boundary: wrapper.wrapperSite,
                },
            ],
        }),
    ).toBe(true);
});

const ENTRY = resolve(import.meta.dir, "allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");
/** A bundle input's absolute path with `/` separators, so suffix checks hold on Windows. */
const absolute = (path: string) => normalizePath(resolve(path));
/** A bundle input's root-relative path with `/` separators. */
const rooted = (path: string) => absolute(resolve(ROOT, path)).slice(ROOT.length + 1);

test("a Node allocation import does not load the display-only oracle or Hyprland instrument", async () => {
    const built = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation.ts")],
        metafile: true,
        external: ["playwright", "webgpu", "chromium-bidi"],
        target: "bun",
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`allocation import graph failed: ${built.logs.map(String).join("\\n")}`);
    const inputs = Object.keys(built.metafile.inputs).map(absolute);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display.ts"))).toBe(false);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display-seat.ts"))).toBe(false);

    const node = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation-sampler.mjs")],
        metafile: true,
        target: "node",
        external: ["webgpu"],
    });
    if (!node.success || node.metafile === undefined)
        throw new Error(`Node sampler graph failed: ${node.logs.map(String).join("\\n")}`);
    const nodeInputs = Object.keys(node.metafile.inputs).map(absolute);
    expect(nodeInputs.some((path) => path.endsWith("/allocation-sampler.mjs"))).toBe(true);
    expect(nodeInputs.some((path) => path.endsWith("/display.ts"))).toBe(false);
    const nodeImports = Object.values(node.metafile.inputs).flatMap((input) => input.imports);
    expect(nodeImports.some((edge) => edge.external && edge.path === "webgpu")).toBe(true);
    expect(nodeImports.some((edge) => edge.external && /^(?:bun:|playwright)/.test(edge.path))).toBe(
        false,
    );
});

// Profiler modules: the `profile` extra, which owns the physics step's timing clock.
const PROFILER = [/^src\/extras\/profile\//];

test("the allocation-gated first-person composition carries no timing or profiling module, so its default step does no diagnostics work", async () => {
    const built = await Bun.build({
        entrypoints: [ENTRY],
        target: "node",
        format: "esm",
        metafile: true,
        plugins: [
            {
                name: "shallot-source",
                setup(build) {
                    build.onResolve({ filter: /^@dylanebert\/shallot$/ }, () => ({
                        path: resolve(ROOT, "src/index.ts"),
                    }));
                },
            },
        ],
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`gated bundle failed: ${built.logs.map(String).join("\n")}`);
    const inputs = built.metafile.inputs;
    const paths = Object.keys(inputs).map(rooted);
    // Non-vacuity: the graph reaches the physics step whose timers this row is about.
    if (!paths.includes("src/standard/physics/solver/step.ts"))
        throw new Error(
            `inconclusive: gated bundle graph lacks the physics step (${paths.length} modules)`,
        );
    const importers = (module: string) =>
        Object.entries(inputs)
            .filter(([, input]) =>
                input.imports.some((edge) => resolve(ROOT, edge.path) === resolve(ROOT, module)),
            )
            .map(([path]) => path);
    const found = Object.keys(inputs).filter((path) =>
        PROFILER.some((pattern) => pattern.test(rooted(path))),
    );
    if (found.length !== 0)
        throw new Error(
            `gated bundle imports profiler modules:\n${found.map((path) => `  ${path} <- ${importers(path).join(", ")}`).join("\n")}`,
        );
});

test("optimization traces refuse even a zero-byte window, including anonymous OSR and repeated compiles", () => {
    const sample = { runtime: "planted trace", warm: 120, frames: 120, control: [], windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites: [], frames: 120, framesAtMost: 120 },
    ] };
    // V8's own records of the sampler's marker compiles (allocation-sampler.mjs traceMarks).
    const marker = (kind: "BEGIN" | "END", label: string) => {
        const fn = `<JSFunction SHALLOT_SAMPLE_${kind}_${Buffer.from(label).toString("hex")} (sfi = 0x9)>`;
        return [
            `[compiling method 0x8 ${fn} (target TURBOFAN_JS), mode: ConcurrencyMode::kSynchronous]`,
            `[completed compiling 0x8 ${fn} (target TURBOFAN_JS) - took 0.000, 0.029, 0.001 ms]`,
        ];
    };
    const lines = [
        ...marker("BEGIN", "after warm 120"),
        '[compiling method 0x1 <JSFunction late (sfi = 0x2)> (target MAGLEV), mode: ConcurrencyMode::kSynchronous]',
        '[completed compiling 0x1 <JSFunction late (sfi = 0x2)> (target MAGLEV)]',
        '[compiling method 0x1 <JSFunction (sfi = 0x3)> (target TURBOFAN_JS) OSR, mode: ConcurrencyMode::kSynchronous]',
        ...marker("END", "after warm 120"),
        ...marker("BEGIN", "after warm 240"),
        ...marker("END", "after warm 240"),
        ...marker("BEGIN", "A/A repeat"),
        ...marker("END", "A/A repeat"),
        JSON.stringify(sample),
    ];
    const result = tracedSample(lines.join("\n"));
    expect(result.windows[0].optimizations).toEqual(["late", "(anonymous)"]);
    expect(allocatesNothing(result)).toBe(false);
    expect(allocationFailure(result)).toContain("after warm 120; optimized late, (anonymous); no byte reading");
    expect(() => tracedSample(JSON.stringify(sample))).toThrow("missing allocation trace markers");
    expect(() => tracedSample(marker("BEGIN", "A/A repeat").join("\n"))).toThrow('incomplete allocation trace: window "A/A repeat" has no end marker');
});

const steadySample = (sites: AllocationSample["windows"][number]["sites"]) => ({
    warm: 120,
    windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites, frames: 120, framesAtMost: 120 },
    ],
});

test("the allocation gate passes when all three expected steady windows sample zero bytes at zero sites", () => {
    const sample = steadySample([]);
    expect(allocatesNothing(sample)).toBe(true);
    expect(allocationFailure(sample)).toBeUndefined();
});

test("the allocation gate reds with the names of all expected windows when a sampler returns no steady windows", () => {
    const sample = { warm: 120, windows: [] as AllocationSample["windows"] };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 120, after warm 240, A/A repeat",
    );
});

test("the allocation gate reds with the names of expected steady windows omitted by an incomplete sampler result", () => {
    const sample = {
        warm: 120,
        windows: [{ label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 }],
    };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 240, A/A repeat",
    );
});

test("the allocation gate reds on any sampled steady allocation and prints its site only for diagnosis", () => {
    const sample = steadySample([
        { site: "stepChunk src/transitional/character/sweep.ts:42", bytes: 96, count: 3 },
    ]);
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady play allocated JavaScript heap; sampler sites are diagnosis only:\n  A/A repeat: 96 B at stepChunk src/transitional/character/sweep.ts:42",
    );
});

test("an allocation with no JavaScript frame below the run frame is credited to the run frame, so the display control's breakpoint literal lands at the frame loop", () => {
    const frame = (id: number, functionName: string, url: string, children: unknown[] = []) => ({
        id,
        callFrame: { functionName, url, scriptId: String(id), lineNumber: 496, columnNumber: 12 },
        children,
    });
    // root -> frame (the page's run frame) -> the breakpoint condition, which has no script URL
    const profile = {
        head: frame(1, "(root)", "", [frame(2, "frame", "http://page/app.js", [frame(3, "", "")])]),
        samples: [{ nodeId: 3, size: 24 }],
    };
    const loop = "frame src/engine/app/index.ts:497:13";
    const sites = attribute(
        profile,
        (callFrame: { functionName: string }) => (callFrame.functionName === "frame" ? loop : undefined),
        () => undefined,
    );
    expect(sites.map((row: { site: string; bytes: number }) => [row.site, row.bytes])).toEqual([[loop, 24]]);
});
