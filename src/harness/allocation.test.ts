import { expect } from "bun:test";
import { resolve } from "node:path";
import { declaredSiteFailures, sentinelFrames } from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";

const ENTRY = resolve(import.meta.dir, "../../examples/first-person/src/allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");

// Profiler modules: the `profile` extra, which owns the physics step's timing clock.
const PROFILER = [/^src\/extras\/profile\//];

check(
    "the gated first-person bundle imports no profiler module",
    {
        claim: "the allocation-gated first-person composition carries no timing or profiling module, so its default step does no diagnostics work",
    },
    async () => {
        const built = await Bun.build({
            entrypoints: [ENTRY],
            target: "node",
            format: "esm",
            metafile: true,
        });
        if (!built.success || built.metafile === undefined)
            throw new Error(`gated bundle failed: ${built.logs.map(String).join("\n")}`);
        const inputs = built.metafile.inputs;
        const paths = Object.keys(inputs).map((path) => resolve(ROOT, path).slice(ROOT.length + 1));
        // Non-vacuity: the graph reaches the physics step whose timers this row is about.
        if (!paths.includes("src/standard/physics/solver/step.ts"))
            throw new Error(
                `inconclusive: gated bundle graph lacks the physics step (${paths.length} modules)`,
            );
        const importers = (module: string) =>
            Object.entries(inputs)
                .filter(([, input]) =>
                    input.imports.some(
                        (edge) => resolve(ROOT, edge.path) === resolve(ROOT, module),
                    ),
                )
                .map(([path]) => path);
        const found = Object.keys(inputs).filter((path) =>
            PROFILER.some((pattern) => pattern.test(resolve(ROOT, path).slice(ROOT.length + 1))),
        );
        if (found.length !== 0)
            throw new Error(
                `gated bundle imports profiler modules:\n${found.map((path) => `  ${path} <- ${importers(path).join(", ")}`).join("\n")}`,
            );
    },
);

// The page sampler needs a display seat, so the declared-site rule is read here instead: it is the same
// production function the oracle calls, driven over a constructed sample. A window is 120 measured frames,
// the sanctioned site allocates its declared 2 per frame, and the red-circled site allocates at a rate
// nothing declares.
const SANCTION = { site: "src/a.ts:10", count: 2 };
const RED_CIRCLE = { site: "src/b.ts:20" };
const sanctionSite = (count: number) => ({ site: `openFrame ${SANCTION.site}`, bytes: 96, count });
const redCircleSite = (count: number) => ({
    site: `upload ${RED_CIRCLE.site}`,
    bytes: 4096,
    count,
});
const windows = (...sites: { site: string; bytes: number; count: number }[][]) =>
    sites.map((rows, index) => ({ label: `window ${index}`, sites: rows, frames: 120 }));
const STEADY = windows(
    [sanctionSite(240), redCircleSite(517)],
    [sanctionSite(240), redCircleSite(499)],
);

check(
    "a declared site off its count by one allocation in one frame reds",
    {
        claim: "the declared-site rule holds a steady sample green, reds a sanctioned site whose window reads one allocation above its derived count, and reds a site no declaration names",
    },
    () => {
        // Non-vacuity: the rule reads green on the sample every mutation below starts from.
        expect(declaredSiteFailures(STEADY, [SANCTION], [RED_CIRCLE])).toEqual([]);
        // One extra allocation, in one frame, of one window: 241 against the derived 2×120. This is the
        // defect rounding the count per frame used to hide, so it is the witness that the equality is exact.
        const offByOne = windows(
            [sanctionSite(240), redCircleSite(517)],
            [sanctionSite(241), redCircleSite(499)],
        );
        expect(declaredSiteFailures(offByOne, [SANCTION], [RED_CIRCLE])).toEqual([
            "sanctioned sites off their derived counts:\n  window 1: src/a.ts:10 read 241 allocations over 120 frames against the declared 2×/f, which is 240",
        ]);
        // One below reds the same way, so the condition is equality and not a ceiling.
        const offByOneUnder = windows(
            [sanctionSite(239), redCircleSite(517)],
            [sanctionSite(240), redCircleSite(499)],
        );
        expect(declaredSiteFailures(offByOneUnder, [SANCTION], [RED_CIRCLE])).toHaveLength(1);
        // A site neither declaration names reds as undeclared, whatever the declared sites do.
        const extra = windows([
            sanctionSite(240),
            redCircleSite(517),
            { site: "leak src/c.ts:30", bytes: 64, count: 7 },
        ]);
        expect(declaredSiteFailures(extra, [SANCTION], [RED_CIRCLE])[0]).toContain(
            "64 B at leak src/c.ts:30",
        );
    },
);

check(
    "a red circle asserts membership only, and reds when it is stale",
    {
        claim: "the declared-site rule accepts a red-circled site at any count, reds it when a window shows it allocating nothing, and reds a stale sanction the same way",
    },
    () => {
        // A red circle carries no count, so any rate holds: it is deferred debt meant to trend to zero,
        // not a budget. Only a declaration that names a count is asserted against one.
        const wild = windows(
            [sanctionSite(240), redCircleSite(3)],
            [sanctionSite(240), redCircleSite(40_000)],
        );
        expect(declaredSiteFailures(wild, [SANCTION], [RED_CIRCLE])).toEqual([]);
        // A red-circled site absent from a window is stale, exactly as a sanction is: a row that names a
        // site allocating nothing is a row nobody retired.
        const goneRedCircle = windows([sanctionSite(240), redCircleSite(517)], [sanctionSite(240)]);
        expect(declaredSiteFailures(goneRedCircle, [SANCTION], [RED_CIRCLE])).toEqual([
            "stale declared rows:\n  window 1: red-circle src/b.ts:20 allocates nothing",
        ]);
        const goneSanction = windows([sanctionSite(240), redCircleSite(517)], [redCircleSite(499)]);
        expect(declaredSiteFailures(goneSanction, [SANCTION], [RED_CIRCLE])).toEqual([
            "stale declared rows:\n  window 1: sanction src/a.ts:10 allocates nothing",
        ]);
    },
);

// The sentinel's own identity, read here because everything the page sampler reports divides by the frame
// count it produces. A profile node carries the definition site the scan keys on: script, line and column.
const frame = (scriptId: string, functionName = "__shallotFrameMark", url = "") => ({
    functionName,
    url,
    scriptId,
    lineNumber: 3,
    columnNumber: 30,
});
const node = (id: number, callFrame: ReturnType<typeof frame>, children: never[] = []) => ({
    id,
    callFrame,
    children,
});
const samples = (...perNode: [number, number][]) =>
    perNode.flatMap(([nodeId, count]) =>
        Array.from({ length: count }, () => ({ nodeId, size: 24 })),
    );
const profileOf = (children: ReturnType<typeof node>[], rows: [number, number][]) => ({
    head: { id: 1, callFrame: frame("root", "(root)", ""), children },
    samples: samples(...rows),
});

check(
    "the frame sentinel refuses a span where its identity is not exactly one frame",
    {
        claim: "the sentinel scan returns the sole matching frame's allocation count, refuses by name when no frame answers to it, and refuses by a different name naming every definition site when more than one does",
    },
    () => {
        // Non-vacuity: one matching frame reads its own sample count, which is the span's frame count.
        expect(sentinelFrames(profileOf([node(2, frame("4"))], [[2, 120]]))).toBe(120);
        // Two definition sites answering to the sentinel's name: `attribute` keys by site name, so these
        // would collapse into one row and their sum, 360, would be read as the frame count. Every per-frame
        // figure would then be a third of the truth and the exact-count assertion would red at the
        // sanctioned sites instead of here.
        const impostor = profileOf(
            [node(2, frame("4")), node(3, { ...frame("5"), lineNumber: 1 })],
            [
                [2, 120],
                [3, 240],
            ],
        );
        expect(() => sentinelFrames(impostor)).toThrow("is not unique in this span");
        // The message names what matched, so the reader is told what is being counted as frames.
        expect(() => sentinelFrames(impostor)).toThrow("script 4 at 4:31: 120 allocations");
        expect(() => sentinelFrames(impostor)).toThrow("script 5 at 2:31: 240 allocations");
        // Nothing answering to the sentinel means the page was not stepping frames under the profiler.
        // It is a different failure from the one above and says so.
        const silent = profileOf([node(2, frame("4", "somethingElse"))], [[2, 99]]);
        expect(() => sentinelFrames(silent)).toThrow("did not sample this span");
        // A frame carrying the reserved name but a served script's url is not the sentinel either.
        const served = profileOf(
            [node(2, frame("4", "__shallotFrameMark", "http://localhost:1/app.js"))],
            [[2, 99]],
        );
        expect(() => sentinelFrames(served)).toThrow("did not sample this span");
        // Two nodes at one definition site are one identity, summed: the rule is on the definition site,
        // not the node count, so a frame reached by two call paths is still the sentinel.
        const twoPaths = profileOf(
            [node(2, frame("4")), node(3, frame("4"))],
            [
                [2, 70],
                [3, 50],
            ],
        );
        expect(sentinelFrames(twoPaths)).toBe(120);
    },
);
