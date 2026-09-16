import { expect } from "bun:test";
import { resolve } from "node:path";
import { declaredSiteFailures } from "@dylanebert/shallot/harness/allocation";
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
