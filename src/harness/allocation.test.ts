import { expect } from "bun:test";
import { resolve } from "node:path";
import { declaredSiteFailures, derivedFrames, where } from "@dylanebert/shallot/harness/allocation";
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
// production function the oracle calls, driven over constructed windows. A window is bracketed at 120 to
// 133 frames by the page's own counter, the sanctioned site allocates its declared 2 per frame over 120 of
// them, and the red-circled site allocates at a rate nothing declares.
const SANCTION = { site: "src/a.ts:10", count: 2 };
const RED_CIRCLE = { site: "src/b.ts:20" };
const sanctionSite = (count: number) => ({ site: `openFrame ${SANCTION.site}`, bytes: 96, count });
const redCircleSite = (count: number) => ({
    site: `upload ${RED_CIRCLE.site}`,
    bytes: 4096,
    count,
});
const windows = (...sites: { site: string; bytes: number; count: number }[][]) =>
    sites.map((rows, index) => ({
        label: `window ${index}`,
        sites: rows,
        frames: 120,
        framesAtMost: 133,
    }));
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
        // One extra allocation, in one frame, of one window: 241 against 2 per frame leaves a remainder,
        // so the window has no whole frame count and the site that broke it is named. This is the defect
        // rounding the count per frame used to hide.
        const offByOne = windows(
            [sanctionSite(240), redCircleSite(517)],
            [sanctionSite(241), redCircleSite(499)],
        );
        const broken = declaredSiteFailures(offByOne, [SANCTION], [RED_CIRCLE]);
        expect(broken).toHaveLength(1);
        expect(broken[0]).toContain("not a whole number of frames at the declared 2×/f");
        expect(broken[0]).toContain("1.81 to 2.01 per frame");
        expect(broken[0]).toContain("src/a.ts:10 read 241 allocations");
        // One below reds the same way, so the condition is exactness and not a ceiling.
        expect(
            declaredSiteFailures(
                windows([sanctionSite(239), redCircleSite(517)]),
                [SANCTION],
                [RED_CIRCLE],
            ),
        ).toHaveLength(1);
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
    "an accessor site matches the declaration that names its file and line",
    {
        claim: "a measured site whose function name is a V8 accessor, and so carries a space of its own, still resolves to the file and line a declaration names, rather than reading as undeclared and stale at once",
    },
    () => {
        // V8 names a setter frame `set transform`, which is two words before the path. Splitting the site
        // at its first space left `transform src/b.ts:20` as the "file", so the red-circle row for that
        // site matched nothing: the verdict called it undeclared in the same breath as it called the row
        // stale. The physics kernel's column setters are exactly this shape.
        expect(where("set transform src/b.ts:20")).toBe("src/b.ts:20");
        expect(where("get size src/b.ts:20")).toBe("src/b.ts:20");
        expect(where("flush src/b.ts:20")).toBe("src/b.ts:20");
        const accessor = windows([
            sanctionSite(240),
            { ...redCircleSite(517), site: "set transform src/b.ts:20" },
        ]);
        expect(declaredSiteFailures(accessor, [SANCTION], [RED_CIRCLE])).toEqual([]);
    },
);

check(
    "sanctioned sites must agree on one frame count inside the page's bracket",
    {
        claim: "the derived frame count is the one every sanctioned site agrees on and lies inside the window's bracket; disagreeing rows, a count outside the bracket, and a window where nothing sanctioned allocated each red by their own name",
    },
    () => {
        const other = { site: "src/d.ts:40", count: 1 };
        // Two rows, both whole, agreeing on 120 frames: the window has a frame count.
        const agreeing = windows([
            sanctionSite(240),
            { site: "submit src/d.ts:40", bytes: 8, count: 120 },
        ]);
        expect(derivedFrames(agreeing[0], [SANCTION, other])).toEqual({ frames: 120 });
        // The same window with one row one allocation high: both quotients are whole, so nothing is
        // ragged, but they disagree and the rule names every row's own reading rather than picking one.
        const disagreeing = windows([
            sanctionSite(240),
            { site: "submit src/d.ts:40", bytes: 8, count: 121 },
        ]);
        const split = derivedFrames(disagreeing[0], [SANCTION, other]);
        expect(split).toHaveProperty("reason");
        expect((split as { reason: string }).reason).toContain("disagree on how many frames");
        expect((split as { reason: string }).reason).toContain(
            "240 allocations at 2×/f is 120 frames",
        );
        expect((split as { reason: string }).reason).toContain(
            "121 allocations at 1×/f is 121 frames",
        );
        // Rows can agree on a number that is not the frame count. The page's own counter bracketed this
        // window at 120 to 133, so an agreed 60 is rejected: it would halve every per-frame figure.
        const halved = windows([
            sanctionSite(120),
            { site: "submit src/d.ts:40", bytes: 8, count: 60 },
        ]);
        const outside = derivedFrames(halved[0], [SANCTION, other]);
        expect((outside as { reason: string }).reason).toContain("outside the 120 to 133 frames");
        // A window where no sanctioned site allocated has no frame count and is not silently zero.
        const silent = windows([redCircleSite(517)]);
        expect((derivedFrames(silent[0], [SANCTION]) as { reason: string }).reason).toContain(
            "no sanctioned site allocated",
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
        const failures = declaredSiteFailures(goneSanction, [SANCTION], [RED_CIRCLE]);
        expect(failures[0]).toBe(
            "stale declared rows:\n  window 1: sanction src/a.ts:10 allocates nothing",
        );
        // and that window then has no frame count of its own, which is its own message
        expect(failures[1]).toContain("no sanctioned site allocated");
    },
);
