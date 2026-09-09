import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { EXAMPLE_GATES } from "./example-gates";
import { skipReason, verify } from "./verify";

// `bun run recipes` — the one entry every recipe row's gate runs through. A dynamic recipe installs a
// `window.__harness` (its `src/smoke.ts`, wired only in its manifest) whose `run()` drives the scene and
// asserts the concept's observable — the platform slides, joints hold or break, friction rates differ, the
// car advances under throttle, the profiler reports GPU time. A static recipe (one whose registry row
// carries a `static` reason) has no runtime observable, so its verdict is verify's own boot + nonblank
// render. Both drive the shipped `shallot verify` through `./verify`, which is what makes the row
// attributable: that wrapper owns the display guard, while a bare `bunx shallot verify` row spawned
// through `sh -c` reaches only a software adapter and reds.
//
// Display-gated exactly like flows: verify needs a real display + a conformant WebGPU adapter, so on a
// headless box it skips honestly. The green run is native; here it proves the wiring.

interface Recipe {
    dir: string;
    // the harness check names this recipe's smoke reports — the run must surface all of them and pass each,
    // never degrade to a bare boot smoke (a harness that readies without a run() reports ok:true otherwise).
    // Empty for a static recipe, which reports no verdict at all.
    checks: string[];
    /** why this recipe has no runtime observable — set iff it has no `src/smoke.ts`. */
    static?: string;
    timeoutMs?: number;
}

// The dynamic recipe dirs are derived from `examples/recipes/*/src/smoke.ts` and the static ones from the
// registry's `static` reasons, so a new recipe is gated by construction — no hand list to drift. The
// per-recipe check names are a lookup; a dynamic recipe not in the map runs with empty checks (still gated
// on verify pass + verdict.ok, just without named-check assertions).
const CHECKS: Record<string, string[]> = {
    "annotate-the-world": ["world annotation advances"],
    "day-night-sky": ["day night sun advances"],
    "overlay-ui": ["overlay hud advances"],
    "physics-playground": ["playground spawns dynamic body"],
    "respond-to-input": ["respond to input responds"],
    "stylize-the-look": ["outline selection advances"],
    "moving-platform": ["platform slides"],
    "animate-with-clips": ["both clips move their targets"],
    joints: ["joints hold their load"],
    "breakable-joints": ["a joint breaks under load"],
    "surface-friction": ["friction rates differ"],
    "drive-a-vehicle": ["car advances under throttle"],
    "measure-performance": ["profiler reports gpu time"],
    "compute-and-readback": ["three charges reduce to 6.00"],
    "gpu-particles": [
        "particles rise off the spawn plane and fall back",
        "the compute buffer is what the vertex stage binds",
    ],
};

/** Build the roster from the recipe directories on disk and the registry's `static` reasons — the two
 *  sources that already exist, so a new recipe is gated by construction. Pure over both inputs so the
 *  static/dynamic split is unit-testable without a filesystem walk. */
export function rosterFrom(
    dirs: readonly string[],
    statics: ReadonlyMap<string, string>,
): Recipe[] {
    return [...dirs]
        .sort()
        .map((dir) => ({ dir, checks: CHECKS[dir] ?? [], static: statics.get(dir) }));
}

const recipesRoot = resolve(import.meta.dir, "../examples/recipes");
const dynamicDirs = existsSync(recipesRoot)
    ? readdirSync(recipesRoot, { withFileTypes: true })
          .filter(
              (e) => e.isDirectory() && existsSync(resolve(recipesRoot, e.name, "src/smoke.ts")),
          )
          .map((e) => e.name)
    : [];
// A row with either reason has no verdict of its own: `static` never moves, `bootOnly` moves but has
// no check asserting its subject yet. Both gate on verify's own boot + nonblank render.
const STATIC_REASONS = new Map(
    EXAMPLE_GATES.filter((row) => row.tier === "recipes" && (row.static ?? row.bootOnly)).map(
        (row): [string, string] => [
            row.dir.slice("examples/recipes/".length),
            (row.static ?? row.bootOnly) as string,
        ],
    ),
);
const RECIPES: Recipe[] = rosterFrom([...dynamicDirs, ...STATIC_REASONS.keys()], STATIC_REASONS);

async function runRecipe(r: Recipe): Promise<boolean> {
    console.log(`\n--- ${r.dir} ---`);
    const result = await verify(`examples/recipes/${r.dir}`, [
        "--timeout",
        String(r.timeoutMs ?? 60_000),
    ]);
    // a static recipe installs no harness, so verify reports no verdict: its gate is verify's own
    // boot + settled-nonblank-render pass. Requiring `verdict.ok` there would fail every static row.
    if (r.static) {
        const ok = result?.pass === true;
        console.log(ok ? `PASS: ${r.dir} (static — ${r.static})` : `FAIL: ${r.dir}`);
        return ok;
    }
    let ok = result?.pass === true && result.verdict?.ok === true;
    for (const name of r.checks) {
        if (!result?.verdict?.checks?.some((c) => c.name === name && c.ok)) {
            console.log(`  ✗ missing or failed check: ${name}`);
            ok = false;
        }
    }
    console.log(ok ? `PASS: ${r.dir}` : `FAIL: ${r.dir}`);
    return ok;
}

/** Pure seam for the population guards — exported so the S3 arm can exercise the empty-glob
 *  guard behaviorally (the real glob matches real recipes, so the empty case can't be triggered
 *  hermetically without removing recipe dirs). Returns an error message if the population is
 *  empty or the selector is unknown, or null if the run should proceed. */
export function populationError(recipeDirs: string[], only?: string): string | null {
    if (only && !recipeDirs.includes(only)) {
        return `no recipe "${only}" — one of: ${recipeDirs.join(", ")}`;
    }
    if (!only && recipeDirs.length === 0) {
        return `no recipes derived from examples/recipes — the scan matched nothing`;
    }
    return null;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run recipes [--recipe <name>]

Runs every recipe through \`shallot verify\` — dynamics smoke where one exists, boot + render for a
registered static recipe. Display-gated (native hardware only).

Options:
  --recipe <name>   Run a single recipe by its directory name (e.g. moving-platform)`);
        process.exit(0);
    }
    const idx = args.indexOf("--recipe");
    const only = idx !== -1 ? args[idx + 1] : undefined;

    const list = only ? RECIPES.filter((r) => r.dir === only) : RECIPES;
    const popErr = populationError(
        RECIPES.map((r) => r.dir),
        only,
    );
    if (popErr) {
        console.error(popErr);
        process.exit(only ? 2 : 1);
    }

    const skip = skipReason();
    if (skip) {
        console.log(`bun run recipes needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    console.log(`Running ${list.length} recipe verification(s)...`);
    let allPass = true;
    for (const r of list) allPass = (await runRecipe(r)) && allPass;

    if (!allPass) {
        console.error("\nFAIL: recipe smoke failed");
        process.exit(1);
    }
    console.log("\nPASS: recipe smoke green");
    process.exit(0);
}

if (import.meta.main) {
    main().catch((err) => {
        console.error(err instanceof Error ? err.message : err);
        process.exit(1);
    });
}
