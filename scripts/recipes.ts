import { resolve } from "node:path";
import { Glob } from "bun";
import { skipReason, verify } from "./verify";

// `bun run recipes` — the recipes' dynamics smoke. Each listed recipe installs a `window.__harness` (its
// `src/smoke.ts`, wired only in its manifest) whose `run()` drives the scene and asserts the concept's
// observable — the platform slides, joints hold or break, friction rates differ, the car advances under
// throttle, the profiler reports GPU time. This drives each through `shallot verify` (the same shipped gate
// `bun bench` / `bun run flows` wrap) and reads the pass/fail verdict. It is the standing regression gate for
// the recipes' behaviour, not just that they render.
//
// Display-gated exactly like flows: verify needs a real display + a conformant WebGPU adapter, so on WSL /
// headless it skips honestly (native hardware only). The green run is native; here it proves the wiring.

interface Recipe {
    dir: string;
    // the harness check names this recipe's smoke reports — the run must surface all of them and pass each,
    // never degrade to a bare boot smoke (a harness that readies without a run() reports ok:true otherwise).
    checks: string[];
    timeoutMs?: number;
}

// The recipe dirs are derived from the glob `examples/recipes/*/src/smoke.ts` so a new smoke is gated by
// construction — no hand list to drift. The per-recipe check names are a lookup; a recipe not in the map
// runs with empty checks (still gated on verify pass + verdict.ok, just without named-check assertions).
const CHECKS: Record<string, string[]> = {
    "moving-platform": ["platform slides"],
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

const recipeGlob = new Glob("*/src/smoke.ts");
const recipeDirs: string[] = [];
for await (const path of recipeGlob.scan({
    cwd: resolve(import.meta.dir, "../examples/recipes"),
})) {
    recipeDirs.push(path.split("/")[0]);
}
recipeDirs.sort();
const RECIPES: Recipe[] = recipeDirs.map((dir) => ({ dir, checks: CHECKS[dir] ?? [] }));

async function runRecipe(r: Recipe): Promise<boolean> {
    console.log(`\n--- ${r.dir} ---`);
    const result = await verify(`examples/recipes/${r.dir}`, [
        "--timeout",
        String(r.timeoutMs ?? 60_000),
    ]);
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
        return `no recipes derived from examples/recipes/*/src/smoke.ts — the glob matched nothing`;
    }
    return null;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run recipes [--recipe <name>]

Runs the physics recipes' dynamics smoke through \`shallot verify\`. Display-gated (native hardware only).

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

    console.log("Running recipe dynamics smoke...");
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
