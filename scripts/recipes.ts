import { skipReason, verify } from "./verify";

// `bun run recipes` — the physics recipes' dynamics smoke. Each of the six ported physics recipes installs a
// `window.__harness` (its `src/smoke.ts`, wired only in its manifest) whose `run()` drives the scene and
// asserts the concept's observable — the platform slides, the rotor spins, joints hold or break, friction
// rates differ, the car advances under throttle. This drives each through `shallot verify` (the same shipped
// gate `bun bench` / `bun run flows` wrap) and reads the pass/fail verdict. It is the standing regression
// gate for the recipes' behaviour, not just that they render.
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

const RECIPES: Recipe[] = [
    { dir: "moving-platform", checks: ["platform slides"] },
    { dir: "joints", checks: ["joints hold their load"] },
    { dir: "breakable-joints", checks: ["a joint breaks under load"] },
    { dir: "surface-friction", checks: ["friction rates differ"] },
    { dir: "drive-a-vehicle", checks: ["car advances under throttle"] },
];

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

    const skip = skipReason();
    if (skip) {
        console.log(`bun run recipes needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const list = only ? RECIPES.filter((r) => r.dir === only) : RECIPES;
    if (only && list.length === 0) {
        console.error(`no recipe "${only}" — one of: ${RECIPES.map((r) => r.dir).join(", ")}`);
        process.exit(2);
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

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
