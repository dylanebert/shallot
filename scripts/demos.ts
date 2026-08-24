import { existsSync, mkdirSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { ROSTER } from "../site/roster";
import { skipReason, type VerifyResult, verify } from "./verify";

// `bun run demos` — the site demo release gate. Builds every showcase demo as an ejected consumer of
// the published package (via `bun run site`, which drives `scripts/build-site.ts`), then runs
// `shallot verify --dist` over every built demo — display-gated, on real hardware. A demo that fails
// to build or render reds the gate; a demo that skips (the display gate refuses a software adapter,
// exit 4) reports as skipped and must not be reportable as green — a release gate that skipped is not
// a release gate that passed.
//
// Display-gated exactly like flows and recipes: verify needs a real display + a conformant WebGPU
// adapter, so on WSL / headless it skips honestly (native hardware only). Unlike those routine
// regression gates, a skip here exits nonzero — this is a release gate, and a skipped release gate
// is not green. The green run is native hardware with every demo verified.
//
// The roster is the single source of truth — imported from `site/roster.ts`, never duplicated. A
// second copy is the exact defect `check-site.ts`'s set-equality gate exists to catch. Ejection and
// building are not re-implemented: `bun run site` already ejects, installs, and builds every roster
// demo into `out/site/<slug>/`. This script drives that and then verifies the built dirs.

const root = resolve(import.meta.dir, "..");
const outDir = resolve(root, "out/site");

// The display gate's refusal message prefix — the CLI's `displayGateMessage` always opens with this,
// so a `VerifyResult` whose `error` starts here is an exit-4 skip rather than an exit-1 fail.
const DISPLAY_GATE_PREFIX = "shallot verify needs a real GPU adapter";

interface DemoOutcome {
    slug: string;
    result: "pass" | "fail" | "skip";
    detail?: string;
}

function isSkip(result: VerifyResult | null): boolean {
    return result?.pass === false && !!result.error?.startsWith(DISPLAY_GATE_PREFIX);
}

async function runDemo(slug: string): Promise<DemoOutcome> {
    console.log(`\n--- ${slug} ---`);

    const demoOut = resolve(outDir, slug);
    if (!existsSync(demoOut)) {
        console.log(`FAIL: ${slug} — no build at out/site/${slug}/ (run \`bun run site\` first)`);
        return { slug, result: "fail", detail: "no build output" };
    }

    // `shallot verify --dist` expects `<dir>/dist/index.html`; build-site.ts puts the built demo at
    // `out/site/<slug>/` (the dist contents, not a dist subdirectory). Create a scratch dir with a
    // `dist/` symlink so verify's --dist path finds the build without re-implementing ejection.
    const scratch = join(tmpdir(), `shallot-demos-${slug}-${Date.now()}`);
    mkdirSync(scratch, { recursive: true });
    symlinkSync(demoOut, join(scratch, "dist"));

    try {
        const result = await verify(scratch, ["--dist", "--timeout", "60000"]);
        if (result === null) {
            console.log(`FAIL: ${slug} — verify crashed before reporting`);
            return { slug, result: "fail", detail: "no verdict" };
        }
        if (isSkip(result)) {
            console.log(`SKIP: ${slug} — display gate refused software adapter`);
            return { slug, result: "skip", detail: result.error };
        }
        const ok = result.pass === true;
        console.log(ok ? `PASS: ${slug}` : `FAIL: ${slug}`);
        if (!ok && result.errors?.length) {
            for (const e of result.errors.slice(0, 3)) console.log(`  ${e.split("\n")[0]}`);
        }
        return { slug, result: ok ? "pass" : "fail" };
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run demos [--demo <slug>]

Builds every showcase demo (via \`bun run site\`) and runs \`shallot verify --dist\` over each —
display-gated, on real hardware. A release gate: a skip (exit 4) is not green.

Options:
  --demo <slug>   Build and verify a single demo by its roster slug`);
        process.exit(0);
    }

    const idx = args.indexOf("--demo");
    const only = idx !== -1 ? args[idx + 1] : undefined;
    if (only && !ROSTER.some((d) => d.slug === only)) {
        console.error(`no demo "${only}" — one of: ${ROSTER.map((d) => d.slug).join(", ")}`);
        process.exit(2);
    }

    const demos = only ? ROSTER.filter((d) => d.slug === only) : ROSTER;

    // --- build ---
    console.log("Building site demos...");
    const buildArgs = only ? ["run", "site", "--demo", only] : ["run", "site"];
    const build = Bun.spawnSync(["bun", ...buildArgs], {
        cwd: root,
        stdout: "inherit",
        stderr: "inherit",
    });
    if (build.exitCode !== 0) {
        console.error("\nFAIL: site build failed — demos gate cannot proceed");
        process.exit(1);
    }

    // --- display gate (pre-check) ---
    const skip = skipReason();
    if (skip) {
        console.log(`\nbun run demos needs native hardware (${skip}). Skipping.`);
        console.log("\nSKIPPED: demos gate skipped — not green (display gate)");
        process.exit(1);
    }

    // --- verify ---
    console.log("\nVerifying site demos (display-gated)...");
    const outcomes: DemoOutcome[] = [];
    for (const demo of demos) {
        outcomes.push(await runDemo(demo.slug));
    }

    const passed = outcomes.filter((o) => o.result === "pass").length;
    const failed = outcomes.filter((o) => o.result === "fail").length;
    const skipped = outcomes.filter((o) => o.result === "skip").length;
    const total = outcomes.length;

    console.log(`\n=== summary ===`);
    console.log(`  ${passed}/${total} verified, ${failed} failed, ${skipped} skipped`);

    if (failed > 0) {
        console.error("\nFAIL: demos gate failed");
        process.exit(1);
    }
    if (skipped > 0) {
        console.error(`\nSKIPPED: demos gate skipped (${skipped}/${total} demos) — not green`);
        process.exit(1);
    }
    console.log(`\nPASS: demos green (${passed}/${total} verified)`);
    process.exit(0);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
