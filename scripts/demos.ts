import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, symlinkSync } from "node:fs";
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
// The gate's unit is a built HTML entry point, not a demo directory. Per demo, the built `*.html` files
// under its output dir are enumerated structurally, and each one that presents a `<canvas>` directly is
// verified through `shallot verify --dist`. A page that hosts a canvas only inside an `<iframe>` (the
// visualization gallery index) is a link surface, not a verifiable unit — the iframe's target page is
// verified directly, so counting the container too would double-count. A demo contributing zero verified
// entry points is a red, not a skip. The per-demo entry-point count is printed so a gate that silently
// stops finding pages is visible in its own output.
//
// Display-gated exactly like flows and recipes: verify needs a real display + a conformant WebGPU
// adapter, so on WSL / headless it skips honestly (native hardware only). Unlike those routine
// regression gates, a skip here exits nonzero — this is a release gate, and a skipped release gate
// is not green. The green run is native hardware with every demo verified.
//
// The roster is the single source of truth — imported from `site/roster.ts`, never duplicated. It is
// derived from `examples/showcase/` by enumeration, so a second copy is impossible by construction
// rather than caught by a gate (the set-equality clause it used to need is gone). Ejection and
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
    entryPoints: number;
    detail?: string;
}

function isSkip(result: VerifyResult | null): boolean {
    return result?.pass === false && !!result.error?.startsWith(DISPLAY_GATE_PREFIX);
}

// A page presents a canvas directly when its own markup contains a `<canvas>` tag. An iframe-hosted
// canvas is verified by verifying the iframe's target page directly; a page whose only canvases live
// behind `<iframe src=...>` (the visualization gallery index) has no `<canvas>` in its own markup and
// is not an entry point. The check is structural — read the built HTML, look for the tag — so it does
// not hardcode any demo's page names and adapts when a multi-page demo adds or renames a page.
function hasDirectCanvas(htmlPath: string): boolean {
    const content = readFileSync(htmlPath, "utf-8");
    return /<canvas[\s>]/i.test(content);
}

// Enumerate every built `*.html` file under a demo's output dir (recursive) and return those that
// present a canvas directly — the verifiable entry points. Structural enumeration from the build
// output, not a hardcoded list: a named list is right today and blind to the next multi-page demo.
function enumerateEntryPoints(demoOut: string): string[] {
    const htmlFiles: string[] = [];
    function walk(dir: string): void {
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            const full = join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(full);
            } else if (entry.name.endsWith(".html")) {
                htmlFiles.push(full);
            }
        }
    }
    walk(demoOut);
    return htmlFiles.filter(hasDirectCanvas);
}

// Create a scratch dir whose `dist/` mirrors the demo's build output but with `dist/index.html`
// replaced by a symlink to the target entry-point HTML file. `shallot verify --dist` serves
// `<dir>/dist/index.html`, so this makes verify serve the specific entry point while keeping the
// sibling assets the page references (via `../assets/...`) reachable through symlinks.
function makeScratch(demoOut: string, entryHtml: string): string {
    const scratch = join(
        tmpdir(),
        `shallot-demos-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    );
    const dist = join(scratch, "dist");
    mkdirSync(dist, { recursive: true });

    // Symlink every top-level entry from the demo output into dist/, except index.html (replaced below).
    for (const entry of readdirSync(demoOut, { withFileTypes: true })) {
        if (entry.name === "index.html") continue;
        symlinkSync(resolve(demoOut, entry.name), join(dist, entry.name));
    }
    // Replace index.html with a symlink to the target entry point.
    symlinkSync(entryHtml, join(dist, "index.html"));

    return scratch;
}

async function runDemo(slug: string): Promise<DemoOutcome> {
    console.log(`\n--- ${slug} ---`);

    const demoOut = resolve(outDir, slug);
    if (!existsSync(demoOut)) {
        console.log(`FAIL: ${slug} — no build at out/site/${slug}/ (run \`bun run site\` first)`);
        return { slug, result: "fail", entryPoints: 0, detail: "no build output" };
    }

    // Enumerate built HTML entry points that present a canvas directly. The gallery index
    // (iframe-only, no direct canvas) is a link surface, not a verifiable unit — an iframe-hosted
    // canvas is verified by verifying the iframe's target page directly.
    const entryPoints = enumerateEntryPoints(demoOut);
    console.log(`  entry points: ${entryPoints.length}`);

    // A demo contributing zero verified entry points is a red, not a skip — the cheapest "fix" for a
    // false red is a false green, so this arm makes a gate that silently stops finding pages visible.
    if (entryPoints.length === 0) {
        console.log(
            `FAIL: ${slug} — zero verifiable entry points (no built HTML page presents a canvas directly)`,
        );
        return { slug, result: "fail", entryPoints: 0, detail: "zero entry points" };
    }

    let allPass = true;
    let skipDetail: string | undefined;
    for (const entryHtml of entryPoints) {
        const label = entryHtml.slice(demoOut.length + 1);
        const scratch = makeScratch(demoOut, entryHtml);
        try {
            const result = await verify(scratch, ["--dist", "--timeout", "60000"]);
            if (result === null) {
                console.log(`  FAIL: ${label} — verify crashed before reporting`);
                allPass = false;
                continue;
            }
            if (isSkip(result)) {
                console.log(`  SKIP: ${label} — display gate refused software adapter`);
                skipDetail = result.error;
                continue;
            }
            const ok = result.pass === true;
            console.log(ok ? `  PASS: ${label}` : `  FAIL: ${label}`);
            if (!ok) {
                allPass = false;
                if (result.errors?.length) {
                    for (const e of result.errors.slice(0, 3))
                        console.log(`    ${e.split("\n")[0]}`);
                }
            }
        } finally {
            rmSync(scratch, { recursive: true, force: true });
        }
    }

    // A skip on any entry point means the display gate refused — the whole demo skips, since the
    // hardware can't verify any page. A fail on any (with no skips) means the demo reds.
    if (skipDetail) {
        return { slug, result: "skip", entryPoints: entryPoints.length, detail: skipDetail };
    }
    return { slug, result: allPass ? "pass" : "fail", entryPoints: entryPoints.length };
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run demos [--demo <slug>]

Builds every showcase demo (via \`bun run site\`) and runs \`shallot verify --dist\` over each
built HTML entry point that presents a canvas directly — display-gated, on real hardware. A
release gate: a skip (exit 4) is not green.

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
    for (const o of outcomes) {
        console.log(
            `  ${o.slug}: ${o.result} (${o.entryPoints} entry point${o.entryPoints === 1 ? "" : "s"})`,
        );
    }

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
