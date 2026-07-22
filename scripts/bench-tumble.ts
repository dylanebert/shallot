import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { REPO_ROOT, skipReason, teardownBridge, verify } from "./verify";

// `bun run bench:tumble` — the batched real-device gate for the tumble gym sample twins (spec tumble-inline
// stage 4c). One browser boot serves every scenario: the WSL bridge (or a local chromium) starts once and is
// reused across a `shallot verify examples/gym --query scenario=<slug>` page per twin, so the whole corpus
// runs in one process on one bridge boot. Every twin's gold assert must pass and its wall time stay under
// the standing budget; the run prints a per-scenario timing table and exits nonzero on any red row.
//
//   ⚠ ONE bridge session at a time. On WSL the bridge is a SINGLE shared host browser (scripts/wsl-bridge.ts);
//   two concurrent runs attach to the same browser and close each other's pages
//   (`browserContext.newPage: … closed`). Never launch this alongside another `bun bench`, `bun run
//   bench:tumble`, `bun run flows`, or `bun run recipes` — they all drive that one browser.

const GYM = "examples/gym";
const INDEX = resolve(REPO_ROOT, "packages/shallot/tests/tumble/samples/index.json");

// Standing per-scenario wall budget: a twin that takes longer than this goes RED. Sized well above the
// heaviest twin's real runtime and well below a hung-scenario watchdog (~300s) — it catches a regression or
// a stall, not normal run-to-run variance.
const BUDGET_S = 90;

// The gym twins = every minted sample gold except the Benchmark category (Large Pyramid + Joint Grid dropped
// in stage 2 — throughput is pile/stress's job; Rain never minted). Derived from the committed gold index,
// never hand-listed: the twins ARE the golds, so a new twin is a new gold and shows up here for free. Each
// scenario registers under its gold slug (`tumble-sample.ts`: `name ?? gold.slug`), so the slug is the
// `--query scenario=` value.
function twinSlugs(): string[] {
    const index = JSON.parse(readFileSync(INDEX, "utf8")) as {
        samples: { slug: string; category: string }[];
    };
    return index.samples.filter((s) => s.category !== "Benchmark").map((s) => s.slug);
}

interface Row {
    slug: string;
    wall: number;
    gold: boolean;
    withinBudget: boolean;
    error?: string;
}

async function runTwin(slug: string): Promise<Row> {
    const t0 = Date.now();
    const result = await verify(
        GYM,
        [
            "--query",
            `scenario=${slug}`,
            "--query",
            "seed=1",
            "--query",
            "warmup=10",
            "--query",
            "frames=20",
            "--memory",
        ],
        true, // quiet — the table is the signal; a red row prints its own detail
    );
    const wall = (Date.now() - t0) / 1000;

    const goldCheck = result?.verdict?.checks?.find((c) => c.name === "gold");
    const gold = result?.pass === true && result.verdict?.ok === true && goldCheck?.ok === true;
    const error = gold
        ? undefined
        : (goldCheck?.detail ??
          result?.error ??
          result?.errors?.[0]?.split("\n")[0] ??
          "no gold verdict");
    return { slug, wall, gold, withinBudget: wall <= BUDGET_S, error };
}

function printTable(rows: Row[]): void {
    const bar = "=".repeat(64);
    console.log(`\n${bar}`);
    console.log(`  tumble gym twins — real-device batch (budget ${BUDGET_S}s/scenario)`);
    console.log(bar);
    console.log(`  ${"scenario".padEnd(34)}${"wall(s)".padStart(9)}   gold   status`);
    console.log(`  ${"-".repeat(34)}${" ".repeat(3)}${"-".repeat(6)}   ----   ------`);
    for (const r of rows) {
        const ok = r.gold && r.withinBudget;
        const status = !r.gold ? "FAIL" : !r.withinBudget ? "SLOW" : "OK";
        console.log(
            `  ${r.slug.padEnd(34)}${r.wall.toFixed(1).padStart(9)}   ${r.gold ? "✓" : "✗"}      ${ok ? "OK" : status}`,
        );
        if (!ok && r.error) console.log(`  ${" ".repeat(34)}   ↳ ${r.error}`);
    }
    console.log(bar);
    const slowest = rows.reduce((m, r) => Math.max(m, r.wall), 0);
    const total = rows.reduce((s, r) => s + r.wall, 0);
    console.log(
        `  ${rows.length} scenarios · total ${total.toFixed(1)}s · slowest ${slowest.toFixed(1)}s\n`,
    );
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run bench:tumble

Runs every tumble gym sample twin through \`shallot verify\` on a real device, reusing one browser session,
and gates each on its committed gold plus a ${BUDGET_S}s/scenario wall budget. Display-gated (native
hardware / the WSL host bridge). ONE bridge session at a time — never run concurrent with another bench /
flows / recipes.

Options:
  --only <slug>   Run a single twin by its gold slug (e.g. stacking-arch)`);
        process.exit(0);
    }

    const skip = skipReason();
    if (skip) {
        console.log(`bun run bench:tumble needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const onlyIdx = args.indexOf("--only");
    const only = onlyIdx !== -1 ? args[onlyIdx + 1] : undefined;
    let slugs = twinSlugs();
    if (only) {
        slugs = slugs.filter((s) => s === only);
        if (slugs.length === 0) {
            console.error(`no tumble twin "${only}" — run without --only to see the full list`);
            process.exit(2);
        }
    }
    console.log(`Running ${slugs.length} tumble gym twins over one browser session...`);
    const rows: Row[] = [];
    try {
        for (const slug of slugs) {
            const row = await runTwin(slug);
            rows.push(row);
            const mark = row.gold && row.withinBudget ? "✓" : "✗";
            console.log(`  ${mark} ${slug.padEnd(34)} ${row.wall.toFixed(1)}s`);
        }
    } finally {
        await teardownBridge();
    }

    printTable(rows);
    const red = rows.filter((r) => !r.gold || !r.withinBudget);
    if (red.length > 0) {
        console.error(`FAIL: ${red.length}/${rows.length} scenarios red`);
        process.exit(1);
    }
    console.log(`PASS: ${rows.length}/${rows.length} tumble gym twins green`);
    process.exit(0);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
