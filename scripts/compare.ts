import { resolve, join } from "path";
import { Database } from "bun:sqlite";

const projectDir = resolve(import.meta.dir, "..");
const dbPath = join(projectDir, "packages/shallot/tests/gpu/results.db");

const db = new Database(dbPath, { readonly: true });

const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/compare.ts [options]

Options:
  --pipeline <name>    Filter by pipeline (default: all)
  --scenario <name>    Filter by scenario (default: all)
  --last <n>           Show last N runs (default: 10)
  --commits            Compare across commits
  --scale              Show scaling data (group by object_count)

Examples:
  bun run scripts/compare.ts --pipeline raster --scale
  bun run scripts/compare.ts --commits --last 5
  bun run scripts/compare.ts --pipeline raster --commits`);
    process.exit(0);
}

let pipeline: string | undefined;
let scenario: string | undefined;
let last = 10;
let showCommits = false;
let showScale = false;

for (let i = 0; i < args.length; i++) {
    if (args[i] === "--pipeline" && i + 1 < args.length) pipeline = args[++i];
    else if (args[i] === "--scenario" && i + 1 < args.length) scenario = args[++i];
    else if (args[i] === "--last" && i + 1 < args.length) last = parseInt(args[++i], 10);
    else if (args[i] === "--commits") showCommits = true;
    else if (args[i] === "--scale") showScale = true;
}

const conditions: string[] = [];
const params: (string | number)[] = [];

if (pipeline) {
    conditions.push("pipeline = ?");
    params.push(pipeline);
}
if (scenario) {
    conditions.push("scenario = ?");
    params.push(scenario);
}

const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

if (showScale) {
    const rows = db
        .prepare(
            `SELECT pipeline, object_count, gpu_avg, gpu_p95, cpu_total, passes_json, cpu_systems_json, commit_hash, timestamp, ramp
		 FROM runs ${where} AND object_count IS NOT NULL
		 ORDER BY timestamp DESC, object_count ASC
		 LIMIT ?`,
        )
        .all(...params, last * 3) as {
        pipeline: string;
        object_count: number;
        gpu_avg: number;
        gpu_p95: number;
        cpu_total: number;
        passes_json: string | null;
        cpu_systems_json: string | null;
        commit_hash: string;
        timestamp: string;
        ramp: number | null;
    }[];

    if (rows.length === 0) {
        console.log("No scaling data found.");
        process.exit(0);
    }

    const byCommit = new Map<string, typeof rows>();
    for (const r of rows) {
        const key = `${r.commit_hash} (${r.timestamp.slice(0, 10)})`;
        if (!byCommit.has(key)) byCommit.set(key, []);
        byCommit.get(key)!.push(r);
    }

    for (const [commit, cRows] of byCommit) {
        console.log(`\n${"=".repeat(60)}`);
        console.log(`  ${commit}`);
        console.log(`${"=".repeat(60)}`);

        const byPipeline = new Map<string, typeof rows>();
        for (const r of cRows) {
            const rampTag = r.ramp ? " (ramp)" : "";
            const key = r.pipeline + rampTag;
            if (!byPipeline.has(key)) byPipeline.set(key, []);
            byPipeline.get(key)!.push(r);
        }

        for (const [pl, pRows] of byPipeline) {
            const sorted = pRows.sort((a, b) => a.object_count - b.object_count);
            console.log(`\n  ${pl}`);
            console.log(
                `  ${"count".padEnd(10)} ${"GPU avg".padEnd(12)} ${"GPU p95".padEnd(12)} ${"CPU total".padEnd(12)}`,
            );
            for (const r of sorted) {
                console.log(
                    `  ${String(r.object_count).padEnd(10)} ${(r.gpu_avg?.toFixed(2) ?? "n/a").padEnd(12)} ${(r.gpu_p95?.toFixed(2) ?? "n/a").padEnd(12)} ${(r.cpu_total?.toFixed(2) ?? "n/a").padEnd(12)}`,
                );
            }
        }
    }
} else if (showCommits) {
    const rows = db
        .prepare(
            `SELECT commit_hash, pipeline, scenario, effects, object_count,
			        gpu_avg, gpu_p95, cpu_total, timestamp
		 FROM runs ${where}
		 ORDER BY timestamp DESC
		 LIMIT ?`,
        )
        .all(...params, last) as {
        commit_hash: string;
        pipeline: string;
        scenario: string;
        effects: string;
        object_count: number | null;
        gpu_avg: number;
        gpu_p95: number;
        cpu_total: number;
        timestamp: string;
    }[];

    if (rows.length === 0) {
        console.log("No results found.");
        process.exit(0);
    }

    const configKey = (r: {
        pipeline: string;
        scenario: string;
        effects: string;
        object_count: number | null;
    }) => `${r.pipeline}|${r.scenario}|${r.effects}|${r.object_count ?? ""}`;

    const prev = new Map<string, number>();
    const deltas = new Map<(typeof rows)[0], string>();
    for (const r of [...rows].reverse()) {
        const key = configKey(r);
        const prevAvg = prev.get(key);
        if (prevAvg !== undefined && prevAvg > 0 && r.gpu_avg != null) {
            const pct = ((r.gpu_avg - prevAvg) / prevAvg) * 100;
            deltas.set(r, `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`);
        }
        if (r.gpu_avg != null) prev.set(key, r.gpu_avg);
    }

    console.log(
        `${"commit".padEnd(10)} ${"pipeline".padEnd(12)} ${"scenario".padEnd(10)} ${"count".padEnd(8)} ${"GPU avg".padEnd(12)} ${"GPU p95".padEnd(12)} ${"CPU".padEnd(12)} ${"Δ%".padEnd(8)} ${"date".padEnd(12)}`,
    );
    for (const r of rows) {
        const delta = deltas.get(r) ?? "";
        console.log(
            `${r.commit_hash.padEnd(10)} ${r.pipeline.padEnd(12)} ${r.scenario.padEnd(10)} ${String(r.object_count ?? "").padEnd(8)} ${(r.gpu_avg?.toFixed(2) ?? "n/a").padEnd(12)} ${(r.gpu_p95?.toFixed(2) ?? "n/a").padEnd(12)} ${(r.cpu_total?.toFixed(2) ?? "n/a").padEnd(12)} ${delta.padEnd(8)} ${r.timestamp.slice(0, 10)}`,
        );
    }
} else {
    const rows = db
        .prepare(`SELECT * FROM runs ${where} ORDER BY timestamp DESC LIMIT ?`)
        .all(...params, last) as Record<string, unknown>[];

    if (rows.length === 0) {
        console.log("No results found.");
        process.exit(0);
    }

    for (const r of rows) {
        console.log(JSON.stringify(r, null, 2));
    }
}

db.close();
