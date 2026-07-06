import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
    type CorpusEntry,
    corpusPresent,
    entryOf,
    type Matrix,
    type MatrixEntry,
    type Status,
    walkCorpus,
} from "../packages/shallot/tests/gltf-corpus";

// gltf-conformance — the loud, human-facing surface of the glTF conformance suite (roadmap "glTF import —
// conformance + regression suite"). Walks the Khronos corpus through the deviceless importer (no Playwright,
// no GPU — `parse` is CPU-only) and prints three things:
//
//   • the STATUS table — supported / partial / unsupported totals, the headline of how much of the corpus the
//     importer carries today.
//   • the FEATURE breakdown — every intentionally-unimplemented feature key and how many models it gates. This
//     is the loud enumeration the request asks for: a skip is visible here, not silently green in a test.
//   • the DRIFT report — every (model, variant) whose outcome differs from the committed `gltf-matrix.json`,
//     so a reviewer reads a diff and decides "regression" vs "newly handled". A parse error is always drift.
//
// Bare run = a dry-run staleness gate (nonzero exit on drift or error). `--write` regenerates the matrix after
// the breakdown has been reviewed against the corpus's `model-index.json` tags + the glTF 2.0 spec — the
// matrix is a reviewed pin, not an unexamined snapshot.
//
// Run: bun run scripts/gltf-conformance.ts [--write]

const MATRIX_PATH = join(import.meta.dir, "../packages/shallot/tests/gltf-matrix.json");

// the matrix, deterministically ordered (models, variants, feature keys all sorted) so a regen is a clean diff
function buildMatrix(entries: CorpusEntry[]): Matrix {
    const matrix: Matrix = {};
    for (const e of entries) {
        if (!e.scene) continue; // an errored entry is drift, never pinned
        (matrix[e.model] ??= {})[e.variant] = entryOf(e.scene);
    }
    return sortMatrix(matrix);
}

function sortMatrix(matrix: Matrix): Matrix {
    const out: Matrix = {};
    for (const model of Object.keys(matrix).sort()) {
        out[model] = {};
        for (const variant of Object.keys(matrix[model]).sort())
            out[model][variant] = matrix[model][variant];
    }
    return out;
}

function loadMatrix(): Matrix | null {
    if (!existsSync(MATRIX_PATH)) return null;
    return JSON.parse(readFileSync(MATRIX_PATH, "utf8")) as Matrix;
}

function sameEntry(a: MatrixEntry, b: MatrixEntry): boolean {
    return (
        a.status === b.status &&
        a.meshes === b.meshes &&
        a.unsupported.length === b.unsupported.length &&
        a.unsupported.every((f, i) => f === b.unsupported[i])
    );
}

function statusTable(entries: CorpusEntry[]): void {
    const counts: Record<Status, number> = { supported: 0, partial: 0, unsupported: 0 };
    let errors = 0;
    for (const e of entries) {
        if (e.error) errors++;
        else if (e.scene) counts[entryOf(e.scene).status]++;
    }
    const total = entries.length;
    console.log("\n  STATUS");
    console.log(`    supported    ${String(counts.supported).padStart(4)}`);
    console.log(`    partial      ${String(counts.partial).padStart(4)}`);
    console.log(`    unsupported  ${String(counts.unsupported).padStart(4)}`);
    if (errors) console.log(`    ERROR        ${String(errors).padStart(4)}`);
    console.log(`    ${"".padEnd(13)}${"----".padStart(4)}`);
    console.log(`    total        ${String(total).padStart(4)} (model, variant) parsed`);
}

// the loud enumeration: each skip key, how many models hit it, and a sample. This is the review surface for
// the matrix — every line here is a feature the importer intentionally doesn't handle yet.
function featureTable(entries: CorpusEntry[]): void {
    const models = new Map<string, Set<string>>();
    for (const e of entries) {
        if (!e.scene) continue;
        for (const f of entryOf(e.scene).unsupported) {
            let set = models.get(f);
            if (!set) models.set(f, (set = new Set()));
            set.add(e.model);
        }
    }
    const rows = [...models.entries()].sort(
        (a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0]),
    );
    console.log(
        "\n  SKIPPED FEATURES (intentionally unimplemented — each must map to a backlog line)",
    );
    if (rows.length === 0) console.log("    (none — corpus fully supported)");
    const width = Math.max(0, ...rows.map((r) => r[0].length));
    for (const [feature, set] of rows) {
        const sample = [...set].sort().slice(0, 3).join(", ");
        console.log(
            `    ${feature.padEnd(width)}  ${String(set.size).padStart(3)}  e.g. ${sample}`,
        );
    }
}

// every (model, variant) that differs from the committed matrix — the regression/new-capability surface
function driftReport(entries: CorpusEntry[], pinned: Matrix | null): number {
    if (!pinned) {
        console.log("\n  DRIFT\n    no committed matrix — run with --write to create it");
        return 0;
    }
    const lines: string[] = [];
    const seen = new Set<string>();
    for (const e of entries) {
        const key = `${e.model}/${e.variant}`;
        seen.add(key);
        if (e.error) {
            lines.push(`    ERROR  ${key}: ${e.error}`);
            continue;
        }
        const now = entryOf(e.scene!);
        const was = pinned[e.model]?.[e.variant];
        if (!was) lines.push(`    NEW    ${key}: ${now.status} ${JSON.stringify(now.unsupported)}`);
        else if (!sameEntry(now, was))
            lines.push(
                `    CHANGE ${key}: ${was.status} ${JSON.stringify(was.unsupported)} (m${was.meshes}) → ${now.status} ${JSON.stringify(now.unsupported)} (m${now.meshes})`,
            );
    }
    for (const model of Object.keys(pinned))
        for (const variant of Object.keys(pinned[model]))
            if (!seen.has(`${model}/${variant}`))
                lines.push(`    GONE   ${model}/${variant}: dropped from corpus`);
    console.log("\n  DRIFT");
    console.log(lines.length ? lines.join("\n") : "    none — matrix matches the corpus");
    return lines.length;
}

async function main(): Promise<void> {
    if (!corpusPresent()) {
        console.error(
            "\n[gltf-conformance] corpus absent — init the gltf-sample-assets submodule:\n" +
                "  git submodule update --init reference/gltf-sample-assets\n",
        );
        process.exit(1);
    }
    const write = process.argv.includes("--write");
    console.log("[gltf-conformance] walking the corpus through the deviceless importer…");
    const entries = await walkCorpus();

    statusTable(entries);
    featureTable(entries);
    const pinned = loadMatrix();
    const drift = driftReport(entries, pinned);
    const errors = entries.filter((e) => e.error).length;

    if (write) {
        const matrix = buildMatrix(entries);
        await Bun.write(MATRIX_PATH, `${JSON.stringify(matrix, null, 2)}\n`);
        console.log(`\n  wrote ${MATRIX_PATH} (${Object.keys(matrix).length} models)`);
        process.exit(errors ? 1 : 0);
    }

    if (errors) {
        console.error(`\n[gltf-conformance] ${errors} parse error(s) — the importer regressed`);
        process.exit(1);
    }
    if (drift) {
        console.error(
            `\n[gltf-conformance] ${drift} drift row(s) — review, then --write to re-pin`,
        );
        process.exit(1);
    }
    console.log("\n[gltf-conformance] clean — matrix pins the corpus");
}

main();
