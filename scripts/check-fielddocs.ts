#!/usr/bin/env bun
/**
 * field-docs freshness gate (part of `bun check`).
 *
 * editor/src/lib/fielddocs.json is the annotation-sourced UI reference data, generated from component
 * JSDoc by `bun run build` (packages/shallot/scripts/fielddocs.ts). The committed copy can drift when a
 * field's doc comment changes without a rebuild, so this re-runs the generator and byte-compares. The
 * editor imports the committed JSON (the browser can't read source), so a stale copy ships wrong tooltips.
 */
import { resolve } from "node:path";
import { componentDocs, serializeDocs } from "../packages/shallot/scripts/fields";

const file = resolve(import.meta.dir, "../packages/shallot/editor/src/lib/fielddocs.json");
const want = serializeDocs(await componentDocs());
const have = await Bun.file(file)
    .text()
    .catch(() => "");

if (have !== want) {
    console.error("✗ field docs out of date (run `bun run build`)");
    process.exit(1);
}
