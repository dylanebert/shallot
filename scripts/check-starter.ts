#!/usr/bin/env bun
/**
 * starter freshness gate (part of `bun check`).
 *
 * examples/templates/* are generated from packages/create-shallot's `template()` by
 * `bun run build` (packages/shallot/scripts/starter.ts). The committed dirs can drift when
 * create-shallot changes without a rebuild, so this byte-compares each file against a fresh
 * render and flags any project file the template doesn't emit. It does not typecheck the
 * generated code against the engine — the root tsc already includes examples/, so a template
 * that calls a dropped API fails there.
 */
import { resolve } from "node:path";
import { Glob } from "bun";
import { TEMPLATES, template } from "../packages/create-shallot/index";

const root = resolve(import.meta.dir, "..");
const artifact = (rel: string) =>
    rel.startsWith("node_modules/") || rel.startsWith("dist/") || rel.startsWith("build/");

const fail: string[] = [];

for (const { dir } of TEMPLATES) {
    const files = template(dir, { shallot: "workspace:*" });
    const target = resolve(root, "examples/templates", dir);
    for (const [rel, want] of Object.entries(files)) {
        const file = Bun.file(resolve(target, rel));
        if (!(await file.exists())) {
            fail.push(`${dir}/${rel}: missing`);
            continue;
        }
        if ((await file.text()) !== want) fail.push(`${dir}/${rel}: stale`);
    }

    // an extra committed file (install/build artifacts excluded) means the template dropped it
    for await (const rel of new Glob("**/*").scan({ cwd: target, dot: true, onlyFiles: true })) {
        if (!artifact(rel) && !(rel in files)) {
            fail.push(`${dir}/${rel}: orphan (not in the template)`);
        }
    }
}

if (fail.length) {
    console.error(
        `✗ templates out of date (run \`bun run build\`):\n${fail.map((f) => `  ${f}`).join("\n")}`,
    );
    process.exit(1);
}
