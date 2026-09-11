import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { readCheckDeclarations } from "./surface";

// `bun run examples:index [--check]`: emit `examples/AGENTS.md` from each source-visible
// `examples/*/shallot.json` (`--root <dir>` is for isolated fixture tests). The index is never
// hand-written; `--check` reds when the committed file differs from what the declarations
// generate. Every source-visible example dir must declare both fields.

const args = Bun.argv.slice(2);
const rootIndex = args.indexOf("--root");
if (rootIndex !== -1 && args[rootIndex + 1] === undefined) {
    console.error("✗ --root requires a directory");
    process.exit(1);
}
const root = resolve(
    rootIndex === -1 ? resolve(import.meta.dir, "..") : (args[rootIndex + 1] as string),
);
const examples = resolve(root, "examples");
const out = resolve(examples, "AGENTS.md");
const KINDS = ["recipe", "showcase"] as const;
type Kind = (typeof KINDS)[number];

const evidence = Bun.spawnSync(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "examples"],
    { cwd: root, stdout: "pipe", stderr: "pipe" },
);
if (!evidence.success) {
    console.error("✗ `git ls-files` failed — examples index needs a Git source file list.");
    process.exit(1);
}

const names = new Set<string>();
for (const path of evidence.stdout.toString().split("\0")) {
    if (!path.startsWith("examples/")) continue;
    const relative = path.slice("examples/".length);
    const slash = relative.indexOf("/");
    if (slash > 0 && slash < relative.length - 1) names.add(relative.slice(0, slash));
}

const rows: { name: string; kind: Kind; description: string; checkSize: string }[] = [];
const errors: string[] = [];
for (const name of [...names].sort()) {
    const path = resolve(examples, name, "shallot.json");
    if (!existsSync(path)) {
        errors.push(`examples/${name}/ has no shallot.json`);
        continue;
    }
    const { kind, description, problem, check } = JSON.parse(readFileSync(path, "utf8"));
    if (!KINDS.includes(kind))
        errors.push(`examples/${name}/shallot.json: kind must be ${KINDS.join(" | ")}`);
    else if (typeof description !== "string" || description.trim() === "")
        errors.push(`examples/${name}/shallot.json: description is missing`);
    else {
        let checkSize = "-";
        if (check !== undefined) {
            if (
                Array.isArray(check) ||
                check === null ||
                typeof check !== "object" ||
                typeof check.file !== "string" ||
                Object.keys(check).some((field) => field !== "file")
            ) {
                errors.push(`examples/${name}/shallot.json: check must be { file }`);
            } else {
                const result = readCheckDeclarations(root, resolve(examples, name, check.file));
                errors.push(...result.errors);
                checkSize = [...new Set(result.rows.map((row) => row.size))].join("/") || "-";
            }
        }
        rows.push({
            name,
            kind,
            checkSize,
            description: (typeof problem === "string" && problem.trim() !== ""
                ? problem
                : description
            ).trim(),
        });
    }
}
if (errors.length > 0) {
    for (const e of errors) console.error(`✗ ${e}`);
    process.exit(1);
}

const cell = (s: string) => s.replaceAll("|", "\\|");
const lines = [
    "# Examples",
    "",
    "From `bun run examples:index` and `examples/*/shallot.json`; edit manifests. Use `bunx shallot dev examples/<name>`.",
];
for (const kind of KINDS) {
    const own = rows.filter((r) => r.kind === kind);
    if (own.length === 0) continue;
    lines.push("", kind === "recipe" ? "## Recipes" : "## Showcase", "");
    if (kind === "recipe") {
        lines.push("| name | description | add |", "| --- | --- | --- |");
        for (const r of own)
            lines.push(
                `| \`${r.name}\` | ${cell(r.description)}${r.checkSize === "-" ? "" : ` (check: ${cell(r.checkSize)})`} | \`bunx shallot add ${r.name}\` |`,
            );
    } else {
        lines.push("| name | description |", "| --- | --- |");
        for (const r of own) lines.push(`| \`${r.name}\` | ${cell(r.description)} |`);
    }
}
const text = `${lines.join("\n")}\n`;

if (process.argv.includes("--check")) {
    const committed = existsSync(out) ? readFileSync(out, "utf8") : "";
    if (committed !== text) {
        console.error(
            "✗ examples/AGENTS.md differs from the manifests; run `bun run examples:index`.",
        );
        process.exit(1);
    }
} else {
    writeFileSync(out, text);
}
