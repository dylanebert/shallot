import { existsSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";
import { scaffold, TEMPLATES, template } from "../../create-shallot/index";

// examples/templates/* are generated artifacts: the published `bun create shallot` templates
// rendered with workspace deps. Regenerating them here (run from build.ts) keeps the in-repo
// examples byte-identical to what a user scaffolds, with create-shallot as the one source.
const root = resolve(import.meta.dir, "../../..");

// install/build artifacts (the dirs the template's own .gitignore covers) aren't template files.
const artifact = (rel: string) =>
    rel.startsWith("node_modules/") || rel.startsWith("dist/") || rel.startsWith("build/");

for (const { dir } of TEMPLATES) {
    const files = template(dir, { shallot: "workspace:*" });
    const target = resolve(root, "examples/templates", dir);
    // remove any project file the template doesn't emit so the dir equals the template output exactly
    if (existsSync(target)) {
        for await (const rel of new Glob("**/*").scan({
            cwd: target,
            dot: true,
            onlyFiles: true,
        })) {
            if (!artifact(rel) && !(rel in files)) rmSync(resolve(target, rel));
        }
    }
    scaffold(target, files);
}
console.log(`Generated ${TEMPLATES.map((t) => `examples/templates/${t.dir}`).join(", ")}`);
