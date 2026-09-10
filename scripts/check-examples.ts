import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { relative, resolve, sep } from "node:path";
import { Glob } from "bun";
import { EXAMPLE_GATES, type ExampleGate } from "./example-gates";

const text = (path: string): string => readFileSync(path, "utf8");
// A workspace's `node_modules` links back to the root, which a recursive read loops on.
const files = (dir: string, suffix: string): string[] => {
    if (!existsSync(dir)) return [];
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
        const path = resolve(dir, entry.name);
        if (entry.isDirectory()) return entry.name === "node_modules" ? [] : files(path, suffix);
        return entry.isFile() && entry.name.endsWith(suffix) ? [path] : [];
    });
};
const childDirs = (dir: string): string[] =>
    existsSync(dir)
        ? readdirSync(dir, { withFileTypes: true })
              .filter((entry) => entry.isDirectory())
              .map((entry) => entry.name)
              .sort()
        : [];

/** True when at least one file under `root` matches `cover`. Walks the glob's own literal prefix rather
 *  than the whole tree, so an orphaned glob is cheap to detect and a live one stops at its first hit. */
export function globHasSubject(root: string, cover: string): boolean {
    if (!cover.includes("*")) return existsSync(resolve(root, cover));
    const literal = cover.split("/").slice(
        0,
        cover.split("/").findIndex((p) => p.includes("*")),
    );
    const base = resolve(root, literal.join("/") || ".");
    if (!existsSync(base)) return false;
    const glob = new Glob(cover);
    return files(base, "").some((path) => glob.match(relative(root, path).split(sep).join("/")));
}

/** Returns every corpus-shape violation. Keeping this pure result seam makes each clause fixture-testable. */
export function checkExamples(root: string, registry: ExampleGate[]): string[] {
    const errors: string[] = [];
    const discovered = [
        ...(["recipes", "flows", "showcase"] as const).flatMap((tier) =>
            childDirs(resolve(root, "examples", tier)).map((name) => `examples/${tier}/${name}`),
        ),
        ...(existsSync(resolve(root, "examples/gym")) ? ["examples/gym"] : []),
    ];
    const registered = registry.map((row) => row.dir);
    for (const dir of discovered.filter((dir) => !registered.includes(dir)))
        errors.push(`example directory has no registry row: ${dir}`);
    for (const dir of registered) {
        const path = resolve(root, dir);
        if (!existsSync(path) || !statSync(path).isDirectory())
            errors.push(`registry row names no example directory: ${dir}`);
    }

    const recipesSource = existsSync(resolve(root, "scripts/recipes.ts"))
        ? text(resolve(root, "scripts/recipes.ts"))
        : "";
    for (const recipe of childDirs(resolve(root, "examples/recipes"))) {
        const dir = resolve(root, "examples/recipes", recipe);
        const row = registry.find((entry) => entry.dir === `examples/recipes/${recipe}`);
        const scenes = files(dir, ".scene").map(text).join("\n");
        const smoke = resolve(dir, "src/smoke.ts");
        // Every recipe row — static or smoked — runs through the one selector. A bare
        // `bunx shallot verify <dir>` row is spawned by the stage-close selector through `sh -c` and
        // never reaches `scripts/verify.ts`, so it misses that wrapper's display guard and reds on a
        // software adapter instead of refusing. One gate shape keeps every row attributable.
        const expectedGate = `bun run recipes --recipe ${recipe}`;
        if (row && row.gate !== expectedGate)
            errors.push(`recipe gate must use selector "${expectedGate}": ${recipe}`);
        if (row?.static) {
            if (/\banimator\s*=|\bbody\s*=/.test(scenes))
                errors.push(`static recipe scene declares animator or body: ${recipe}`);
            if (existsSync(smoke)) errors.push(`static recipe also has src/smoke.ts: ${recipe}`);
            continue;
        }
        if (!existsSync(smoke))
            errors.push(`recipe has neither src/smoke.ts nor static reason: ${recipe}`);
        const manifestPath = resolve(dir, "shallot.json");
        const manifest = existsSync(manifestPath) ? text(manifestPath) : "";
        if (!/["']?\.\/src\/smoke(?:\.ts)?["']?/.test(manifest))
            errors.push(`recipe manifest does not wire src/smoke.ts: ${recipe}`);
        const checksBlock =
            recipesSource.match(/const CHECKS[^=]*=\s*\{([\s\S]*?)\n\};/)?.[1] ?? "";
        if (!new RegExp(`(?:["']${recipe}["']|\\b${recipe}\\b)\\s*:`).test(checksBlock))
            errors.push(`recipe has no CHECKS entry: ${recipe}`);
    }

    // Two-way cone completeness: a row with no cover is never selected, and a glob matching nothing is a
    // cone that silently stopped covering its subject (a renamed or deleted directory). Both read green
    // in the selector's dry-run, which is why they are refused here instead.
    for (const row of registry) {
        if (row.covers.length === 0) {
            errors.push(`registry row declares no covers glob: ${row.dir}`);
            continue;
        }
        for (const cover of row.covers) {
            if (!globHasSubject(root, cover))
                errors.push(`covers glob matches no file: ${row.dir} -> ${cover}`);
        }
    }

    const covers = registry.flatMap((row) => row.covers.map((cover) => new Glob(cover)));
    for (const dir of discovered) {
        let population = 0;
        for (const file of files(resolve(root, dir), "")) {
            const rel = relative(root, file).split(sep).join("/");
            if (
                rel
                    .split("/")
                    .some((part) =>
                        [
                            "node_modules",
                            "dist",
                            "out",
                            "build",
                            "test-results",
                            "playwright-report",
                        ].includes(part),
                    )
            )
                continue;
            if (!/\.(?:[cm]?[jt]sx?|svelte|scene|json|html|css)$/.test(rel)) continue;
            population++;
            if (!covers.some((cover) => cover.match(rel)))
                errors.push(`example source has no covers row: ${rel}`);
        }
        if (population === 0)
            errors.push(`example directory yielded no governed source files: ${dir}`);
    }

    for (const scene of files(resolve(root, "examples"), ".scene")) {
        const source = text(scene);
        for (const match of source.matchAll(/\banimator\s*=\s*(["'])(.*?)\1/g)) {
            if (!/\bclip\s*:/.test(match[2]))
                errors.push(`animator names no clip: ${scene.slice(root.length + 1)}`);
        }
    }

    for (const spec of files(resolve(root, "examples/showcase"), ".playwright.ts")) {
        const source = text(spec);
        if (
            !/import[\s\S]*?\bisDegradedBootMessage\b[\s\S]*?from\s*["']@dylanebert\/shallot\/harness["']/.test(
                source,
            )
        )
            errors.push(
                `showcase Playwright spec does not import isDegradedBootMessage: ${spec.slice(root.length + 1)}`,
            );
    }
    // Either published motion reading counts: `assertMotion` is the one-shot form, `frameDifference`
    // the non-throwing one a retrying `expect.poll` needs. The property is an imported motion arm from
    // the published harness, not one spelling of it.
    for (const row of registry.filter((entry) => entry.tier === "showcase" && entry.motion)) {
        const specs = files(resolve(root, row.dir), ".playwright.ts").map(text).join("\n");
        if (
            !/import[\s\S]*?\b(?:assertMotion|frameDifference)\b[\s\S]*?from\s*["']@dylanebert\/shallot\/harness["']/.test(
                specs,
            )
        )
            errors.push(`autonomous showcase has no imported motion arm: ${row.dir}`);
    }
    return errors;
}

if (import.meta.main) {
    const root = resolve(import.meta.dir, "..");
    const errors = checkExamples(root, EXAMPLE_GATES);
    if (errors.length) {
        console.error(errors.map((error) => `✗ ${error}`).join("\n"));
        process.exit(1);
    }
    console.log(`✓ example corpus (${EXAMPLE_GATES.length} registered directories)`);
}
