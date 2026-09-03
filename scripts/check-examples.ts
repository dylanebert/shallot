import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { EXAMPLE_GATES, type ExampleGate } from "./example-gates";

const text = (path: string): string => readFileSync(path, "utf8");
const files = (dir: string, suffix: string): string[] => {
    if (!existsSync(dir)) return [];
    return readdirSync(dir, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith(suffix))
        .map((entry) => resolve(entry.parentPath, entry.name));
};
const childDirs = (dir: string): string[] =>
    existsSync(dir)
        ? readdirSync(dir, { withFileTypes: true })
              .filter((entry) => entry.isDirectory())
              .map((entry) => entry.name)
              .sort()
        : [];

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
    for (const row of registry.filter((entry) => entry.tier === "showcase" && entry.motion)) {
        const specs = files(resolve(root, row.dir), ".playwright.ts").map(text).join("\n");
        if (
            !/import[\s\S]*?\bassertMotion\b[\s\S]*?from\s*["']@dylanebert\/shallot\/harness["']/.test(
                specs,
            )
        )
            errors.push(`autonomous showcase has no imported assertMotion arm: ${row.dir}`);
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
