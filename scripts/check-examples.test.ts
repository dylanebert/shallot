import { afterEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { checkExamples } from "./check-examples";
import type { ExampleGate } from "./example-gates";

const fixtures: string[] = [];
const make = (): string => {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-check-examples-"));
    fixtures.push(root);
    mkdirSync(resolve(root, "examples/recipes/static/public/scenes"), { recursive: true });
    mkdirSync(resolve(root, "examples/flows/flow"), { recursive: true });
    mkdirSync(resolve(root, "examples/showcase/demo/test"), { recursive: true });
    mkdirSync(resolve(root, "examples/gym"), { recursive: true });
    mkdirSync(resolve(root, "scripts"), { recursive: true });
    writeFileSync(
        resolve(root, "examples/recipes/static/public/scenes/main.scene"),
        "<entity />\n",
    );
    writeFileSync(
        resolve(root, "scripts/recipes.ts"),
        "const CHECKS: Record<string, string[]> = {\n    moving: ['moves'],\n};\n",
    );
    writeFileSync(
        resolve(root, "examples/showcase/demo/test/demo.playwright.ts"),
        "import { isDegradedBootMessage } from '@dylanebert/shallot/harness';\nvoid isDegradedBootMessage;\n",
    );
    return root;
};
const registry = (motion = false): ExampleGate[] => [
    {
        dir: "examples/recipes/static",
        tier: "recipes",
        covers: ["examples/recipes/static/**"],
        gate: "bunx shallot verify examples/recipes/static",
        static: "fixture has no runtime behavior",
    },
    {
        dir: "examples/flows/flow",
        tier: "flows",
        covers: ["examples/flows/flow/**"],
        gate: "bun run flows --flow flow",
    },
    {
        dir: "examples/showcase/demo",
        tier: "showcase",
        covers: ["examples/showcase/demo/**"],
        gate: "bun run --cwd examples/showcase/demo gate",
        motion,
    },
    {
        dir: "examples/gym",
        tier: "gym",
        covers: ["examples/gym/**"],
        gate: "bun bench --for examples/gym",
    },
];
afterEach(() => {
    for (const fixture of fixtures.splice(0)) rmSync(fixture, { recursive: true, force: true });
});

test("registry coverage is bidirectional", () => {
    const root = make();
    mkdirSync(resolve(root, "examples/flows/unregistered"));
    const rows = [...registry(), { ...registry()[0], dir: "examples/recipes/missing" }];
    const errors = checkExamples(root, rows);
    expect(errors).toContain("example directory has no registry row: examples/flows/unregistered");
    expect(errors).toContain("registry row names no example directory: examples/recipes/missing");
});

test("moving recipes require smoke, manifest wiring, and a CHECKS row", () => {
    const root = make();
    mkdirSync(resolve(root, "examples/recipes/moving/src"), { recursive: true });
    writeFileSync(
        resolve(root, "examples/recipes/moving/src/plugin.ts"),
        "export default { systems: [tick] };\n",
    );
    writeFileSync(resolve(root, "examples/recipes/moving/shallot.json"), "{}\n");
    const rows = [
        ...registry(),
        {
            dir: "examples/recipes/moving",
            tier: "recipes" as const,
            covers: ["examples/recipes/moving/**"],
            gate: "bunx shallot verify examples/recipes/moving",
        },
    ];
    const errors = checkExamples(root, rows).join("\n");
    expect(errors).toContain("recipe has neither src/smoke.ts nor static reason: moving");
    expect(errors).toContain("recipe manifest does not wire src/smoke.ts: moving");
    expect(errors).not.toContain("recipe has no CHECKS entry: moving");
});

test("every showcase Playwright spec imports the degraded-boot predicate", () => {
    const root = make();
    writeFileSync(
        resolve(root, "examples/showcase/demo/test/demo.playwright.ts"),
        "import { test } from '@playwright/test';\nvoid test;\n",
    );
    expect(checkExamples(root, registry()).join("\n")).toContain(
        "does not import isDegradedBootMessage",
    );
});

test("every animator attribute names a clip and cannot use the static opt-out", () => {
    const root = make();
    writeFileSync(
        resolve(root, "examples/recipes/static/public/scenes/main.scene"),
        '<entity animator="loop: 1; target: @ball" />\n',
    );
    const errors = checkExamples(root, registry()).join("\n");
    expect(errors).toContain("animator names no clip");
    expect(errors).toContain("static recipe scene declares animator or body: static");
});

test("autonomous showcase rows require an imported assertMotion arm", () => {
    const root = make();
    expect(checkExamples(root, registry(true))).toContain(
        "autonomous showcase has no imported assertMotion arm: examples/showcase/demo",
    );
});

test("a complete static fixture is green", () => {
    const root = make();
    expect(checkExamples(root, registry())).toEqual([]);
});
