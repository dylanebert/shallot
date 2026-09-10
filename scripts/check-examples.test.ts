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
    // every cone in the fixture registry needs a real subject, or the completeness clause reds the
    // baseline and no mutation below can be attributed to itself
    writeFileSync(resolve(root, "examples/flows/flow/main.ts"), "export const flow = 1;\n");
    writeFileSync(resolve(root, "examples/gym/main.ts"), "export const gym = 1;\n");
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
        gate: "bun run recipes --recipe static",
        static: "fixture has no runtime behavior",
    },
    {
        dir: "examples/flows/flow",
        tier: "showcase",
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
            gate: "bun run recipes --recipe moving",
        },
    ];
    const errors = checkExamples(root, rows).join("\n");
    expect(errors).toContain("recipe has neither src/smoke.ts nor static reason: moving");
    expect(errors).toContain("recipe manifest does not wire src/smoke.ts: moving");
    expect(errors).not.toContain("recipe has no CHECKS entry: moving");
});

test("smoked recipe rows must use the recipe selector", () => {
    const root = make();
    mkdirSync(resolve(root, "examples/recipes/moving/src"), { recursive: true });
    writeFileSync(resolve(root, "examples/recipes/moving/src/smoke.ts"), "export default {};\n");
    writeFileSync(
        resolve(root, "examples/recipes/moving/shallot.json"),
        '{"plugins":["./src/smoke.ts"]}\n',
    );
    const rows = [
        ...registry(),
        {
            dir: "examples/recipes/moving",
            tier: "recipes" as const,
            covers: ["examples/recipes/moving/**"],
            gate: "bunx shallot verify examples/recipes/moving",
        },
    ];
    expect(checkExamples(root, rows)).toContain(
        'recipe gate must use selector "bun run recipes --recipe moving": moving',
    );
});

// A bare `bunx shallot verify` row is spawned by the stage-close selector through `sh -c`, missing the
// display guard. The static rows used to be exempt from the selector rule and carried exactly that shape.
test("a static recipe row must use the selector too", () => {
    const root = make();
    const rows = registry().map((row) =>
        row.dir === "examples/recipes/static"
            ? { ...row, gate: "bunx shallot verify examples/recipes/static" }
            : row,
    );
    expect(checkExamples(root, rows)).toContain(
        'recipe gate must use selector "bun run recipes --recipe static": static',
    );
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

test("autonomous showcase rows require an imported motion arm", () => {
    const root = make();
    expect(checkExamples(root, registry(true))).toContain(
        "autonomous showcase has no imported motion arm: examples/showcase/demo",
    );
});
for (const helper of ["assertMotion", "frameDifference"]) {
    test(`autonomous showcase accepts the published ${helper} presence`, () => {
        const root = make();
        writeFileSync(
            resolve(root, "examples/showcase/demo/test/demo.playwright.ts"),
            `import { isDegradedBootMessage, ${helper} } from '@dylanebert/shallot/harness';\n`,
        );
        expect(checkExamples(root, registry(true))).toEqual([]);
    });
}

for (const extra of [
    "",
    "import { frameDifferences } from '@dylanebert/shallot/harness';",
    "import { frameDifference } from 'another-package';",
    "import { assertMotion } from 'another-package';",
]) {
    test(`autonomous showcase refuses missing published motion presence: ${extra || "none"}`, () => {
        const root = make();
        writeFileSync(
            resolve(root, "examples/showcase/demo/test/demo.playwright.ts"),
            `import { isDegradedBootMessage } from '@dylanebert/shallot/harness';\n${extra}\n`,
        );
        expect(checkExamples(root, registry(true))).toEqual([
            "autonomous showcase has no imported motion arm: examples/showcase/demo",
        ]);
        expect(checkExamples(root, registry(false))).toEqual([]);
    });
}

test("either published motion reading satisfies the autonomous showcase arm", () => {
    for (const symbol of ["assertMotion", "frameDifference"]) {
        const root = make();
        mkdirSync(resolve(root, "examples/showcase/demo/test"), { recursive: true });
        writeFileSync(
            resolve(root, "examples/showcase/demo/test/motion.playwright.ts"),
            `import { ${symbol} } from "@dylanebert/shallot/harness";\n`,
        );
        expect(checkExamples(root, registry(true)).join("\n")).not.toContain(
            "no imported motion arm",
        );
    }
});

test("a live shared cover cannot hide an uncovered example source", () => {
    const root = make();
    const rows = registry();
    rows[1].covers = ["examples/gym/**"];
    expect(checkExamples(root, rows)).toEqual([
        "example source has no covers row: examples/flows/flow/main.ts",
    ]);
});

test("removed cover, orphaned glob, and renamed or deleted source refuse independently", () => {
    for (const mutation of ["remove", "orphan", "rename", "delete"]) {
        const root = make();
        const rows = registry();
        rows[1].covers = ["examples/flows/flow/main.ts"];
        expect(checkExamples(root, rows)).toEqual([]);
        if (mutation === "remove") rows[1].covers = [];
        if (mutation === "orphan") rows[1].covers.push("examples/flows/flow/*.svelte");
        if (mutation === "rename" || mutation === "delete") {
            rmSync(resolve(root, "examples/flows/flow/main.ts"));
            if (mutation === "rename")
                writeFileSync(resolve(root, "examples/flows/flow/renamed.ts"), "export {};\n");
        }
        const errors = checkExamples(root, rows).join("\n");
        expect(errors).toContain(
            mutation === "remove" ? "declares no covers glob" : "covers glob matches no file",
        );
        if (mutation === "rename") expect(errors).toContain("example source has no covers row");
    }
});

test("every discovered directory must yield governed files", () => {
    const root = make();
    rmSync(resolve(root, "examples/flows/flow/main.ts"));
    const rows = registry();
    rows[1].covers = ["examples/gym/**"];
    expect(checkExamples(root, rows)).toEqual([
        "example directory yielded no governed source files: examples/flows/flow",
    ]);
});

for (const output of [
    "dist",
    "out",
    "build",
    "node_modules",
    "test-results",
    "playwright-report",
]) {
    test(`output ${output} is not demanded by source covers`, () => {
        const root = make();
        const rows = registry();
        rows[1].covers = ["examples/flows/flow/main.ts"];
        mkdirSync(resolve(root, "examples/flows/flow", output));
        writeFileSync(resolve(root, "examples/flows/flow", output, "generated.ts"), "export {};\n");
        expect(checkExamples(root, rows)).toEqual([]);
        writeFileSync(resolve(root, "examples/flows/flow/uncovered.ts"), "export {};\n");
        expect(checkExamples(root, rows)).toEqual([
            "example source has no covers row: examples/flows/flow/uncovered.ts",
        ]);
    });
}

test("a complete static fixture is green", () => {
    const root = make();
    expect(checkExamples(root, registry())).toEqual([]);
});
