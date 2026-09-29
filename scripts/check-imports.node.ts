import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(20_000);

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { checkImports } from "./check-imports";

function withFixture(run: (root: string) => void): void {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-check-imports-"));
    try {
        run(root);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

function put(root: string, path: string, source: string): void {
    const file = resolve(root, "src", path);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, source);
}

test("the import boundary resolves TypeScript specifiers, scans each source extension and rejects imports missing from the compiler trace", () => {
    withFixture((root) => {
        writeFileSync(
            resolve(root, "package.json"),
            JSON.stringify({
                name: "@dylanebert/shallot",
                exports: {
                    "./input": "./src/core/input/index.ts",
                    "./rendering": "./src/core/rendering/index.ts",
                    "./extras": "./src/extras/index.ts",
                    "./vite": {
                        types: "./src/project/vite.ts",
                        default: "./dist/vite.js",
                    },
                },
            }),
        );
        writeFileSync(
            resolve(root, "tsconfig.json"),
            JSON.stringify({
                compilerOptions: {
                    module: "ESNext",
                    moduleResolution: "Bundler",
                    target: "ESNext",
                    noEmit: true,
                    strict: true,
                    skipLibCheck: true,
                    types: [],
                    paths: { "@core/*": ["./src/core/*"] },
                },
                include: ["src"],
            }),
        );

        put(root, "core/input/index.ts", "export interface Input {}\n");
        put(root, "core/input/fixtures/page.ts", 'import "@dylanebert/shallot/rendering";\n');
        put(root, "core/input/fixtures/support.ts", 'import "./page";\n');
        put(root, "core/input/check.fixture.ts", 'import "@dylanebert/shallot/rendering";\n');
        put(
            root,
            "core/input/product.ts",
            'import "./fixtures/page";\nimport "./check.fixture";\n',
        );
        put(root, "project/vite.ts", "export {};\n");
        put(root, "core/rendering/index.ts", 'import "@dylanebert/shallot/input";\n');
        put(root, "core/rendering/js-path.ts", 'import "../input/index.js";\n');
        put(root, "core/rendering/alias.ts", 'import "@core/input";\n');
        put(root, "core/rendering/view.tsx", 'import "@dylanebert/shallot/input";\n');
        put(root, "core/rendering/view.mts", 'import "@dylanebert/shallot/input";\n');
        put(
            root,
            "core/rendering/view.cts",
            'import type { Input } from "@dylanebert/shallot/input";\n',
        );
        put(
            root,
            "core/rendering/types.d.ts",
            'import type { Input } from "@dylanebert/shallot/input";\n',
        );
        put(
            root,
            "core/rendering/missing.d.ts",
            'import type { Missing } from "not-installed";\nexport type Broken = Missing;\n',
        );
        put(root, "core/rendering/ignored.test.ts", 'import "@dylanebert/shallot/extras";\n');
        put(root, "extras/index.ts", "export {};\n");
        put(root, "standard/loading/index.ts", 'import "@dylanebert/shallot/extras";\n');
        put(
            root,
            "core/rendering/tooling.ts",
            'import type { Plan } from "../../project/generate";\nvoid (0 as unknown as Plan);\n',
        );
        put(root, "project/generate.ts", "export interface Plan {}\n");
        put(
            root,
            "extras/physics/index.ts",
            'export const load = () => import("../../core/rendering");\n',
        );
        put(root, "engine/runtime/index.ts", "export {};\n");
        put(root, "engine/runtime/internal.ts", "export interface Internal {}\n");
        put(
            root,
            "transitional/legacy/index.ts",
            "// Destination: engine; owner: legacy.md.\nexport {};\n",
        );
        put(
            root,
            "transitional/legacy/product.ts",
            'import "../../core/input/fixtures/page";\nimport "../../core/input/check.fixture";\n',
        );

        expect(checkImports(root)).toEqual([
            "src/core/input/product.ts:1: product module imports private fixture core/input/fixtures/page.ts",
            "src/core/input/product.ts:2: product module imports private fixture core/input/check.fixture.ts",
            "src/core/rendering/alias.ts:1: sibling import core/rendering → core/input",
            "src/core/rendering/index.ts:1: sibling import core/rendering → core/input",
            "src/core/rendering/js-path.ts:1: sibling import core/rendering → core/input",
            'src/core/rendering/missing.d.ts:1: unresolved import "not-installed"',
            "src/core/rendering/tooling.ts:1: game module core/rendering imports tooling module project",
            "src/core/rendering/types.d.ts:1: sibling import core/rendering → core/input",
            "src/core/rendering/view.cts:1: sibling import core/rendering → core/input",
            "src/core/rendering/view.mts:1: sibling import core/rendering → core/input",
            "src/core/rendering/view.tsx:1: sibling import core/rendering → core/input",
            "src/extras/physics/index.ts:1: physics module extras/physics imports rendering module core/rendering",
            "src/standard/loading/index.ts:1: standard imports outward to extras/index",
            "src/transitional/legacy/product.ts:1: product module imports private fixture core/input/fixtures/page.ts",
            "src/transitional/legacy/product.ts:2: product module imports private fixture core/input/check.fixture.ts",
            "src/transitional/legacy/index.ts:1: pending roadmap migration (still red): // Destination: engine; owner: legacy.md.",
        ]);
    });
    withFixture((root) => {
        writeFileSync(
            resolve(root, "tsconfig.json"),
            JSON.stringify({
                compilerOptions: {
                    module: "ESNext",
                    moduleResolution: "Bundler",
                    noEmit: true,
                },
                include: ["src/extras"],
            }),
        );
        put(root, "core/rendering/index.ts", 'import "../../extras/fog";\n');
        put(root, "extras/fog/index.ts", "export {};\n");
        expect(checkImports(root)).toEqual([
            'src/core/rendering/index.ts:1: unresolved import "../../extras/fog"',
        ]);
    });
});

test("the repository engine runtime does not import past the ECS barrel", () => {
    const root = resolve(import.meta.dir, "..");
    expect(
        checkImports(root).filter((red) => red.startsWith("src/engine/runtime/gpu.ts:")),
    ).toEqual([]);
});

test("engine runtime imports ECS APIs through the ECS barrel", () => {
    withFixture((root) => {
        writeFileSync(
            resolve(root, "tsconfig.json"),
            JSON.stringify({
                compilerOptions: {
                    module: "ESNext",
                    moduleResolution: "Bundler",
                    noEmit: true,
                },
                include: ["src"],
            }),
        );
        put(root, "engine/ecs/index.ts", 'export { useState } from "./component";\n');
        put(root, "engine/ecs/component.ts", "export function useState(): void {}\n");
        put(
            root,
            "engine/runtime/gpu.ts",
            'import { useState } from "../ecs/component";\nvoid useState;\n',
        );

        expect(checkImports(root)).toEqual([
            "src/engine/runtime/gpu.ts:1: import past engine/ecs/index.ts → engine/ecs/component.ts",
        ]);

        put(root, "engine/runtime/gpu.ts", 'import { useState } from "../ecs";\nvoid useState;\n');
        expect(checkImports(root)).toEqual([]);
    });
});
