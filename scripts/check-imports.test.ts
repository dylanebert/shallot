import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
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

check(
    "import boundary resolves TypeScript specifiers",
    {
        claim: "the import boundary resolves TypeScript specifiers, scans each source extension and rejects imports missing from the compiler trace",
        size: "integration",
        subject: "scripts/check-imports",
    },
    () => {
        withFixture((root) => {
            writeFileSync(
                resolve(root, "package.json"),
                JSON.stringify({
                    name: "@dylanebert/shallot",
                    exports: {
                        "./input": "./src/core/input/index.ts",
                        "./extras": "./src/extras/index.ts",
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
            put(
                root,
                "harness/index.ts",
                'export type Internal = import("../engine/runtime/internal").Internal;\n',
            );
            put(root, "engine/runtime/index.ts", "export {};\n");
            put(root, "engine/runtime/internal.ts", "export interface Internal {}\n");
            put(
                root,
                "transitional/legacy/index.ts",
                "// Destination: engine; owner: legacy.md.\nexport {};\n",
            );

            expect(checkImports(root)).toEqual([
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
                "src/harness/index.ts:1: import past engine/runtime/index.ts → engine/runtime/internal.ts",
                "src/standard/loading/index.ts:1: standard imports outward to extras/index",
                "src/transitional/legacy/index.ts:1: // Destination: engine; owner: legacy.md.",
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
    },
);
