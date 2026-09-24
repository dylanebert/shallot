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
        claim: "the import boundary resolves self-name, extension and alias specifiers and scans each TypeScript source extension",
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

            const reds = checkImports(root);
            expect(reds).toHaveLength(12);
            for (const file of [
                "src/core/rendering/index.ts",
                "src/core/rendering/js-path.ts",
                "src/core/rendering/alias.ts",
                "src/core/rendering/view.tsx",
                "src/core/rendering/view.mts",
                "src/core/rendering/view.cts",
                "src/core/rendering/types.d.ts",
            ]) {
                expect(reds.some((red) => red.startsWith(`${file}:`))).toBe(true);
            }
            expect(reds.some((red) => red.startsWith("src/standard/loading/index.ts:"))).toBe(true);
            expect(reds.some((red) => red.startsWith("src/core/rendering/tooling.ts:"))).toBe(true);
            expect(reds.some((red) => red.startsWith("src/extras/physics/index.ts:"))).toBe(true);
            expect(reds.some((red) => red.startsWith("src/harness/index.ts:"))).toBe(true);
            expect(
                reds.some((red) => red.includes("// Destination: engine; owner: legacy.md.")),
            ).toBe(true);
            expect(reds.some((red) => red.includes("ignored.test.ts"))).toBe(false);
        });
    },
);
