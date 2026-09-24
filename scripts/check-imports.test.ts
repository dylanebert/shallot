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

function expectOne(root: string, reason: string): void {
    const reds = checkImports(root);
    expect(reds).toHaveLength(1);
    expect(reds[0]).toContain(reason);
}

check(
    "import boundary fixture rules red",
    {
        claim: "the import boundary rejects outward and tooling imports, sibling imports, physics-rendering imports, index bypasses and transitional modules",
    },
    () => {
        withFixture((root) => {
            put(root, "standard/rendering/index.ts", 'import "../../extras/fog";\n');
            put(root, "extras/fog/index.ts", "export {};\n");
            expectOne(root, "standard imports outward to extras/fog");
        });
        withFixture((root) => {
            put(
                root,
                "core/rendering/index.ts",
                'import type { Plan } from "../../project/generate";\n',
            );
            put(root, "project/generate.ts", "export interface Plan {}\n");
            expectOne(root, "game module core/rendering imports tooling module project");
        });
        withFixture((root) => {
            put(root, "core/rendering/index.ts", 'export { input } from "../input";\n');
            put(root, "core/input/index.ts", "export const input = 1;\n");
            expectOne(root, "sibling import core/rendering → core/input");
        });
        withFixture((root) => {
            put(
                root,
                "extras/physics/index.ts",
                'export const load = () => import("../../core/rendering");\n',
            );
            put(root, "core/rendering/index.ts", "export {};\n");
            expectOne(
                root,
                "physics module extras/physics imports rendering module core/rendering",
            );
        });
        withFixture((root) => {
            put(
                root,
                "harness/index.ts",
                'export type Internal = import("../engine/runtime/internal").Internal;\n',
            );
            put(root, "engine/runtime/index.ts", "export {};\n");
            put(root, "engine/runtime/internal.ts", "export interface Internal {}\n");
            expectOne(root, "import past engine/runtime/index.ts → engine/runtime/internal.ts");
        });
        withFixture((root) => {
            put(
                root,
                "transitional/legacy/index.ts",
                "// Destination: engine; owner: legacy.md.\nexport {};\n",
            );
            expectOne(root, "// Destination: engine; owner: legacy.md.");
        });
    },
);
