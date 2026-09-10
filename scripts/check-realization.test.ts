import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { checkRealization } from "./check-realization";

test("declared realization grants physical and registered targets, refuses missing bin/files and unregistered descendants", async () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-realization-"));
    const write = (file: string, value: unknown) => {
        mkdirSync(dirname(join(root, file)), { recursive: true });
        writeFileSync(join(root, file), JSON.stringify(value));
    };
    try {
        write("package.json", { workspaces: ["packages/*"] });
        write("packages/shallot/bin/cli.ts", "canonical CLI");
        const pkg = {
            bin: { shallot: "./bin/cli.ts" },
            files: [
                "dist",
                "examples",
                "rust/audio/pkg",
                "!absent-exclusion",
                "physical",
            ],
        };
        write("packages/shallot/package.json", pkg);
        write("packages/shallot/physical", "maintained");
        expect(await checkRealization(root)).toEqual([]);
        write("packages/shallot/package.json", { ...pkg, bin: { shallot: "./bin" } });
        expect((await checkRealization(root)).join("\n")).toContain("bin: ./bin is missing");
        write("packages/shallot/package.json", {
            ...pkg,
            bin: { shallot: "./bin/gone.ts" },
            files: [...pkg.files, "absent", "dist/undeclared.js"],
        });
        const errors = await checkRealization(root);
        expect(errors).toHaveLength(3);
        expect(errors.join("\n")).toContain("bin: ./bin/gone.ts");
        expect(errors.join("\n")).toContain("files: absent");
        expect(errors.join("\n")).toContain("files: dist/undeclared.js");
        write("packages/consumer/package.json", { bin: "missing.ts", files: ["dist"] });
        expect(await checkRealization(root)).toHaveLength(5);
        write("packages/consumer/missing.ts", "physical bin");
        write("packages/consumer/dist/index.js", "physical files");
        expect(await checkRealization(root)).toHaveLength(3);
        write("packages/shallot/package.json", pkg);
        expect(await checkRealization(root)).toEqual([]);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
