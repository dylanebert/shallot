import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { validateDeclaration } from "./declaration";

check(
    "a declaration defaults its size, requirements and budget",
    { claim: "check() defaults a declaration to a hermetic unit row and its 250ms ceiling" },
    () => {
        expect(validateDeclaration("here", { claim: "unit" })).toEqual({
            claim: "unit",
            size: "unit",
            requires: [],
            budget: 250,
        });
        expect(validateDeclaration("here", { claim: "integration", size: "integration" })).toEqual({
            claim: "integration",
            size: "integration",
            requires: [],
            budget: 20000,
        });
    },
);

check(
    "a declaration refuses retired and unknown vocabulary",
    { claim: "check() refuses tier and class fields plus unknown sizes and requirement tags" },
    () => {
        expect(() => validateDeclaration("here", { claim: "old", tier: "step" })).toThrow(
            "retired field `tier`",
        );
        expect(() => validateDeclaration("here", { claim: "old", class: "pure" })).toThrow(
            "retired field `class`",
        );
        expect(() => validateDeclaration("here", { claim: "bad", size: "smoke" })).toThrow(
            "has size `smoke`",
        );
        expect(() => validateDeclaration("here", { claim: "bad", requires: ["network"] })).toThrow(
            "requirement tag `network`",
        );
        expect(() =>
            validateDeclaration("here", { claim: "bad", requires: ["gpu", "display", "deploy"] }),
        ).not.toThrow();
        expect(
            validateDeclaration("here", {
                claim: "integration",
                size: "integration",
                subject: ["src/a.ts", "src/b.ts"],
            }).subject,
        ).toEqual(["src/a.ts", "src/b.ts"]);
        expect(() => validateDeclaration("here", { claim: "bad", subject: [] })).toThrow(
            "non-empty string `subject`",
        );
        const tree = mkdtempSync(join(resolve(import.meta.dir, "../.."), ".surface-requirement-"));
        const file = join(tree, "missing.test.ts");
        writeFileSync(
            file,
            `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n` +
                'check("missing", { claim: "missing GPU refuses", requires: ["gpu"] }, () => {});\n',
        );
        try {
            const proc = Bun.spawnSync(["bun", "test", file], {
                cwd: resolve(import.meta.dir, "../.."),
                env: { ...process.env, SHALLOT_UNIT_ONLY: "" },
            });
            expect(proc.exitCode).not.toBe(0);
            expect(proc.stderr.toString() + proc.stdout.toString()).toContain(
                "refused check missing GPU refuses",
            );
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "a declaration enforces size ceilings and requirement shape",
    { claim: "check() rejects invalid requires values and budgets over the size ceiling" },
    () => {
        expect(() => validateDeclaration("here", { claim: "bad", requires: "chromium" })).toThrow(
            "array of strings",
        );
        for (const budget of [0, -1, Number.NaN, "10"]) {
            expect(() => validateDeclaration("here", { claim: "bad", budget })).toThrow(
                "positive finite `budget`",
            );
        }
        expect(() => validateDeclaration("here", { claim: "slow", budget: 251 })).toThrow(
            "above the unit ceiling of 250ms",
        );
        expect(() =>
            validateDeclaration("here", { claim: "slow", size: "integration", budget: 20001 }),
        ).toThrow("above the integration ceiling of 20000ms");
    },
);

check(
    "the declared budget is the runner timeout",
    {
        claim: "check() passes the declared budget to the runner, so a body that overruns it fails as a timeout",
        size: "integration",
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(root, ".surface-budget-"));
        const file = join(tree, "overrun.test.ts");
        writeFileSync(
            file,
            'import { check } from "@dylanebert/shallot/harness/check";\n' +
                'check("overruns", { claim: "overruns its budget", budget: 20 }, async () => {\n' +
                "    await Bun.sleep(2000);\n});\n",
        );
        try {
            const proc = Bun.spawnSync(["bun", "test", file], {
                cwd: root,
                env: { ...process.env, SHALLOT_UNIT_ONLY: "", SHALLOT_INTEGRATION_ONLY: "" },
            });
            expect(proc.exitCode).not.toBe(0);
            expect(proc.stderr.toString()).toContain("timed out after 20ms");
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "the public import path resolves",
    {
        claim: "check() is reachable at @dylanebert/shallot/harness/check, the path an extension imports",
    },
    () => {
        expect(typeof check).toBe("function");
    },
);
