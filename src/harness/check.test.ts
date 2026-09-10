import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { validateDeclaration } from "./declaration";

const BASE = { claim: "base", class: "pure", tier: "step", premises: [], budget: 10 };

check(
    "a declaration missing a field refuses",
    {
        claim: "check() refuses a declaration missing any of claim, class, tier, premises, budget",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 100,
    },
    () => {
        for (const field of ["claim", "class", "tier", "premises", "budget"]) {
            const partial: Record<string, unknown> = { ...BASE };
            delete partial[field];
            expect(() => validateDeclaration("here", partial)).toThrow(
                `invalid declaration: here is missing \`${field}\``,
            );
        }
        expect(() => validateDeclaration("here", BASE)).not.toThrow();
    },
);

check(
    "a declaration outside the class and tier vocabulary refuses",
    {
        claim: "check() refuses a class or tier outside the four classes and six tiers",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 100,
    },
    () => {
        expect(() => validateDeclaration("here", { ...BASE, class: "unit" })).toThrow(
            "has class `unit`",
        );
        expect(() => validateDeclaration("here", { ...BASE, tier: "smoke" })).toThrow(
            "has tier `smoke`",
        );
        for (const value of ["pure", "process", "seat", "oracle"]) {
            expect(() => validateDeclaration("here", { ...BASE, class: value })).not.toThrow();
        }
        for (const value of ["step", "gpu", "browser", "headed", "built", "live"]) {
            expect(() => validateDeclaration("here", { ...BASE, tier: value })).not.toThrow();
        }
    },
);

check(
    "a bad premises list or budget refuses",
    {
        claim: "check() refuses non-string premises and a budget that is not a positive finite number",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 100,
    },
    () => {
        expect(() => validateDeclaration("here", { ...BASE, premises: "cmake" })).toThrow(
            "array of strings",
        );
        expect(() => validateDeclaration("here", { ...BASE, premises: [1] })).toThrow(
            "array of strings",
        );
        for (const budget of [0, -1, Number.NaN, "10"]) {
            expect(() => validateDeclaration("here", { ...BASE, budget })).toThrow(
                "positive finite `budget`",
            );
        }
    },
);

check(
    "the declared budget is the runner timeout",
    {
        claim: "check() passes the declared budget to the runner, so a body that overruns it fails as a timeout",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(root, ".surface-budget-"));
        const file = join(tree, "overrun.test.ts");
        writeFileSync(
            file,
            'import { check } from "@dylanebert/shallot/harness/check";\n' +
                'check("overruns", { claim: "overruns its budget", class: "pure", tier: "step", premises: [], budget: 20 }, async () => {\n' +
                "    await Bun.sleep(2000);\n});\n",
        );
        try {
            const proc = Bun.spawnSync(["bun", "test", file], { cwd: root });
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
        class: "pure",
        tier: "step",
        premises: [],
        budget: 100,
    },
    () => {
        expect(typeof check).toBe("function");
    },
);
