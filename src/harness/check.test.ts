import { expect } from "bun:test";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
            validateDeclaration("here", {
                claim: "bad",
                requires: ["gpu", "display", "deploy", "cargo"],
            }),
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
                'check("missing", { claim: "missing display refuses", requires: ["display"] }, () => {});\n',
        );
        try {
            const proc = Bun.spawnSync(["bun", "test", file], {
                cwd: resolve(import.meta.dir, "../.."),
                env: { ...process.env, SHALLOT_UNIT_ONLY: "" },
            });
            expect(proc.exitCode).not.toBe(0);
            expect(proc.stderr.toString() + proc.stdout.toString()).toContain(
                "refused check missing display refuses",
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
            const environment = { ...process.env };
            delete environment.KEX_S3_ROW;
            const proc = Bun.spawnSync(["bun", "test", file], {
                cwd: root,
                env: { ...environment, SHALLOT_UNIT_ONLY: "", SHALLOT_INTEGRATION_ONLY: "" },
            });
            expect(proc.exitCode).not.toBe(0);
            expect(proc.stderr.toString()).toContain("timed out after 20ms");
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "an exact row selector skips every other declaration size",
    {
        claim: "KEX_S3_ROW selects exactly one claim and skips unrelated unit bodies",
        size: "integration",
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(root, ".surface-row-selector-"));
        const file = join(tree, "selector.test.ts");
        const sideEffect = join(tree, "unit-reached");
        const unitClaim = "unselected unit must not execute";
        const targetClaim = "selected integration executes with a real verdict";
        writeFileSync(
            file,
            `import { appendFileSync } from "node:fs";\n` +
                `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n` +
                `check("unit", { claim: ${JSON.stringify(unitClaim)} }, () => {\n` +
                `    appendFileSync(${JSON.stringify(sideEffect)}, "reached");\n` +
                `    throw new Error("the skipped unit body was reached");\n` +
                `});\n` +
                `check("target", { claim: ${JSON.stringify(targetClaim)}, size: "integration" }, () => ({ ok: true }));\n`,
        );
        try {
            const proc = Bun.spawnSync(
                ["bun", "test", "--max-concurrency=1", "--pass-with-no-tests", file],
                {
                    cwd: root,
                    env: {
                        ...process.env,
                        KEX_S3_ROW: targetClaim,
                        SHALLOT_UNIT_ONLY: "",
                        SHALLOT_INTEGRATION_ONLY: "",
                    },
                },
            );
            const output = proc.stdout.toString() + proc.stderr.toString();
            expect(proc.exitCode).toBe(0);
            expect(output).toContain(targetClaim);
            expect(output).toContain('"result":"pass"');
            expect(output.match(/shallot verdict/g)?.length).toBe(1);
            expect(output).toContain("1 pass");
            expect(output).toContain("1 skip");
            expect(output).not.toContain("no tests");
            expect(existsSync(sideEffect)).toBe(false);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "the installed bin refuses unknown commands",
    {
        claim: "the installed shallot bin exits non-zero instead of silently accepting an unknown command",
        size: "integration",
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const proc = Bun.spawnSync(["bun", resolve(root, "bin/shallot.ts"), "test:integration"], {
            cwd: root,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(proc.exitCode).toBe(1);
        expect(proc.stdout.toString() + proc.stderr.toString()).toContain(
            "unknown command: test:integration",
        );
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

check(
    "a row declared for another host is skipped and reported, never refused",
    {
        claim: "a row declared for one host refuses or runs on another, so a host that cannot hold its premise reports a failure against the claim",
        size: "integration",
        subject: ["src/harness/declaration.ts", "src/harness/check.ts"],
    },
    () => {
        expect(validateDeclaration("here", { claim: "seat", host: "mac" }).host).toBe("mac");
        expect(validateDeclaration("here", { claim: "seat" }).host).toBeUndefined();
        expect(() => validateDeclaration("here", { claim: "seat", host: "windows" })).toThrow(
            "has host `windows`",
        );

        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(root, ".surface-host-"));
        const reached = join(tree, "body-reached");
        // Two files, because a refused requirement throws at registration and would take the other row
        // down with it. `plain` carries no premise, so its body is the non-vacuity witness on any host;
        // `gated` carries one no host here supplies, so a mismatch must report `unrun` rather than that
        // requirement's refusal.
        const plainFile = join(tree, "plain.test.ts");
        const gatedFile = join(tree, "gated.test.ts");
        const head =
            `import { appendFileSync } from "node:fs";\n` +
            `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n`;
        writeFileSync(
            plainFile,
            `${head}check("plain", { claim: "a row declared for the omarchy seat", size: "integration", host: "omarchy" }, () => {\n` +
                `    appendFileSync(${JSON.stringify(reached)}, "reached");\n` +
                `    throw new Error("the other host's body ran here");\n` +
                `});\n`,
        );
        writeFileSync(
            gatedFile,
            `${head}check("gated", { claim: "an omarchy row whose requirement no host here supplies", size: "integration", host: "omarchy", requires: ["display"] }, () => {});\n`,
        );
        const run = (file: string, host: string) => {
            const environment = { ...process.env };
            delete environment.KEX_S3_ROW;
            const proc = Bun.spawnSync(["bun", "test", "--pass-with-no-tests", file], {
                cwd: root,
                env: { ...environment, SHALLOT_HOST: host, SHALLOT_UNIT_ONLY: "" },
            });
            return {
                exitCode: proc.exitCode,
                output: proc.stdout.toString() + proc.stderr.toString(),
            };
        };
        try {
            // Skipped and reported: the verdict names the declared host, the run stays green, and the
            // body never executes.
            const elsewhere = run(plainFile, "mac");
            expect(elsewhere.exitCode).toBe(0);
            expect(elsewhere.output).toContain('"result":"unrun"');
            expect(elsewhere.output).toContain("declared for host omarchy; this host is mac");
            expect(elsewhere.output).not.toContain('"result":"refused"');
            expect(existsSync(reached)).toBe(false);
            // Non-vacuity: on its own host the body runs and its failure is reported.
            const here = run(plainFile, "omarchy");
            expect(here.exitCode).not.toBe(0);
            expect(existsSync(reached)).toBe(true);
            expect(here.output).not.toContain('"result":"unrun"');
            // The mismatch resolves before requirements, so the unavailable premise is never probed on
            // the wrong host — and is still refused on the right one.
            const gatedElsewhere = run(gatedFile, "mac");
            expect(gatedElsewhere.exitCode).toBe(0);
            expect(gatedElsewhere.output).toContain('"result":"unrun"');
            expect(gatedElsewhere.output).not.toContain("display seat unavailable");
            const gatedHere = run(gatedFile, "omarchy");
            expect(gatedHere.exitCode).not.toBe(0);
            expect(gatedHere.output).toContain("display seat unavailable");
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
