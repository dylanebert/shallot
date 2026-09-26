import { expect } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
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
    {
        claim: "check() refuses tier and class fields plus unknown sizes and requirement tags",
        size: "integration",
        subject: ["src/harness/declaration.ts", "src/harness/check.ts"],
    },
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
                requires: ["gpu", "browser", "display", "cargo", "node"],
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
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-requirement-"));
        try {
            const file = join(tree, "missing.test.ts");
            writeFileSync(
                file,
                `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n` +
                    'check("missing", { claim: "missing display refuses", requires: ["display"] }, () => {});\n',
            );
            const environment = { ...process.env };
            delete environment.KEX_S3_ROW;
            const proc = Bun.spawnSync(["bun", "test", file], {
                cwd: resolve(import.meta.dir, "../.."),
                env: { ...environment, SHALLOT_UNIT_ONLY: "" },
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
        expect(() => validateDeclaration("here", { claim: "bad", requires: "gpu" })).toThrow(
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
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-budget-"));
        try {
            const file = join(tree, "overrun.test.ts");
            writeFileSync(
                file,
                `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n` +
                    'check("overruns", { claim: "overruns its budget", budget: 20 }, async () => {\n' +
                    "    await Bun.sleep(2000);\n});\n",
            );
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
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-row-selector-"));
        try {
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
    "the installed bin refuses unknown commands without resolving PATH helpers",
    {
        claim: "the installed shallot bin refuses an unknown verb with a shallot-verb executable on PATH and points to shallot --help without running it",
        size: "integration",
        subject: ["bin/shallot.ts", "src/cli/index.ts"],
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(tmpdir(), "shallot-unknown-command-"));
        const fixtureDir = join(tree, "bin");
        const marker = join(tree, "fixture-ran");
        try {
            mkdirSync(fixtureDir);
            const fixture = join(fixtureDir, "shallot-fixture");
            writeFileSync(fixture, `#!/bin/sh\nprintf ran > "$SHALLOT_FIXTURE_MARKER"\nexit 0\n`);
            chmodSync(fixture, 0o755);
            const proc = Bun.spawnSync(
                [process.execPath, resolve(root, "bin/shallot.ts"), "fixture"],
                {
                    cwd: root,
                    env: {
                        ...process.env,
                        PATH: `${fixtureDir}${delimiter}${process.env.PATH ?? ""}`,
                        SHALLOT_FIXTURE_MARKER: marker,
                    },
                    stdout: "pipe",
                    stderr: "pipe",
                },
            );
            expect(proc.exitCode).toBe(1);
            expect(proc.stdout.toString()).toBe("");
            expect(proc.stderr.toString().trim()).toBe(
                "unknown command: fixture\nSee `shallot --help` for available commands.",
            );
            expect(existsSync(marker)).toBe(false);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "a retired host declaration is refused before body or prerequisite",
    {
        claim: "an old host declaration is refused before its body or prerequisite runs, while ordinary admission ignores workstation identity variables",
        size: "integration",
        subject: ["src/harness/declaration.ts", "src/harness/check.ts"],
    },
    () => {
        expect(() => validateDeclaration("here", { claim: "retired", host: "mac" })).toThrow(
            "retired field `host`",
        );

        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-host-"));
        try {
            const reached = join(tree, "body-reached");
            const head =
                `import { appendFileSync } from "node:fs";\n` +
                `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n`;
            const retired = join(tree, "retired.test.ts");
            writeFileSync(
                retired,
                `${head}check("retired", { claim: "retired host declaration", size: "integration", host: "mac", requires: ["display"] }, () => {\n` +
                    `    appendFileSync(${JSON.stringify(reached)}, "reached");\n` +
                    `});\n`,
            );
            const ordinary = join(tree, "ordinary.test.ts");
            writeFileSync(
                ordinary,
                `${head}check("ordinary", { claim: "ordinary declaration" }, () => {\n` +
                    `    appendFileSync(${JSON.stringify(reached)}, "reached");\n` +
                    `});\n`,
            );
            const run = (file: string) => {
                const environment = { ...process.env };
                delete environment.KEX_S3_ROW;
                const proc = Bun.spawnSync(["bun", "test", "--pass-with-no-tests", file], {
                    cwd: root,
                    env: {
                        ...environment,
                        SHALLOT_HOST: "invented-seat",
                        HYPRLAND_INSTANCE_SIGNATURE: "invented-compositor",
                        SHALLOT_UNIT_ONLY: "",
                    },
                });
                return {
                    exitCode: proc.exitCode,
                    output: proc.stdout.toString() + proc.stderr.toString(),
                };
            };

            const refused = run(retired);
            expect(refused.exitCode).not.toBe(0);
            expect(refused.output).toContain("retired field `host`");
            expect(refused.output).not.toContain("display seat unavailable");
            expect(existsSync(reached)).toBe(false);

            const admitted = run(ordinary);
            expect(admitted.exitCode).toBe(0);
            expect(existsSync(reached)).toBe(true);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "a refusal is classified by the missing-premise type, never by its message",
    {
        claim: "check() reports refused only for a body throwing the harness's missing-premise type, and reports fail for every other throw whatever its message says, so a red claim cannot relabel itself as a premise the run never reached",
        size: "integration",
        subject: ["src/harness/check.ts", "src/harness/verdict.ts"],
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-refusal-"));
        try {
            const head =
                `import { check } from ${JSON.stringify(resolve(import.meta.dir, "check.ts"))};\n` +
                `import { MissingPremise } from ${JSON.stringify(resolve(import.meta.dir, "verdict.ts"))};\n`;
            const write = (name: string, claim: string, thrown: string) => {
                const file = join(tree, `${name}.test.ts`);
                writeFileSync(
                    file,
                    `${head}check(${JSON.stringify(name)}, { claim: ${JSON.stringify(claim)}, size: "integration" }, () => {\n` +
                        `    throw ${thrown};\n});\n`,
                );
                const environment = { ...process.env };
                delete environment.KEX_S3_ROW;
                const proc = Bun.spawnSync(["bun", "test", file], {
                    cwd: root,
                    env: { ...environment, SHALLOT_UNIT_ONLY: "", SHALLOT_INTEGRATION_ONLY: "" },
                });
                return {
                    exitCode: proc.exitCode,
                    output: proc.stdout.toString() + proc.stderr.toString(),
                };
            };
            // A body that fails its claim is a red, and stays one.
            const claimed = write(
                "claimed",
                "a body that fails its claim reports fail",
                'new Error("warm first-person page frames allocate:\\nafter warm 480: 9001 bytes")',
            );
            expect(claimed.exitCode).not.toBe(0);
            expect(claimed.output).toContain('"result":"fail"');
            expect(claimed.output).not.toContain('"result":"refused"');

            // The one class that is a refusal carries its reason, and still exits nonzero.
            const premise = write(
                "premise",
                "a body whose host cannot supply the premise reports refused",
                'new MissingPremise("the page presents at 59.6 Hz under the heap sampler")',
            );
            expect(premise.exitCode).not.toBe(0);
            expect(premise.output).toContain('"result":"refused"');
            expect(premise.output).toContain(
                '"reason":"the page presents at 59.6 Hz under the heap sampler"',
            );

            // The messages the prefix classifier used to promote: a page full of product exceptions and an
            // oracle's own sensitivity gate are both the claim failing, whatever their text.
            const threw = write(
                "threw",
                "a body reporting page errors reports fail",
                'new Error("inconclusive: the page threw:\\nTypeError: mesh is null")',
            );
            expect(threw.exitCode).not.toBe(0);
            expect(threw.output).toContain('"result":"fail"');
            expect(threw.output).not.toContain('"result":"refused"');

            const gate = write(
                "gate",
                "a body failing its own control gate reports fail",
                'new Error("inconclusive: the control read 12.0/f at frame src/engine/app/index.ts:558, not above the A/A window\'s 6400.0/f")',
            );
            expect(gate.exitCode).not.toBe(0);
            expect(gate.output).toContain('"result":"fail"');
            expect(gate.output).not.toContain('"result":"refused"');
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
