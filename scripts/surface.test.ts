import { expect } from "bun:test";
import { cpSync, mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const FIXTURES = resolve(ROOT, "scripts/fixtures/surface");

// Fixture check files are stored with a trailing `.fixture` so the real discovery and the real
// runner never see them; materializing strips it, giving the production readers a real tree.
function unfixture(dir: string): void {
    for (const entry of readdirSync(dir)) {
        const path = join(dir, entry);
        if (statSync(path).isDirectory()) {
            unfixture(path);
        } else if (entry.endsWith(".fixture")) {
            renameSync(path, path.slice(0, -".fixture".length));
        }
    }
}

function seed(name: string): string {
    const tree = mkdtempSync(join(tmpdir(), `shallot-surface-${name}-`));
    cpSync(resolve(FIXTURES, name), tree, { recursive: true });
    for (const dir of ["src", "scripts", "examples"])
        mkdirSync(join(tree, dir), { recursive: true });
    unfixture(tree);
    return tree;
}

function run(script: string, tree: string): { code: number; out: string; err: string } {
    const proc = Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts", script), "--list", "--root", tree],
        {
            cwd: ROOT,
        },
    );
    return {
        code: proc.exitCode ?? -1,
        out: proc.stdout.toString().trim(),
        err: proc.stderr.toString().trim(),
    };
}

function reader(name: string): { code: number; out: string; err: string } {
    const tree = seed(name);
    try {
        return run("check-surface.ts", tree);
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
}

check(
    "--list prints exactly the declared population",
    {
        claim: "surface.ts --list prints one row per declared check in the tree, from files and manifests, and nothing else",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const tree = seed("clean");
        try {
            const { code, out } = run("surface.ts", tree);
            expect(code).toBe(0);
            expect(out.split("\n")).toEqual([
                "claim             class    tier     premises    budget   file",
                "alpha holds       pure     step     -           50ms     src/alpha.test.ts",
                "alpha refuses     pure     step     cmake       200ms    src/alpha.test.ts",
                "beta builds       process  built    -           30000ms  scripts/beta.tier.ts",
                "demo recipe runs  process  browser  playwright  20000ms  examples/demo/check.test.ts",
                "4 checks",
            ]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "an undeclared check file reds the reader",
    {
        claim: "check-surface.ts reds on a test-suffix file that registers no check() declaration",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const { code, err } = reader("undeclared");
        expect(code).toBe(1);
        expect(err).toContain(
            "undeclared check file: src/naked.test.ts registers no check() declaration",
        );
    },
);

check(
    "a duplicate claim reds the reader",
    {
        claim: "check-surface.ts reds when two checks declare the same claim, naming both files",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const { code, err } = reader("duplicate");
        expect(code).toBe(1);
        expect(err).toContain('duplicate claim: "same claim" declared in');
        expect(err).toContain("src/one.test.ts");
        expect(err).toContain("src/two.test.ts");
    },
);

check(
    "an over-budget step declaration reds the reader",
    {
        claim: "check-surface.ts reds on a step-tier declaration whose budget is above the 1000 ms ceiling",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const { code, err } = reader("over-budget");
        expect(code).toBe(1);
        expect(err).toContain(
            'over-budget declaration: "slow step" in src/slow.test.ts declares 1500ms above the step ceiling of 1000ms',
        );
    },
);

check(
    "an orphan quarantine row reds the reader",
    {
        claim: "check-surface.ts reds when quarantine.json names a claim no check declares, and treats an absent file as zero rows",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const orphan = reader("orphan");
        expect(orphan.code).toBe(1);
        expect(orphan.err).toContain(
            'orphan quarantine row: "claim nobody declares" names no check in the population',
        );
        const absent = reader("clean");
        expect(absent.code).toBe(0);
        expect(absent.err).toBe("");
    },
);

check(
    "the reader passes the shipped tree",
    {
        claim: "check-surface.ts is green on the engine's own tree, so the population is never an empty scan",
        class: "process",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const proc = Bun.spawnSync(["bun", resolve(ROOT, "scripts/check-surface.ts")], {
            cwd: ROOT,
        });
        expect(proc.stderr.toString().trim()).toBe("");
        expect(proc.exitCode).toBe(0);
        expect(proc.stdout.toString()).toMatch(/^\d+ declared checks/);
        expect(Number(proc.stdout.toString().split(" ")[0])).toBeGreaterThan(0);
    },
);
