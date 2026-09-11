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
        size: "integration",
    },
    () => {
        const tree = seed("clean");
        try {
            const { code, out } = run("surface.ts", tree);
            expect(code).toBe(0);
            expect(out.split("\n")).toEqual([
                "claim             size         requires  budget   file",
                "alpha holds       unit         -         250ms    src/alpha.test.ts",
                "alpha refuses     unit         -         250ms    src/alpha.test.ts",
                "beta builds       integration  -         20000ms  scripts/beta.tier.ts",
                "demo recipe runs  integration  chromium  20000ms  examples/demo/check.test.ts",
                "4 checks (parsed 4; 0 quarantined)",
            ]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
        const defaults = seed("defaults");
        try {
            const { code, out } = run("surface.ts", defaults);
            expect(code).toBe(0);
            expect(out).toContain("browser defaults");
            expect(out).toContain("integration");
            expect(out).toContain("chromium");
            expect(out).toContain("20000ms");
        } finally {
            rmSync(defaults, { recursive: true, force: true });
        }
    },
);

check(
    "an undeclared check file reds the reader",
    {
        claim: "check-surface.ts reds on a test-suffix file that registers no check() declaration",
        size: "integration",
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
        size: "integration",
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
    "an over-budget unit declaration reds the reader",
    {
        claim: "check-surface.ts reds on a unit declaration whose budget is above the 250 ms ceiling",
        size: "integration",
    },
    () => {
        const { code, err } = reader("over-budget");
        expect(code).toBe(1);
        expect(err).toContain(
            'invalid declaration: src/slow.test.ts check("slow unit") budget 251ms is above the unit ceiling of 250ms',
        );
    },
);

check(
    "an orphan quarantine row reds the reader",
    {
        claim: "check-surface.ts reds when quarantine.json names a claim no check declares, and treats an absent file as zero rows",
        size: "integration",
    },
    () => {
        const orphan = reader("orphan");
        expect(orphan.code).toBe(1);
        expect(orphan.err).toContain(
            'orphan quarantine row: claim "claim nobody declares" names no check in the population',
        );
        const absent = reader("clean");
        expect(absent.code).toBe(0);
        expect(absent.err).toBe("");
    },
);

check(
    "a non-literal declaration reds the reader",
    {
        claim: "check-surface.ts reds a check whose options use a spread, identifier or computed value, naming its file",
        size: "integration",
    },
    () => {
        const nonLiteral = reader("non-literal");
        expect(nonLiteral.code).toBe(1);
        expect(nonLiteral.err).toContain(
            'non-literal declaration: src/spread.test.ts check("spread declaration")',
        );
        expect(nonLiteral.err).toContain("src/identifier.test.ts");
        expect(nonLiteral.err).toContain("src/computed.test.ts");
    },
);

check(
    "an expired quarantine row reds the reader",
    {
        claim: "check-surface.ts reds a quarantine row whose ISO expiry is in the past",
        size: "integration",
    },
    () => {
        const expired = reader("expired");
        expect(expired.code).toBe(1);
        expect(expired.err).toContain('expired quarantine row: "expired claim" expired 2020-01-01');
    },
);

check(
    "a recipe manifest must name a file that declares a check",
    {
        claim: "check-surface.ts reds when a recipe manifest names a file with no check declaration",
        size: "integration",
    },
    () => {
        const missing = reader("manifest-no-check");
        expect(missing.code).toBe(1);
        expect(missing.err).toContain("undeclared check file: examples/no-check/check.test.ts");
    },
);

check(
    "the reader passes the shipped tree",
    {
        claim: "check-surface.ts is green on the engine's own tree, so the population is never an empty scan",
        size: "integration",
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
