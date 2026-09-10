import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const DECLARED = "a".repeat(40);
const RECORDED = "b".repeat(40);

function seed(recorded: string): string {
    const tree = mkdtempSync(join(Bun.env.TMPDIR ?? "/tmp", "shallot-reference-pin-"));
    mkdirSync(join(tree, "crates/physics"), { recursive: true });
    mkdirSync(join(tree, "src/standard/physics/solver/fixtures"), { recursive: true });
    writeFileSync(
        join(tree, "crates/physics/reference.json"),
        `${JSON.stringify({ url: "https://example.test/box3d", branch: "harness", commit: DECLARED })}\n`,
    );
    writeFileSync(
        join(tree, "src/standard/physics/solver/fixtures/reference-pin.json"),
        `${JSON.stringify({ commit: recorded })}\n`,
    );
    return tree;
}

function run(root: string): { code: number; out: string; err: string } {
    const proc = Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts/check-reference-pin.ts"), "--root", root],
        { cwd: ROOT },
    );
    return {
        code: proc.exitCode ?? -1,
        out: proc.stdout.toString().trim(),
        err: proc.stderr.toString().trim(),
    };
}

check(
    "the shipped solver fixtures carry the declared Box3D commit",
    {
        claim: "check-reference-pin.ts passes only when the committed Box3D reference commit equals the solver fixture generator record",
        tier: "built",
    },
    () => {
        const result = run(ROOT);
        expect(result.code).toBe(0);
        expect(result.err).toBe("");
        expect(result.out).toContain("reference pin");
    },
);

check(
    "editing the reference commit reds the freshness reader",
    {
        claim: "check-reference-pin.ts reds when reference.json is changed without regenerating the solver fixtures",
        tier: "built",
    },
    () => {
        const tree = seed(RECORDED);
        try {
            const result = run(tree);
            expect(result.code).toBe(1);
            expect(result.err).toContain("reference pin mismatch");
            expect(result.err).toContain(DECLARED);
            expect(result.err).toContain(RECORDED);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
