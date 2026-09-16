import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { runAdd } from "./add";
import { PROJECT_GITIGNORE } from "./add-fragments";

function recipes(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-"));
    mkdirSync(join(root, "examples/demo"), { recursive: true });
    writeFileSync(join(root, "examples/demo/shallot.json"), '{"kind":"recipe"}\n');
    return root;
}

check(
    "add writes the project ignore",
    { claim: "shallot add gives a copied recipe the canonical .gitignore" },
    async () => {
        const root = recipes();
        const log = console.log;
        console.log = () => {};
        try {
            const dest = join(root, "out");
            expect(
                await runAdd(["demo", dest], {
                    recipesDir: join(root, "examples"),
                    version: "0.0.0",
                }),
            ).toBe(0);
            expect(readFileSync(join(dest, ".gitignore"), "utf8")).toBe(PROJECT_GITIGNORE);
        } finally {
            console.log = log;
            rmSync(root, { recursive: true, force: true });
        }
    },
);
