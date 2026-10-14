import { setDefaultTimeout, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { buildProjectDeclaration } from "./project-declaration";

setDefaultTimeout(TIMEOUT.node);

test("published project-tooling declarations are current", () => {
    const root = resolve(import.meta.dir, "..");
    for (const entry of ["testing", "vite"] as const) {
        const built = resolve(root, `dist/${entry}.d.ts`);
        const fresh = buildProjectDeclaration(entry);
        if (!existsSync(built)) {
            throw new Error(
                `dist/${entry}.d.ts is missing; run \`bun run prepare\` to generate it.`,
            );
        }
        if (readFileSync(built, "utf8") !== fresh) {
            throw new Error(`dist/${entry}.d.ts is stale; run \`bun run prepare\` to refresh it.`);
        }
        console.log(`${`dist/${entry}.d.ts`} is fresh`);
    }
});
