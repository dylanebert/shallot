import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import { Glob } from "bun";
import { packageTimeoutErrors, timeoutErrors } from "./timeout-policy";

setDefaultTimeout(CEILING.node);

test("test tiers and browser subjects follow the shared timeouts", async () => {
    const root = resolve(import.meta.dir, "..");
    const errors = packageTimeoutErrors(await Bun.file(resolve(root, "package.json")).text());
    for (const directory of ["src", "examples", "scripts", "diagnostics"]) {
        for (const path of new Glob(
            "**/{*.test.ts,*.node.ts,*.e2e.ts,playwright.config.ts}",
        ).scanSync(resolve(root, directory))) {
            errors.push(
                ...timeoutErrors(
                    `${directory}/${path}`,
                    await Bun.file(resolve(root, directory, path)).text(),
                ),
            );
        }
    }
    errors.push(
        ...timeoutErrors(
            "src/project/testing.js",
            await Bun.file(resolve(root, "src/project/testing.js")).text(),
        ),
    );
    expect(errors).toEqual([]);
});
