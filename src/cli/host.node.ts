import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { bunChild } from "../../scripts/bun-child";
import { removeTestDirectory } from "../../scripts/test-directory";

for (const command of ["dev", "build", "preview"]) {
    test(`${command} refuses linux on macOS before any build or launch`, () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-host-"));
        const marker = join(dir, "spawned");
        const entry = import.meta.resolve("./index.ts");
        try {
            const result = Bun.spawnSync({
                ...bunChild([
                    "-e",
                    `
                    import { main } from ${JSON.stringify(entry)};
                    import { appendFileSync } from "node:fs";
                    Object.defineProperty(process, "platform", { value: "darwin" });
                    Bun.spawn = Bun.spawnSync = (...args) => {
                        appendFileSync(${JSON.stringify(marker)}, JSON.stringify(args));
                        throw new Error("unexpected spawn");
                    };
                    await main([${JSON.stringify(command)}, "--target", "linux", "--portable", ${JSON.stringify(dir)}]);
                `,
                ]),
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(result.exitCode).toBe(1);
            expect(result.stderr.toString().trim()).toBe(
                "Target linux is not supported on host macOS; a linux target needs a Linux host.",
            );
            expect(result.stdout.toString()).toBe("");
            expect(() => readFileSync(marker)).toThrow();
        } finally {
            removeTestDirectory(dir);
        }
    });
}
