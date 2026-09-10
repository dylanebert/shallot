import { expect, test } from "bun:test";
import { resolve } from "node:path";

for (const site of ["adapter", "device", "instance"]) {
    for (const mode of [
        "ownership",
        "gc",
        "omit",
        "delete-assertion",
        "native-failure",
        "decode-failure",
        ...(site === "instance" ? ["status-failure"] : []),
    ]) {
        test(`${site} real feature output: ${mode}`, () => {
            const child = Bun.spawnSync(
                [
                    process.execPath,
                    resolve(import.meta.dir, "native-features.fixture.ts"),
                    site,
                    mode,
                ],
                { timeout: 4000 },
            );
            const output = child.stdout.toString() + child.stderr.toString();
            if (mode === "omit" || mode === "delete-assertion") {
                expect(output).toContain(`${site}.missing-output-owner`);
                expect(output).toContain("forwarded: 0");
                if (mode === "delete-assertion")
                    expect(output).toContain("nonforwarding safety stop");
                expect(child.exitCode).toBe(1);
            } else {
                expect(output).toContain("FEATURE_OUTPUT_PASS");
                expect(child.exitCode).toBe(0);
            }
        });
    }
}
