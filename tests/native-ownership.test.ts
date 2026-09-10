import { expect, test } from "bun:test";
import { resolve } from "node:path";

function run(mode: string, target: string, omit = "", mutation = "", shape = "nonempty") {
    const child = Bun.spawnSync(
        [process.execPath, resolve(import.meta.dir, "native-ownership.fixture.ts"), mode, target],
        {
            env: { ...process.env, OMIT: omit, MUTATE: mutation, ADAPTER_SHAPE: shape },
            timeout: 4000,
        },
    );
    return { exit: child.exitCode, output: child.stdout.toString() + child.stderr.toString() };
}
for (const mode of ["ownership", "gc"]) {
    for (const shape of ["nonempty", "undefined", "null"]) {
        test(`${mode} independently qualifies actual acquisition with ${shape} adapter options`, () => {
            const result = run(mode, "device", "", "", shape);
            expect(result.output).toContain("returns: 2");
            expect(result.output).toContain("pass: true");
            expect(result.exit).toBe(0);
        });
    }
}
test("real GC also pressures actual adapter entry", () => {
    const result = run("gc", "adapter");
    expect(result.output).toContain("returns: 2");
    expect(result.exit).toBe(0);
});
for (const [field, predicate] of [
    ["count", "feature count"],
    ["pointer", "feature pointer identity"],
    ["chain", "device outer chain"],
    ["feature", "u32 feature values"],
    ["size", "u64 buffer size"],
]) {
    for (const bits of ["low", "high"]) {
        test(`semantic full-width reader rejects ${field}-${bits} before forwarding`, () => {
            const result = run("layout", "device", "", `${field}-${bits}`);
            expect(result.output).toContain(predicate!);
            expect(result.output).toContain("returns: 1");
            expect(result.exit).toBe(1);
        });
    }
}
for (const site of ["adapter", "device"]) {
    const fields =
        site === "adapter"
            ? ["outer", "request-info"]
            : [
                  "outer",
                  "features",
                  "label",
                  "limits",
                  "inline-child",
                  "request-info",
                  "label-view",
                  "queue",
                  "queue-label-view",
                  "loss-info",
                  "error-info",
              ];
    for (const field of [...fields, "premature-release", "release-on-return"]) {
        test(`${site} removed ${field} fails independent ownership before native consumption`, () => {
            const result = run("ownership", site, field);
            expect(result.output).toContain(
                `${site}.${field === "premature-release" || field === "release-on-return" ? "outer" : field} ownership`,
            );
            expect(result.output).toContain(
                `returns: ${(site === "adapter" ? 0 : 1) + (field === "release-on-return" ? 1 : 0)}`,
            );
            expect(result.exit).toBe(1);
        });
    }
}
