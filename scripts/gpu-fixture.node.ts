import { expect, setDefaultTimeout, test } from "bun:test";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "./bun-child";

setDefaultTimeout(TIMEOUT.node);

function run(followed: boolean) {
    const child = Bun.spawnSync({
        ...bunChild(["test", join(import.meta.dir, "fixtures/device-error/device-error.ts")]),
        env: { ...process.env, SHALLOT_DEVICE_ERROR_FOLLOWED: followed ? "1" : "0" },
        stdout: "pipe",
        stderr: "pipe",
    });
    const output = `${child.stdout}${child.stderr}`;
    const verdict = (name: string) =>
        output.match(new RegExp(`^\\((pass|fail)\\) ${name}`, "m"))?.[1];
    return { exitCode: child.exitCode, output, verdict };
}

test("a GPU test that raises an undeclared device error fails itself, not the test after it", () => {
    const { exitCode, output, verdict } = run(true);
    expect(verdict("raises an undeclared device error"), output).toBe("fail");
    expect(verdict("follows the error"), output).toBe("pass");
    expect(exitCode, output).toBe(1);
});

test("a file's last GPU test that raises an undeclared device error fails itself", () => {
    const { exitCode, output, verdict } = run(false);
    expect(verdict("raises an undeclared device error"), output).toBe("fail");
    expect(exitCode, output).toBe(1);
});
