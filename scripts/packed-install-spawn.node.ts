import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { bunCommand } from "./bun-child";
import { spawnPackedInstall } from "./packed-install-spawn";

setDefaultTimeout(CEILING.node);

const BOUND = 2000;

test("a stalled install is killed inside its bound and retains the last step", () => {
    const start = performance.now();
    expect(() =>
        spawnPackedInstall(
            bunCommand([
                "-e",
                'console.log("Resolving dependencies"); setInterval(() => {}, 1000)',
            ]),
            process.cwd(),
            "fake install",
            BOUND,
        ),
    ).toThrow(
        `fake install timed out after ${BOUND} ms (last 8192 bytes per stream):\nstdout:\nResolving dependencies\n`,
    );
    expect(performance.now() - start).toBeLessThan(CEILING.node / 2);
});

test("early stderr cannot displace the last stdout step in a timeout tail", () => {
    let failure = "";
    try {
        spawnPackedInstall(
            bunCommand([
                "-e",
                'process.stderr.write("early stderr:" + "x".repeat(17000)); console.log("LAST STEP package-last"); setInterval(() => {}, 1000)',
            ]),
            process.cwd(),
            "fake install",
            BOUND,
        );
    } catch (error) {
        failure = String(error);
    }
    expect(failure).toContain("stdout:\nLAST STEP package-last\n");
    expect(failure).toContain("stderr:\n");
    expect(failure).not.toContain("early stderr:");
});

test("a normal install returns captured stdout and stderr untouched", () => {
    expect(
        spawnPackedInstall(
            bunCommand(["-e", 'console.log("installed"); console.error("done")']),
            process.cwd(),
            "fake install",
            BOUND,
        ),
    ).toBe("installed\ndone\n");
});
