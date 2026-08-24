import { describe, expect, test } from "bun:test";
import { type CliArgs, parseCliArgs } from "./cli";

describe("parseCliArgs", () => {
    test("routes verify/recipe before the shared parse, carrying the rest of argv untouched", () => {
        const v = parseCliArgs(["verify", "--dist", "--json"]);
        expect(v).toEqual({ kind: "delegate", cmd: "verify", rest: ["--dist", "--json"] });

        const r = parseCliArgs(["recipe", "joints", "dest"]);
        expect(r).toEqual({ kind: "delegate", cmd: "recipe", rest: ["joints", "dest"] });
    });

    test("bare invocation prints usage, exit 0; an unrecognized subcommand exits 1", () => {
        expect(parseCliArgs([])).toEqual({ kind: "usage", exitCode: 0 });
        expect(parseCliArgs(["nope"])).toEqual({ kind: "usage", exitCode: 1 });
    });

    test("--help / -h short-circuits to usage exit 0, even with other flags present", () => {
        expect(parseCliArgs(["dev", "--help"])).toEqual({ kind: "usage", exitCode: 0 });
        expect(parseCliArgs(["-h"])).toEqual({ kind: "usage", exitCode: 0 });
    });

    test("dev/build/run resolve dir + target/release/portable/port/strict-port", () => {
        const a = parseCliArgs([
            "build",
            "examples/x",
            "--target",
            "mac",
            "--release",
            "--portable",
        ]);
        expect(a).toEqual({
            kind: "run",
            subcmd: "build",
            dir: "examples/x",
            target: "mac",
            release: true,
            portable: true,
            port: undefined,
            strictPort: false,
        });
    });

    test("bare dir defaults to '.'; --port and --port= both parse; --strict-port sets the flag", () => {
        const a = parseCliArgs(["dev", "--port", "5300", "--strict-port"]) as Extract<
            CliArgs,
            { kind: "run" }
        >;
        expect(a.dir).toBe(".");
        expect(a.port).toBe(5300);
        expect(a.strictPort).toBe(true);

        const b = parseCliArgs(["dev", "--port=4000"]) as Extract<CliArgs, { kind: "run" }>;
        expect(b.port).toBe(4000);
    });

    test("an unrecognized -flag throws rather than silently falling through to usage", () => {
        expect(() => parseCliArgs(["dev", "--nope"])).toThrow("unknown option: --nope");
    });

    // HOLE-RECORD ARM — `--port abc` should throw (verify.ts:112's own JSDoc states the policy: "a typo
    // must not silently no-op or flow NaN"), but cli.ts:83 `parseInt`'s `--port` with no validation, so
    // `--port abc` silently becomes NaN and flows to devConfig's port (dev.ts:65) into vite. The bare
    // `toThrow()` distinguishes a throw from a clean return, so a NaN-substituting default leaves it red —
    // but it CANNOT tell the validation throw from an unrelated one (the `unknown option` guard four lines
    // up throws too), so S2 tightens it to assert the message, as that sibling arm already does. S2 of this
    // spec (audit-cli-numeric-flags) is the unit that flips this arm green.
    test("--port abc throws instead of flowing NaN to vite — RED today (cli.ts:83 parseInt, no validation)", () => {
        expect(() => parseCliArgs(["dev", "--port", "abc"])).toThrow();
    });
});
