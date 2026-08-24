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

    // `--port abc` throws with a message naming the flag, so a typo doesn't flow NaN into vite's port
    // (the policy in verify.ts's `--port` JSDoc: "a typo must not silently no-op or flow NaN"). The
    // message assertion distinguishes this validation throw from the `unknown option` guard above.
    test("--port abc throws with a message naming the flag instead of flowing NaN to vite", () => {
        expect(() => parseCliArgs(["dev", "--port", "abc"])).toThrow('invalid --port value "abc"');
    });

    test("an empty --port value is rejected as empty, not silently coerced to 0", () => {
        expect(() => parseCliArgs(["dev", "--port="])).toThrow(
            'invalid --port value "" — must not be empty',
        );
    });

    test("a whitespace-only --port value is rejected, not silently coerced to 0", () => {
        expect(() => parseCliArgs(["dev", "--port", " "])).toThrow(
            'invalid --port value " " — must not be empty',
        );
    });

    test("--port 8080abc is rejected as non-numeric, not truncated to 8080", () => {
        expect(() => parseCliArgs(["dev", "--port", "8080abc"])).toThrow(
            'invalid --port value "8080abc"',
        );
    });

    test("--port 8080.5 is rejected as a non-integer, not passed to vite as a float", () => {
        expect(() => parseCliArgs(["dev", "--port", "8080.5"])).toThrow(
            'invalid --port value "8080.5" — expected an integer',
        );
    });
});
