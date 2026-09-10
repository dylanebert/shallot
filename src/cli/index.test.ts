import { describe, expect, test } from "bun:test";
import { chmodSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { type CliArgs, parseCliArgs, resolveExternal } from "./index";

describe("parseCliArgs", () => {
    test("routes add and unknown verbs before the shared parse, carrying the rest of argv untouched", () => {
        const r = parseCliArgs(["add", "joints", "dest"]);
        expect(r).toEqual({ kind: "add", rest: ["joints", "dest"] });
        const x = parseCliArgs(["deploy", "--prod", "dest"]);
        expect(x).toEqual({ kind: "external", verb: "deploy", rest: ["--prod", "dest"] });
        expect(parseCliArgs(["create", "game"])).toEqual({ kind: "create" });
        expect(parseCliArgs(["check"])).toEqual({ kind: "check" });
    });

    test("bare invocation prints usage, exit 0", () => {
        expect(parseCliArgs([])).toEqual({ kind: "usage", exitCode: 0 });
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
            open: true,
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

    test("--no-open turns the dev server's browser launch off; the human default leaves it on", () => {
        // the tab a gate's web server used to open in the operator's own browser is this flag's job
        const opened = parseCliArgs(["dev"]) as Extract<CliArgs, { kind: "run" }>;
        expect(opened.open).toBe(true);

        const quiet = parseCliArgs(["dev", "--no-open", "--port", "5300"]) as Extract<
            CliArgs,
            { kind: "run" }
        >;
        expect(quiet.open).toBe(false);
        expect(quiet.port).toBe(5300);
    });

    test("an unrecognized -flag throws rather than silently falling through to usage", () => {
        expect(() => parseCliArgs(["dev", "--nope"])).toThrow("unknown option: --nope");
    });

    // `--port abc` throws with a message naming the flag, so a typo doesn't flow NaN into vite's port
    // (a typo must not silently no-op or flow NaN). The message assertion distinguishes this validation throw from the `unknown option` guard above.
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

test("an unknown verb resolves to shallot-<verb> on PATH and runs with the remaining args", () => {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "shallot-ext-")));
    const bin = join(dir, "shallot-hello");
    writeFileSync(bin, '#!/bin/sh\necho "hello:$*"\n');
    chmodSync(bin, 0o755);
    expect(resolveExternal("hello", { PATH: dir }, dir)).toBe(bin);
    expect(resolveExternal("absent", { PATH: dir }, dir)).toBeNull();
    expect(resolveExternal("../hello", { PATH: dir }, dir)).toBeNull();

    const cli = resolve(import.meta.dir, "../../bin/shallot.ts");
    const env = { ...process.env, PATH: `${dir}${delimiter}${process.env.PATH}` };
    const run = Bun.spawnSync(["bun", cli, "hello", "a", "--b"], { cwd: dir, env });
    expect(run.exitCode).toBe(0);
    expect(run.stdout.toString().trim()).toBe("hello:a --b");
    const miss = Bun.spawnSync(["bun", cli, "absent"], { cwd: dir, env });
    expect(miss.exitCode).toBe(1);
    expect(miss.stderr.toString()).toContain("unknown command: absent");
});
