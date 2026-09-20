import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "../..");
const BIN = resolve(ROOT, "bin/shallot.ts");

type Result = { code: number; stdout: string; stderr: string };

function cli(...args: string[]): Result {
    const result = Bun.spawnSync([process.execPath, BIN, ...args], {
        cwd: ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    return {
        code: result.exitCode ?? -1,
        stdout: result.stdout.toString(),
        stderr: result.stderr.toString(),
    };
}

function position(text: string, section: string): number {
    const index = text.indexOf(section);
    expect(index).toBeGreaterThanOrEqual(0);
    return index;
}

check(
    "top-level help leads with common use",
    {
        claim: "shallot top-level help presents purpose, usage, commands, and common examples before detailed options while retaining native build guidance",
    },
    () => {
        const result = cli("--help");
        expect(result.code).toBe(0);
        expect(result.stderr).toBe("");
        const help = result.stdout;
        expect(position(help, "shallot — run and build a shallot project")).toBeLessThan(
            position(help, "Usage"),
        );
        expect(position(help, "Usage")).toBeLessThan(position(help, "Commands"));
        expect(position(help, "Commands")).toBeLessThan(position(help, "Common examples"));
        expect(position(help, "Common examples")).toBeLessThan(position(help, "Options"));
        expect(help).toContain("shallot dev                  Run with hot reload");
        expect(help).toContain(
            "shallot add first-person     Copy the first-person recipe into ./first-person",
        );
        expect(help).toContain("source checkout");
        expect(help).toContain("set CEF_PATH");
        expect(help).toContain("Debug builds always compile from source.");
        expect(help).toContain("Required on Linux");
    },
);
