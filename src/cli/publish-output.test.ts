import { expect, spyOn, test } from "bun:test";
import * as childProcess from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildWeb } from "../project/build";
import { runWrangler } from "./publish-cloudflare";
import { runButler } from "./publish-itch";

for (const status of [0, 1]) {
    test(`publish build output is ${status ? "complete on failure" : "quiet on success"}`, async () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-build-output-"));
        const vite = join(root, "node_modules/vite");
        mkdirSync(vite, { recursive: true });
        writeFileSync(join(vite, "package.json"), JSON.stringify({ bin: "cli.js" }));
        writeFileSync(
            join(vite, "cli.js"),
            `console.log('build stdout');console.error('build stderr');process.exit(${status});`,
        );
        const output: string[] = [];
        const stderr = spyOn(process.stderr, "write").mockImplementation((text) => {
            output.push(String(text));
            return true;
        });
        try {
            if (status)
                await expect(buildWeb(root, { stdout: "quiet" })).rejects.toThrow(
                    "vite build exited",
                );
            else await buildWeb(root, { stdout: "quiet" });
            expect(output.join("")).toBe(status ? "build stdout\nbuild stderr\n" : "");
        } finally {
            stderr.mockRestore();
            rmSync(root, { recursive: true, force: true });
        }
    });
}

test("butler failure shows all provider output, not only its last error", () => {
    const output: string[] = [];
    const stderr = spyOn(process.stderr, "write").mockImplementation((text) => {
        output.push(String(text));
        return true;
    });
    const spawn = spyOn(childProcess, "spawnSync").mockReturnValue({
        status: 1,
        signal: null,
        pid: 1,
        output: [],
        stdout: '{"type":"error","message":"provider failure"}\nprovider detail\n',
        stderr: "provider stderr\n",
    });
    try {
        const result = runButler("fixture-butler", ["push"]);
        expect(result.status).toBe(1);
        expect(output.join("")).toContain("provider failure");
        expect(output.join("")).toContain("provider detail");
        expect(output.join("")).toContain("provider stderr");
    } finally {
        spawn.mockRestore();
        stderr.mockRestore();
    }
});

for (const status of [0, 1]) {
    test(`interactive Wrangler deploy keeps prompts and ${status ? "restores its full log on failure" : "quiets its report"}`, () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-wrangler-output-"));
        const outputPath = join(root, "result.jsonl");
        const output: string[] = [];
        const stderr = spyOn(process.stderr, "write").mockImplementation((text) => {
            output.push(String(text));
            return true;
        });
        writeFileSync(`${outputPath}.log`, "full provider deploy log");
        const spawn = spyOn(childProcess, "spawnSync").mockReturnValue({
            status,
            signal: null,
            pid: 1,
            output: [],
            stdout: "",
            stderr: "",
        });
        try {
            expect(runWrangler(root, "fixture", ["deploy"], outputPath, true).status).toBe(status);
            const options = spawn.mock.calls[0]![2]!;
            expect(options.stdio).toEqual(["inherit", 2, 2]);
            expect(options.env?.WRANGLER_LOG).toBe("warn");
            expect(options.env?.WRANGLER_LOG_SANITIZE).toBe("true");
            expect(options.env?.WRANGLER_LOG_PATH).toBe(`${outputPath}.log`);
            expect(output.join("")).toBe(status ? "full provider deploy log\n" : "");
        } finally {
            spawn.mockRestore();
            stderr.mockRestore();
            rmSync(root, { recursive: true, force: true });
        }
    });
}
