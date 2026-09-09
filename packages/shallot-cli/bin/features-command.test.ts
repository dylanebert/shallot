import { afterEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const dirs: string[] = [];
afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

// Only replace the native emit/launch leaf, in a fresh process. Keep CLI parsing, command dispatch,
// feature discovery, verdict and nativeOutDir real; the allowed arm writes to the caller's output.
for (const command of ["build", "run"]) {
    for (const [target, portable, allowed, gap, required] of [
        ["linux", false, false, false, false],
        ["linux", true, true, false, false],
        ["mac", false, true, false, false],
        ["windows", false, true, false, false],
        ["mac", false, false, true, true],
        ["mac", false, true, true, false],
        ["mac", true, true, true, true],
    ] as const) {
        test(`${command} ${target} portable=${portable} gap=${gap} required=${required}: ${allowed ? "emits" : "refuses before emit"}`, () => {
            const dir = mkdtempSync(join(tmpdir(), "shallot-floor-command-"));
            dirs.push(dir);
            writeFileSync(
                join(dir, "shallot.json"),
                JSON.stringify({ plugins: { Physics: true, Local: "external-floor-plugin" } }),
            );
            const plugin = join(dir, "node_modules/external-floor-plugin");
            mkdirSync(plugin, { recursive: true });
            writeFileSync(
                join(plugin, "package.json"),
                JSON.stringify({ name: "external-floor-plugin", main: "index.js" }),
            );
            writeFileSync(
                join(plugin, "index.js"),
                `
console.log("PLUGIN_LOADED");
export default { name: "Local", ${required ? "features" : "preferredFeatures"}: ["timestamp-query", "subgroups"] };
`,
            );
            const preload = join(dir, "preload.ts");
            const native = resolve(import.meta.dir, "native.ts");
            writeFileSync(
                preload,
                `
import { mock } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as native from ${JSON.stringify(native)};
import { WEBVIEW_UNSUPPORTED } from ${JSON.stringify(resolve(import.meta.dir, "features.ts"))};
${gap ? `WEBVIEW_UNSUPPORTED.mac = ["timestamp-query"];` : ""}
const emit = async (projectDir, outputDir, opts) => {
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(join(outputDir, "emitted.json"), JSON.stringify({ projectDir, opts }));
    console.log("NATIVE_EMIT");
    process.exit(0);
};
mock.module(${JSON.stringify(native)}, () => ({ ...native,
    bundleNativeLinux: emit, bundleNativeMac: emit, bundleNativeWindows: emit,
}));
`,
            );
            const result = Bun.spawnSync(
                [
                    process.execPath,
                    "--preload",
                    preload,
                    resolve(import.meta.dir, "cli.ts"),
                    command,
                    dir,
                    "--target",
                    target,
                    ...(portable ? ["--portable"] : []),
                ],
                { cwd: dir, env: process.env },
            );
            const output = result.stdout.toString() + result.stderr.toString();
            const artifact = join(
                dir,
                "build",
                target,
                `debug-${portable ? "portable" : "system"}`,
                "emitted.json",
            );
            if (allowed) {
                expect(output).toContain("NATIVE_EMIT");
                expect(result.exitCode).toBe(0);
                expect(JSON.parse(readFileSync(artifact, "utf8"))).toEqual({
                    projectDir: dir,
                    opts: { release: false, portable },
                });
            } else {
                expect(output).toContain("Cannot build");
                expect(output).toContain(gap ? "timestamp-query" : "WebGPU base floor");
                expect(output).toContain("--portable");
                expect(result.exitCode).toBe(1);
                expect(output).not.toContain("NATIVE_EMIT");
                expect(existsSync(artifact)).toBe(false);
                expect(existsSync(join(dir, "build"))).toBe(false);
                expect(existsSync(join(dir, "dist"))).toBe(false);
            }
            expect(output).toContain("PLUGIN_LOADED");
        });
    }
}
