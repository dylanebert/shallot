import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { BUN_CHILD, bunCommand } from "../../scripts/bun-child";

setDefaultTimeout(CEILING.node);

import {
    chmodSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { nativeOutDir } from "../native";
import {
    linuxPreviewEnv,
    previewProject,
    resolvePreviewTarget,
    windowsPreviewCommand,
} from "./preview";

test("web preview belongs to Vite while native preview resolves its native shell target", () => {
    expect(resolvePreviewTarget("mac")).toEqual({ kind: "mac" });
    expect(resolvePreviewTarget("linux")).toEqual({ kind: "linux" });
    expect(resolvePreviewTarget("windows")).toEqual({ kind: "windows" });
    expect(resolvePreviewTarget("web")).toEqual({ kind: "unknown", target: "web" });
});

test("a native debug process receives the Vite URL instead of its packaged asset URL", async () => {
    const project = mkdtempSync(join(tmpdir(), "shallot-native-dev-url-"));
    const name = basename(project);
    // The host's own target: a Windows host launches an .exe, others a Linux executable script.
    const windows = process.platform === "win32";
    const output = nativeOutDir(project, windows ? "windows" : "linux", false, false, true);
    mkdirSync(output, { recursive: true });
    const observed = join(project, "observed-url.txt");
    if (windows) {
        const source = join(project, "app.ts");
        writeFileSync(
            source,
            `require("node:fs").writeFileSync(${JSON.stringify(observed)}, process.env.SHALLOT_DEV_URL ?? "");\n`,
        );
        const compiled = Bun.spawnSync(
            bunCommand(["build", "--compile", source, "--outfile", join(output, `${name}.exe`)]),
            { ...BUN_CHILD, stdout: "pipe", stderr: "pipe" },
        );
        expect(compiled.exitCode, compiled.stderr.toString()).toBe(0);
    } else {
        const app = join(output, name);
        writeFileSync(app, `#!/bin/sh\nprintf '%s' "$SHALLOT_DEV_URL" > '${observed}'\n`);
        chmodSync(app, 0o755);
    }
    try {
        const status = await previewProject(project, {
            target: windows ? "windows" : "linux",
            devUrl: "http://localhost:5173/",
        });
        expect(status).toBe(0);
        expect(readFileSync(observed, "utf8")).toBe("http://localhost:5173/");
    } finally {
        rmSync(project, { recursive: true, force: true, maxRetries: 10 });
    }
});

test("native preview retains its platform launch command and portable Linux library path", () => {
    expect(windowsPreviewCommand("C:\\game folder", "my-game")).toBe(
        "cd 'C:\\game folder'; .\\my-game.exe",
    );
    expect(
        linuxPreviewEnv("/games/app", true, { LD_LIBRARY_PATH: "/system" }).LD_LIBRARY_PATH,
    ).toBe("/games/app/cef:/system");
    expect(
        linuxPreviewEnv("/games/app", false, { LD_LIBRARY_PATH: "/system" }).LD_LIBRARY_PATH,
    ).toBe("/system");
});

test("native preview streams the shell's output while the shell runs", async () => {
    const project = mkdtempSync(join(tmpdir(), "shallot-native-preview-stream-"));
    const name = basename(project);
    const windows = process.platform === "win32";
    const output = nativeOutDir(project, windows ? "windows" : "linux", false, false, false);
    mkdirSync(output, { recursive: true });
    const release = join(project, "release");
    const source = join(project, "app.ts");
    // The shell prints its line, then waits for the test, so a buffered preview shows nothing until it exits.
    writeFileSync(
        source,
        `console.error("[shallot] asset server on http://127.0.0.1:1");\nwhile (!require("node:fs").existsSync(${JSON.stringify(release)})) await Bun.sleep(20);\n`,
    );
    const compiled = Bun.spawnSync(
        bunCommand([
            "build",
            "--compile",
            source,
            "--outfile",
            join(output, windows ? `${name}.exe` : name),
        ]),
        { ...BUN_CHILD, stdout: "pipe", stderr: "pipe" },
    );
    expect(compiled.exitCode, compiled.stderr.toString()).toBe(0);
    const preview = resolve(import.meta.dir, "preview.ts");
    const cli = Bun.spawn(
        bunCommand([
            "-e",
            `const { previewProject } = await import(${JSON.stringify(preview)}); process.exit(await previewProject(${JSON.stringify(project)}, { target: ${JSON.stringify(windows ? "windows" : "linux")} }));`,
        ]),
        { ...BUN_CHILD, stdout: "pipe", stderr: "pipe" },
    );
    try {
        let seen = "";
        const reader = cli.stderr.getReader();
        const deadline = Date.now() + CEILING.node / 2;
        while (!seen.includes("asset server on") && Date.now() < deadline) {
            const chunk = await Promise.race([
                reader.read(),
                Bun.sleep(deadline - Date.now()).then(() => undefined),
            ]);
            if (!chunk || chunk.done) break;
            seen += new TextDecoder().decode(chunk.value);
        }
        expect(existsSync(release)).toBe(false);
        expect(seen).toContain("[shallot] asset server on http://127.0.0.1:1");
        reader.releaseLock();
        writeFileSync(release, "");
        expect(await cli.exited).toBe(0);
    } finally {
        cli.kill();
        rmSync(project, { recursive: true, force: true, maxRetries: 10 });
    }
});
