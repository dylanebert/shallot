import { expect, setDefaultTimeout, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";
import { nativeOutDir } from "../native";
import { runVite } from "../project/vite-command";
import { previewProject } from "./preview";

setDefaultTimeout(TIMEOUT.node);

test("native preview returns 143 when its executable terminates itself with SIGTERM", async () => {
    const project = mkdtempSync(join(tmpdir(), "shallot-preview-signal-"));
    try {
        const output = nativeOutDir(project, "linux", false, false);
        mkdirSync(output, { recursive: true });
        const executable = join(output, basename(project));
        writeFileSync(executable, "#!/bin/sh\nkill -TERM $$\n");
        chmodSync(executable, 0o755);
        expect(await previewProject(project, { target: "linux" })).toBe(143);
    } finally {
        removeTestDirectory(project);
    }
});

test("Vite returns 143 when its executable terminates itself with SIGTERM", () => {
    const project = mkdtempSync(join(tmpdir(), "shallot-vite-signal-"));
    try {
        const vite = join(project, "node_modules/vite");
        mkdirSync(vite, { recursive: true });
        writeFileSync(join(vite, "package.json"), JSON.stringify({ bin: "cli.js" }));
        writeFileSync(join(vite, "cli.js"), 'process.kill(process.pid, "SIGTERM");\n');
        expect(runVite(project, "build", [], "quiet")).toBe(143);
    } finally {
        removeTestDirectory(project);
    }
});
