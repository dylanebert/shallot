import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
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
    const output = nativeOutDir(project, "linux", false, false);
    mkdirSync(output, { recursive: true });
    const observed = join(project, "observed-url.txt");
    const app = join(output, name);
    writeFileSync(app, `#!/bin/sh\nprintf '%s' "$SHALLOT_DEV_URL" > '${observed}'\n`);
    chmodSync(app, 0o755);
    try {
        const status = await previewProject(project, {
            target: "linux",
            devUrl: "http://localhost:5173/",
        });
        expect(status).toBe(0);
        expect(readFileSync(observed, "utf8")).toBe("http://localhost:5173/");
    } finally {
        rmSync(project, { recursive: true, force: true });
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
