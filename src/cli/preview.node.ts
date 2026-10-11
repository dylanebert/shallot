import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    linuxPreviewEnv,
    previewEnvironment,
    resolvePreviewTarget,
    windowsPreviewCommand,
} from "./preview";

test("web preview belongs to Vite while native preview resolves its native shell target", () => {
    expect(resolvePreviewTarget("mac")).toEqual({ kind: "mac" });
    expect(resolvePreviewTarget("linux")).toEqual({ kind: "linux" });
    expect(resolvePreviewTarget("windows")).toEqual({ kind: "windows" });
    expect(resolvePreviewTarget("web")).toEqual({ kind: "unknown", target: "web" });
});

test("a native debug launch receives the Vite URL instead of a packaged asset URL", () => {
    expect(
        previewEnvironment("http://localhost:5173/", { SHALLOT_DEV_URL: "asset://game" }),
    ).toEqual({ SHALLOT_DEV_URL: "http://localhost:5173/" });
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
