import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { chromium } from "playwright/test";
import { assertChromiumInstalled } from "./browser-install";

const root = resolve(import.meta.dirname, "..");

export default function browserSetup(): void {
    assertChromiumInstalled(chromium.executablePath());
    // The page-side capture helper, bundled for addScriptTag; setup builds it so the same Playwright run
    // works in a copy and in the engine repo, where the example has no manifest of its own.
    execFileSync(
        "bun",
        [
            "build",
            "./tests/capture-page.ts",
            "--target=browser",
            "--format=iife",
            "--outfile=.artifacts/capture-page.js",
        ],
        { cwd: root, stdio: "inherit" },
    );
}
