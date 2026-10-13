import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { chromium } from "playwright/test";
import { assertChromiumInstalled } from "./browser-install";

const root = resolve(import.meta.dirname, "..");

export default function browserSetup(): void {
    assertChromiumInstalled(chromium.executablePath());
    // Build the page-side capture helper so the same Playwright run works in a copy and in the engine repo.
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
