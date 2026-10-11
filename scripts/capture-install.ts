import type { Page } from "playwright/test";

/**
 * Install the public `captureFrame` on a page and wait until it is callable: `addScriptTag` can return
 * before the module script it adds has run, so a capture evaluated at once can find no global
 * (loading-screen's "captureFrame is not installed" on macOS, PR 531 run 38174256358).
 */
export async function installCaptureFrame(page: Page, entry: string): Promise<void> {
    await page.addScriptTag({ path: entry, type: "module" });
    await page.waitForFunction(() => typeof window.shallotCaptureFrame === "function");
}
