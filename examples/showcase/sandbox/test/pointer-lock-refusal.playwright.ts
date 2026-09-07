import { isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { expect, test } from "@playwright/test";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// The desktop half of the "the gun can't aim here" surface. Pointer Lock is not universal on a mouse
// device either: a browser can omit `requestPointerLock` entirely, or reject the request (sandboxed
// frame, missing user gesture, an exit too recent). Both are stubbed at the prototype before the app
// loads, so the real engine code takes its real branch — the capability probe in `standard/player` and
// the notice `src/sandbox.ts` mounts off `pointerLockStatus()`. The assertion pair is the visible
// notice text plus an empty page-error log: the old unguarded `requestPointerLock().catch()` threw a
// TypeError inside the click listener (missing method) and swallowed the rejection silently.
// Runs by path — `cd examples/showcase/sandbox && bunx playwright test test/pointer-lock-refusal.playwright.ts`.

const REASON = "test refusal: no user gesture";

async function boot(page: import("@playwright/test").Page): Promise<string[]> {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        // console.warn is the engine's own latched refusal notice here, not a failure
        if (m.type() === "error" || isDegradedBootMessage(m.text())) {
            errors.push(`[console.${m.type()}] ${m.text()}`);
        }
    });
    await page.goto("/");
    const adapter = await adapterName(page);
    console.log(`sandbox pointer-lock refusal adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await page.waitForFunction(() => {
        const c = document.querySelector("canvas");
        return c instanceof HTMLCanvasElement && c.width > 0 && c.height > 0;
    });
    return errors;
}

test("sandbox showcase — a browser without requestPointerLock explains itself", async ({
    page,
}) => {
    await page.addInitScript(() => {
        // the touch-only/WebView shape: the method simply isn't there
        delete (HTMLCanvasElement.prototype as { requestPointerLock?: unknown }).requestPointerLock;
    });
    const errors = await boot(page);
    await page
        .locator("canvas")
        .first()
        .click({ position: { x: 40, y: 40 } });
    await expect(page.locator(".sandbox-touch-notice")).toContainText("Pointer lock unavailable");
    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});

test("sandbox showcase — a refused pointer-lock request explains itself", async ({ page }) => {
    await page.addInitScript((reason: string) => {
        (
            HTMLCanvasElement.prototype as unknown as { requestPointerLock: () => Promise<void> }
        ).requestPointerLock = () => Promise.reject(new Error(reason));
    }, REASON);
    const errors = await boot(page);
    await page
        .locator("canvas")
        .first()
        .click({ position: { x: 40, y: 40 } });
    await expect(page.locator(".sandbox-touch-notice")).toContainText("Pointer lock refused");
    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});
