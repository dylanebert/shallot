import { expect, test } from "@playwright/test";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// S5's showcase touch smoke (`shallot-mobile-controls` spec): the sandbox demo loads clean under a real
// `hasTouch` mobile context and shows the desktop-only notice (`src/ui.ts`'s `touchNotice`) — the gun
// aims through Pointer Lock, which has no touch equivalent (Out of scope: touch FPS scheme), so this demo
// carries no touch-interaction assertion, only the clean-load + notice pair. Runs by path —
// `cd examples/showcase/sandbox && bunx playwright test test/touch-smoke.playwright.ts` — display-gated
// (WSL bridge, `playwright.global-setup.ts`), never part of `bun run test`.

test.use({
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 },
});

test("sandbox showcase — loads clean and shows the desktop-only notice on touch", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(`[console.error] ${m.text()}`);
    });

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`sandbox touch smoke adapter: ${adapter || "none offered"}`);
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

    await expect(page.locator(".sandbox-touch-notice")).toContainText("Desktop only");

    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});
