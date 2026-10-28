import { mkdirSync } from "node:fs";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);
for (const claim of ["marker", "cube", "outside"] as const) {
    test(`${claim}: captured minimap pixels hold the arena claim`, async ({ page }) => {
        page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
        await page.goto("http://127.0.0.1:4197/tests/stepped.html");
        await page.waitForFunction(() => "evidence" in window);
        const result = await page.evaluate(
            (claim) =>
                (window as unknown as { evidence(c: string): Promise<number> }).evidence(claim),
            claim,
        );
        console.log(
            claim,
            result,
            await page.evaluate(() => (window as unknown as { adapter: unknown }).adapter),
        );
        if (claim === "outside") expect(result).toBe(0);
        else expect(result).toBeGreaterThan(20);
        if (process.env.SHALLOT_CAPTURE_DIR && claim === "cube") {
            mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
            await page.evaluate(() => (window as unknown as { frame(): Promise<void> }).frame());
            await page
                .locator("canvas")
                .screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/desktop.png` });
        }
    });
}
