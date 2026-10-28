import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { pointerControl } from "../pointer";

test.use(pipelineWarnings);

for (const view of [
    { name: "desktop", width: 1280, height: 720, hasTouch: false, hint: true },
    { name: "phone", width: 390, height: 844, hasTouch: false, hint: false },
    { name: "coarse pointer", width: 1280, height: 720, hasTouch: true, hint: false },
]) {
    test.describe(view.name, () => {
        test.use({
            viewport: { width: view.width, height: view.height },
            hasTouch: view.hasTouch,
        });
        for (const scene of ["Contact", "Driving", "Geometric Mover"]) {
            test(`${scene} earns a key hint only on a desktop fine pointer`, async ({ page }) => {
                await page.goto("/tests/controls.html");
                await page.waitForFunction(() => "controlsKey" in window);
                await pointerControl(
                    page,
                    page.getByRole("button", { name: "Scene", exact: true }),
                );
                await pointerControl(page, page.getByRole("button", { name: scene, exact: true }));
                const hint = page.locator(".input-help");
                if (view.hint) await expect(hint).toBeVisible();
                else await expect(hint).toBeHidden();
                const panel = page.locator(".controls");
                await expect(panel).toBeVisible();
                expect(
                    await panel.evaluate((element) => {
                        const summary = element.querySelector("summary")!;
                        const r = summary.getBoundingClientRect();
                        return summary.contains(
                            document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2),
                        );
                    }),
                ).toBe(true);
                const summary = panel.locator("summary");
                await pointerControl(page, summary);
                await expect(panel).not.toHaveAttribute("open", "");
                await pointerControl(page, summary);
                await expect(panel).toHaveAttribute("open", "");
            });
        }
    });
}
