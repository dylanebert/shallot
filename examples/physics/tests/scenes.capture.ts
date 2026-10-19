import { expect, test } from "playwright/test";
import { capture } from "./capture";

test("stack, hover, picker and Bridge look review", async ({ page }) => {
    await page.goto("/");
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await expect(picker).toBeEnabled();
    await page.waitForTimeout(1000);
    await capture(page, "box-stack");
    await page.mouse.move(640, 300);
    await capture(page, "hover");
    await page.mouse.move(20, 20);
    await picker.click();
    await capture(page, "switcher");
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await expect(picker).toBeEnabled();
    await page.waitForTimeout(6000);
    await capture(page, "bridge");
    await picker.click();
    await page.getByRole("button", { name: "Box Stack", exact: true }).click();
    await expect(picker).toBeEnabled();
    await capture(page, "switch-to-stack");
});
