import { expect, test } from "playwright/test";

test("the panel renders dependent controls and checkbox input hides them", async ({ page }) => {
    await page.goto("/");
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await expect(picker).toBeEnabled();
    await picker.click();
    await page.getByRole("button", { name: "Wheel", exact: true }).click();
    await expect(picker).toBeEnabled();
    const panel = page.getByRole("region", { name: "Wheel controls", exact: true });
    const steering = panel.getByRole("checkbox", { name: "Steering", exact: true });
    const limit = panel.getByRole("checkbox", { name: "Steering Limit", exact: true });
    const minimum = panel.getByRole("slider", { name: "Min Degrees", exact: true });
    await steering.check();
    await expect(limit).toBeVisible();
    await limit.check();
    await expect(minimum).toBeVisible();
    await limit.uncheck();
    await expect(minimum).toBeHidden();
    await steering.uncheck();
    await expect(limit).toBeHidden();
    await picker.click();
    await page.getByRole("button", { name: "Box Stack", exact: true }).click();
    await expect(picker).toBeEnabled();
    await expect(panel).toBeHidden();
});
