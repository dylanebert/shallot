import { expect, test } from "playwright/test";

test("keyboard controls retain disabled nested settings and disclose the root", async ({
    page,
}) => {
    await page.goto("/");
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await picker.focus();
    await page.keyboard.press("Enter");
    await page.getByRole("button", { name: "Wheel", exact: true }).focus();
    await page.keyboard.press("Enter");
    await expect(picker).toBeFocused();
    const panel = page.locator(".controls");
    const steering = panel.getByRole("checkbox", { name: "Steering", exact: true });
    const limit = panel.getByRole("checkbox", { name: "Steering Limit", exact: true });
    const minimum = panel.getByRole("slider", { name: "Min Degrees", exact: true });
    await steering.focus();
    await page.keyboard.press("Space");
    await limit.focus();
    await page.keyboard.press("Space");
    await minimum.focus();
    const value = Number(await minimum.inputValue());
    await page.keyboard.press("ArrowRight");
    await expect(minimum).toHaveValue(String(value + Number(await minimum.getAttribute("step"))));
    const retained = await minimum.inputValue();
    const before = await panel.boundingBox();
    await steering.focus();
    await page.keyboard.press("Space");
    await expect(limit).toBeDisabled();
    await expect(limit).toBeChecked();
    await expect(minimum).toBeDisabled();
    await expect(minimum).toHaveValue(retained);
    expect(await panel.boundingBox()).toEqual(before);
    const title = panel.locator("summary");
    await title.focus();
    await page.keyboard.press("Enter");
    await expect(steering).toBeHidden();
    await page.keyboard.press("Space");
    await expect(steering).toBeVisible();
});
