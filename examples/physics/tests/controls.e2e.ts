import { expect, test } from "playwright/test";
import { dragRange, gameKeyMovesCamera, pointerControl, tabSkipsChrome } from "./pointer";

test("pointer controls retain disabled nested settings and disclose the root without focus", async ({
    page,
}) => {
    await page.goto("/tests/controls.html");
    await page.waitForFunction(() => "controlsKey" in window);
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await pointerControl(page, picker);
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.getByRole("button", { name: "Wheel", exact: true }));
    await gameKeyMovesCamera(page);
    const panel = page.locator(".controls");
    const steering = panel.getByRole("checkbox", { name: "Steering", exact: true });
    const limit = panel.getByRole("checkbox", { name: "Steering Limit", exact: true });
    const minimum = panel.getByRole("slider", { name: "Min Degrees", exact: true });
    await pointerControl(page, steering);
    await pointerControl(page, limit);
    await gameKeyMovesCamera(page);
    await dragRange(page, minimum);
    await gameKeyMovesCamera(page);
    const retained = await minimum.inputValue();
    const before = await panel.boundingBox();
    await pointerControl(page, steering);
    await expect(limit).toBeDisabled();
    await expect(limit).toBeChecked();
    await expect(minimum).toBeDisabled();
    await expect(minimum).toHaveValue(retained);
    expect(await panel.boundingBox()).toEqual(before);
    const title = panel.locator("summary");
    await pointerControl(page, title);
    await expect(steering).toBeHidden();
    await gameKeyMovesCamera(page);
    await pointerControl(page, title);
    await expect(steering).toBeVisible();
    await tabSkipsChrome(page);
});
