import { expect, test } from "playwright/test";

test("switching scenes resets the Bridge control and keeps the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    const scene = page.getByRole("combobox", { name: "Scene" });
    await expect(scene).toBeEnabled();
    await expect(page.getByRole("slider")).toBeHidden();
    await scene.selectOption("bridge");
    await expect(scene).toBeEnabled();
    const gravity = page.getByRole("slider", { name: "Gravity scale" });
    await expect(gravity).toHaveValue("1");
    await gravity.fill("-0.5");
    await scene.selectOption("boxStack");
    await expect(scene).toBeEnabled();
    await expect(gravity).toBeHidden();
    await scene.selectOption("bridge");
    await expect(scene).toBeEnabled();
    await expect(gravity).toHaveValue("1");
    await expect(page.locator("canvas")).toBeVisible();
    expect(errors).toEqual([]);
});
