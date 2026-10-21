import { holdFrames } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "./pointer";

test("switching scenes resets the Bridge control and keeps the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    // Each switched scene steps and draws once; held frames keep the picker's clicks from waiting on draws.
    const frames = await holdFrames(page, "[aria-label='Scene navigation']");
    await page.goto("/");
    const scene = page.getByRole("button", { name: "Scene", exact: true });
    await expect(scene).toBeEnabled();
    const canvas = await page.locator("canvas").elementHandle();
    await expect(page.getByRole("slider")).toBeHidden();
    await scene.click();
    await page.mouse.click(20, 20);
    await expect(scene).toHaveAttribute("aria-expanded", "false");
    await bodyHasFocus(page);
    await pointerControl(page, scene);
    await page.keyboard.press("Escape");
    await expect(scene).toHaveAttribute("aria-expanded", "false");
    await bodyHasFocus(page);
    await pointerControl(page, scene);
    await bodyHasFocus(page);
    await expect(
        page
            .getByRole("region", { name: "Joints", exact: true })
            .getByRole("button", { name: "Bridge" }),
    ).toBeVisible();
    await expect(
        page
            .getByRole("region", { name: "Stacking", exact: true })
            .getByRole("button", { name: "Box Stack" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await expect(scene).toHaveText("Bridge");
    await expect(scene).toBeEnabled();
    await frames.run();
    const gravity = page.getByRole("slider", { name: "Gravity scale" });
    await expect(gravity).toHaveValue("1");
    await gravity.fill("-0.5");
    await scene.click();
    await page.getByRole("button", { name: "Box Stack", exact: true }).click();
    await expect(scene).toHaveText("Box Stack");
    await expect(scene).toBeEnabled();
    await frames.run();
    await expect(gravity).toBeHidden();
    await scene.click();
    await expect(
        page
            .getByRole("region", { name: "Joints", exact: true })
            .getByRole("button", { name: "Bridge" }),
    ).toBeVisible();
    await expect(
        page
            .getByRole("region", { name: "Stacking", exact: true })
            .getByRole("button", { name: "Box Stack" }),
    ).toBeVisible();
    await page.keyboard.press("Escape");
    await scene.click();
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await expect(scene).toHaveText("Bridge");
    await expect(scene).toBeEnabled();
    await frames.run();
    await expect(gravity).toHaveValue("1");
    await expect(page.locator("canvas")).toBeVisible();
    expect(
        await page.locator("canvas").evaluate((current, original) => current === original, canvas),
    ).toBe(true);
    expect(errors).toEqual([]);
});
