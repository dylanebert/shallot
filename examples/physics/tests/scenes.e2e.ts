import { expect, test } from "playwright/test";

test("switching scenes resets the Bridge control and keeps the canvas", async ({ page }) => {
    // Frames are for a person judging the look: captured only on request, as hud's are, since a
    // software GPU cannot draw these scenes live within the test budget.
    async function capture(name: string) {
        const dir = process.env.SHALLOT_CAPTURE_DIR;
        if (!dir) return;
        // Show the stack landing and the bridge bent under gravity, not their initial poses.
        await page.waitForTimeout(name === "bridge" ? 6000 : 1000);
        await page.waitForFunction(() => {
            const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
            return (
                canvas.width === Math.round(canvas.clientWidth * devicePixelRatio) &&
                canvas.height === Math.round(canvas.clientHeight * devicePixelRatio)
            );
        });
        await page.evaluate(async () => {
            await (
                window as unknown as { capturePhysicsFrame: () => Promise<void> }
            ).capturePhysicsFrame();
        });
        await page.screenshot({ path: `${dir}/${name}.png` });
        await page.locator("#captured-frame").evaluate((element) => element.remove());
    }
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await page.addScriptTag({ type: "module", url: "/tests/capture-page.ts" });
    await page.waitForFunction(() => "capturePhysicsFrame" in window);
    const scene = page.getByRole("button", { name: "Scene", exact: true });
    await expect(scene).toBeEnabled();
    await expect(page.getByRole("slider")).toBeHidden();
    await capture("box-stack");
    await page.mouse.move(640, 300);
    await capture("hover");
    await page.mouse.move(20, 20);
    await scene.click();
    await page.mouse.click(20, 20);
    await expect(scene).toHaveAttribute("aria-expanded", "false");
    await scene.focus();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Escape");
    await expect(scene).toHaveAttribute("aria-expanded", "false");
    await expect(scene).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("button", { name: "Box Stack", exact: true })).toBeFocused();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Enter");
    await expect(scene).toHaveText("Bridge");
    await page.getByRole("button", { name: "Next scene" }).click();
    await expect(scene).toHaveText("Box Stack");
    for (const direction of ["Next", "Previous"]) {
        await page.getByRole("button", { name: `${direction} scene` }).click();
        await expect(scene).toHaveText("Bridge");
        await page.getByRole("button", { name: `${direction} scene` }).click();
        await expect(scene).toHaveText("Box Stack");
    }
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
    await capture("switcher");
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await expect(scene).toBeEnabled();
    const gravity = page.getByRole("slider", { name: "Gravity scale" });
    await expect(gravity).toHaveValue("1");
    await capture("bridge");
    await gravity.fill("-0.5");
    await scene.click();
    await page.getByRole("button", { name: "Box Stack", exact: true }).click();
    await expect(scene).toBeEnabled();
    await expect(gravity).toBeHidden();
    await capture("switch-to-stack");
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
    await capture("switcher");
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await expect(scene).toBeEnabled();
    await expect(gravity).toHaveValue("1");
    await expect(page.locator("canvas")).toBeVisible();
    expect(errors).toEqual([]);
});
