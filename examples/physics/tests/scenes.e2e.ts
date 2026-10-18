import { expect, test } from "playwright/test";

test("switching scenes resets the Bridge control and keeps the canvas", async ({
    page,
}, testInfo) => {
    async function capture(name: string) {
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
                window as Window & { capturePhysicsFrame: () => Promise<void> }
            ).capturePhysicsFrame();
        });
        await page.screenshot({ path: testInfo.outputPath(`${name}.png`) });
        await page.locator("#captured-frame").evaluate((element) => element.remove());
    }
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await page.addScriptTag({ type: "module", url: "/tests/capture-page.ts" });
    await page.waitForFunction(() => "capturePhysicsFrame" in window);
    const scene = page.getByRole("combobox", { name: "Scene" });
    await expect(scene).toBeEnabled();
    await expect(page.getByRole("slider")).toBeHidden();
    await capture("box-stack");
    await scene.selectOption("bridge");
    await expect(scene).toBeEnabled();
    const gravity = page.getByRole("slider", { name: "Gravity scale" });
    await expect(gravity).toHaveValue("1");
    await capture("bridge");
    await gravity.fill("-0.5");
    await scene.selectOption("boxStack");
    await expect(scene).toBeEnabled();
    await expect(gravity).toBeHidden();
    await capture("switch-to-stack");
    await scene.selectOption("bridge");
    await expect(scene).toBeEnabled();
    await expect(gravity).toHaveValue("1");
    await expect(page.locator("canvas")).toBeVisible();
    expect(errors).toEqual([]);
});
