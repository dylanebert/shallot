import type { Page } from "playwright/test";

export async function capture(
    page: Page,
    name: string,
    frames?: { run(): Promise<void> },
): Promise<void> {
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (!directory) throw new Error("Set SHALLOT_CAPTURE_DIR for the manual look review");
    await page.addScriptTag({ type: "module", url: "/tests/capture-page.ts" });
    await page.waitForFunction(() => "capturePhysicsFrame" in window);
    await page.waitForFunction(() => {
        const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
        return (
            canvas.width === Math.round(canvas.clientWidth * devicePixelRatio) &&
            canvas.height === Math.round(canvas.clientHeight * devicePixelRatio)
        );
    });
    const captured = page.evaluate(async () => {
        await (window as unknown as { capturePhysicsFrame(): Promise<void> }).capturePhysicsFrame();
    });
    if (frames) await frames.run();
    await captured;
    await page.screenshot({ path: `${directory}/${name}.png` });
    await page.locator("#captured-frame").evaluate((element) => element.remove());
}
