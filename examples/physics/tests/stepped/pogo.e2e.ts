import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);

for (const scene of ["village", "geometricMover"]) {
    test(`${scene}'s pogo support ray is visible at desktop and phone sizes`, async ({
        page,
    }, info) => {
        await page.goto("/tests/controls.html?capture");
        await page.waitForFunction(() => "prepareMoverCapture" in window);
        await page.evaluate(
            (scene) =>
                (
                    window as unknown as { prepareMoverCapture(scene: string): Promise<void> }
                ).prepareMoverCapture(scene),
            scene,
        );
        for (const width of [1280, 390]) {
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            const result = await page.evaluate(() =>
                (
                    window as unknown as {
                        drawMoverCapture(): Promise<{
                            onGround: boolean;
                            ray: { origin: { y: number }; end: { y: number } };
                            bottom: number;
                        }>;
                    }
                ).drawMoverCapture(),
            );
            expect(result.onGround).toBe(true);
            expect(result.bottom - result.ray.end.y).toBeGreaterThan(0.55);
            expect(result.bottom - result.ray.end.y).toBeLessThan(0.65);
            await page.screenshot({
                path: info.outputPath(`${scene}-${width}${width === 390 ? "x844" : "x720"}.png`),
            });
        }
    });
}
