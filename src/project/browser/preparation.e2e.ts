import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);
test("a fixed-ratio auto-bound canvas stack prepares one blit without binding views early", async ({ page }) => {
    await page.goto('/motes-preparation.html?family=particles&samples=1&stack');
    await page.waitForFunction(() => 'preparedSamples' in window);
    const actual = await page.evaluate(() => {
        const control = window as unknown as { preparedSamples: number[]; preparedUpscaling: number; initialViews: number; dispose(): void };
        const result = { samples: control.preparedSamples, upscaling: control.preparedUpscaling, views: control.initialViews };
        control.dispose();
        return result;
    });
    expect(actual).toEqual({ samples: [1], upscaling: 1, views: 0 });
});
for (const family of ["particles", "volumetric-fog"]) {
    for (const samples of [0, 1, 4]) {
        test(`${family}: initial sample count ${samples} compiles only the reachable motes variant`, async ({ page }) => {
            await page.goto(`/motes-preparation.html?family=${family}&samples=${samples}`);
            await page.waitForFunction(() => "preparedSamples" in window);
            expect(await page.evaluate(() => (window as unknown as { preparedSamples: number[] }).preparedSamples))
                .toEqual(samples ? [samples] : []);
            expect(await page.evaluate(() => (window as unknown as { preparedUpscaling: number }).preparedUpscaling)).toBe(0);
            await page.evaluate(() => (window as unknown as { dispose(): void }).dispose());
        });
    }
}
