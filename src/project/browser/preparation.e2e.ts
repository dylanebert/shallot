import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);
test("a fixed-ratio auto-bound canvas stack prepares one blit without binding views early", async ({ page }) => {
    await page.goto('/motes-preparation.html?stack');
    await page.waitForFunction(() => 'preparedSamples' in window);
    const actual = await page.evaluate(() => {
        const control = window as unknown as { preparedSamples: number[]; preparedUpscaling: number; initialViews: number; dispose(): void };
        const result = { samples: control.preparedSamples, upscaling: control.preparedUpscaling, views: control.initialViews };
        control.dispose();
        return result;
    });
    expect(actual).toEqual({ samples: [1], upscaling: 1, views: 0 });
});
test("a lone fixed-ratio canvas camera prepares its motes variant and one upscaling pipeline", async ({ page }) => {
    await page.goto("/motes-preparation.html");
    await page.waitForFunction(() => "preparedSamples" in window);
    const actual = await page.evaluate(() => {
        const control = window as unknown as { preparedSamples: number[]; preparedUpscaling: number; dispose(): void };
        const result = { samples: control.preparedSamples, upscaling: control.preparedUpscaling };
        control.dispose();
        return result;
    });
    expect(actual).toEqual({ samples: [1], upscaling: 1 });
});
