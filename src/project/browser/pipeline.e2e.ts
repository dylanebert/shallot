import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

for (const create of ["newPage", "newContext"] as const) {
    test(`the shared warning fixture covers manual browser.${create}`, async ({ browser, baseURL }) => {
        await expect(pipelineWarnings.browser({ browser }, async (guarded) => {
            const page = create === "newPage" ? await guarded.newPage() : await (await guarded.newContext()).newPage();
            try {
                await page.goto(new URL("/pipeline.html", baseURL).href);
                await page.waitForFunction(() => "ready" in window);
            } finally {
                await page.context().close();
            }
        })).rejects.toThrow('Pipeline "late-control" owned by "LateOwner"');
    });
}

test("the shared browser fixture fails a late-pipeline control naming its pipeline and owner", async ({ page }) => {
    await expect(pipelineWarnings.page({ page }, async () => {
        await page.goto("/pipeline.html");
        await page.waitForFunction(() => "ready" in window);
    })).rejects.toThrow('Pipeline "late-control" owned by "LateOwner"');
});
