import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

for (const resized of [false, true]) {
    test(`registered image ignores its embedded linear RGB profile${resized ? " when resized" : ""}`, async ({
        page,
    }) => {
        await page.goto(resized ? "/?resize" : "/");
        await expect(page.locator("body")).toHaveText("[64,128,192,255]");
    });
}
