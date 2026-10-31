import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

test("GPU resampling matches a CPU box reference within two bytes upscale and one otherwise", async ({
    page,
}) => {
    await page.goto("/resample.html");
    await expect(page.locator("body")).toHaveText(
        "upscale, exact and downscale match CPU box reference",
    );
});

for (const resized of [false, true]) {
    test(`registered image ignores its embedded linear RGB profile${resized ? " when resized" : ""}`, async ({
        page,
    }) => {
        await page.goto(resized ? "/?resize" : "/");
        await expect(page.locator("body")).toHaveText("[64,128,192,255]");
    });
}
