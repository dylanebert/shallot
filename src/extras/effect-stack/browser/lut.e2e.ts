import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

for (const mixed of [true, false]) {
    test(`chromatic LUT resolves ${mixed ? "a small image registered after a 2048 image" : "a lone 1024 image"}`, async ({
        page,
    }) => {
        await page.goto(mixed ? "/?mixed" : "/");
        await expect(page.locator("body")).toHaveText(
            `LUT ${mixed ? 1 : 0} preserves the white input`,
        );
    });
}
