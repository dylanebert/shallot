import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

for (const multiple of [false, true]) {
    test(`registered image ignores its embedded linear RGB profile${multiple ? " alongside larger and data images" : ""}`, async ({
        page,
    }) => {
        await page.goto(multiple ? "/?multiple" : "/");
        await expect(page.locator("body")).toHaveText("[64,128,192,255]");
        const color = [{ width: 1, height: 1, format: "rgba8unorm-srgb", mips: 1 }];
        const data: object[] = [];
        if (multiple) {
            color.push({ width: 2, height: 2, format: "rgba8unorm-srgb", mips: 2 });
            data.push({ width: 1, height: 1, format: "rgba8unorm", mips: 1 });
            await expect(page.locator("body")).toHaveAttribute("data-shared-sampler", "true");
        }
        await expect(page.locator("body")).toHaveAttribute(
            "data-native",
            JSON.stringify([color, data]),
        );
    });
}
