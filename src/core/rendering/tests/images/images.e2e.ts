import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

test("native registered image mips match the original texels' box average within one byte", async ({
    page,
}) => {
    await page.goto("/mips.html");
    await expect(page.locator("body")).toContainText('"maxError":');
    const { maxError, ...dimensions } = JSON.parse(await page.locator("body").innerText());
    expect(dimensions).toEqual({ width: 512, height: 512, mips: 10 });
    expect(maxError).toBeLessThanOrEqual(1);
});

test("late URL and Blob images match every warm color and data mip without unrelated allocations", async ({
    page,
}) => {
    await page.goto("/late.html");
    await expect(page.locator("body")).toHaveText(
        JSON.stringify({
            equal: true,
            loaded: { textures: 4, groups: 12 },
            steady: { textures: 0, groups: 0 },
            disposed: { textures: 0, groups: 0 },
        }),
    );
});

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
