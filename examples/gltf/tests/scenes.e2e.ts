import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

async function texturePixels(page: Page, blue = false, bright = false) {
    const png = await page.locator("canvas").screenshot();
    return page.evaluate(
        async ({ data, blue, bright }) => {
            const image = await createImageBitmap(
                await (await fetch(`data:image/png;base64,${data}`)).blob(),
            );
            const canvas = document.createElement("canvas");
            canvas.width = image.width;
            canvas.height = image.height;
            const context = canvas.getContext("2d")!;
            context.drawImage(image, 0, 0);
            image.close();
            const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
            let colored = 0;
            for (let i = 0; i < rgba.length; i += 4) {
                if (
                    bright
                        ? i < rgba.length / 2 && Math.max(rgba[i], rgba[i + 1], rgba[i + 2]) > 100
                        : blue
                          ? rgba[i + 2] > rgba[i] * 1.3 && rgba[i + 2] > rgba[i + 1] * 1.1
                          : rgba[i + 1] > rgba[i] * 1.3 && rgba[i + 1] > rgba[i + 2] * 1.3
                )
                    colored++;
            }
            return colored;
        },
        { data: png.toString("base64"), blue, bright },
    );
}

test.use(pipelineWarnings);
for (const phone of [false, true]) {
    test.describe(phone ? "phone" : "desktop", () => {
        test.use({
            viewport: phone ? { width: 390, height: 844 } : { width: 1280, height: 720 },
            hasTouch: phone,
        });
        test("the textured truck draws and a catalog switch draws on the same canvas", async ({
            page,
        }, info) => {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            await page.goto("/");
            const picker = page.getByRole("button", { name: "Scene", exact: true });
            await expect(picker).toHaveText("CesiumMilkTruck");
            const canvas = await page.locator("canvas").elementHandle();
            // The accepted truck's green texture, not its silhouette or page chrome.
            await expect.poll(() => texturePixels(page)).toBeGreaterThan(100);
            const before = await page.locator("canvas").screenshot();
            await info.attach("truck", { body: before, contentType: "image/png" });
            if (phone) await picker.tap();
            else await picker.click();
            for (const name of ["core", "issues", "testing", "written"])
                await expect(page.getByRole("region", { name, exact: true })).toBeVisible();
            const box = page
                .getByRole("region", { name: "core", exact: true })
                .getByRole("button", { name: "BoxTextured", exact: true });
            if (phone) await box.tap();
            else await box.click();
            await expect(picker).toHaveText("BoxTextured");
            await expect(picker).toHaveAttribute("aria-expanded", "false");
            await expect
                .poll(async () => (await page.locator("canvas").screenshot()).equals(before))
                .toBe(false);
            // BoxTextured's Cesium logo has a blue sky; the carbon ground does not.
            await expect.poll(() => texturePixels(page, true)).toBeGreaterThan(100);
            await info.attach("box-textured", {
                body: await page.locator("canvas").screenshot(),
                contentType: "image/png",
            });
            expect(
                await page
                    .locator("canvas")
                    .evaluate((current, original) => current === original, canvas),
            ).toBe(true);
            expect(
                await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
            ).toBe(true);
            await expect(page.getByRole("slider")).toHaveCount(0);
            expect(errors).toEqual([]);
        });
        for (const name of ["SimpleSparseAccessor", "NegativeScaleTest", "MeshPrimitiveModes"])
            test(`${name} draws from its scene URL and remains selected in the catalog`, async ({
                page,
            }, info) => {
                await page.goto(`/?scene=${name}`);
                const picker = page.getByRole("button", { name: "Scene", exact: true });
                await expect(picker).toHaveText(name);
                await expect.poll(() => texturePixels(page, false, true)).toBeGreaterThan(100);
                await info.attach(name, {
                    body: await page.locator("canvas").screenshot(),
                    contentType: "image/png",
                });
                if (phone) await picker.tap();
                else await picker.click();
                await expect(
                    page
                        .getByRole("region", { name: "testing", exact: true })
                        .getByRole("button", { name, exact: true }),
                ).toHaveAttribute("aria-current", "true");
            });
    });
}
