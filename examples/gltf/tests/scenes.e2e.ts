import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { sceneTests, texturePixels } from "./scenes";

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
            await picker.waitFor();
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
    });
}
sceneTests(["SimpleSparseAccessor", "NegativeScaleTest", "MeshPrimitiveModes"]);
