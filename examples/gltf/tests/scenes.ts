import { expect, type Page, test } from "playwright/test";

export async function texturePixels(page: Page, blue = false, bright = false) {
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

export function sceneTests(names: readonly string[]) {
    for (const phone of [false, true]) {
        test.describe(phone ? "phone" : "desktop", () => {
            test.use({
                viewport: phone ? { width: 390, height: 844 } : { width: 1280, height: 720 },
                hasTouch: phone,
            });
            for (const name of names)
                test(`${name} draws from its scene URL and remains selected in the catalog`, async ({
                    page,
                }, info) => {
                    await page.goto(`/?scene=${name}`);
                    const picker = page.getByRole("button", { name: "Scene", exact: true });
                    await picker.waitFor();
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
}
