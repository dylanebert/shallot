import type { Page } from "playwright/test";

export async function scenePixels(page: Page, blue = false) {
    const png = await page.locator("canvas").screenshot();
    return page.evaluate(
        async ({ data, blue }) => {
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
                    blue
                        ? rgba[i + 2] > rgba[i] * 1.3 && rgba[i + 2] > rgba[i + 1] * 1.1
                        : rgba[i + 1] > rgba[i] * 1.3 && rgba[i + 1] > rgba[i + 2] * 1.3
                )
                    colored++;
            }
            return colored;
        },
        { data: png.toString("base64"), blue },
    );
}
