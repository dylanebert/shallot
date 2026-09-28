import { expect, test } from "playwright/test";

test("a browser capture preserves the declared page geometry, shows its color tag, and is byte-identical across two captures of one rendered state", async ({
    page,
}) => {
    await page.goto("/");
    const canvas = page.locator("#scene");
    await expect(canvas).toHaveClass(/visible/);
    expect(page.viewportSize()).toEqual({ width: 1280, height: 720 });
    await page.waitForTimeout(200);

    const first = await page.screenshot();
    const second = await page.screenshot();
    expect(first.equals(second), "two captures of the same rendered state are byte-identical").toBe(
        true,
    );

    const pixels = await page.evaluate(async (encoded) => {
        const image = await createImageBitmap(
            new Blob([Uint8Array.from(atob(encoded), (value) => value.charCodeAt(0))], {
                type: "image/png",
            }),
        );
        const surface = document.createElement("canvas");
        surface.width = image.width;
        surface.height = image.height;
        const context = surface.getContext("2d")!;
        context.drawImage(image, 0, 0);
        const rgba = context.getImageData(0, 0, image.width, image.height).data;
        image.close();
        const line = document.querySelector<HTMLElement>(".description")!.getBoundingClientRect();
        let ink = 0;
        for (
            let y = Math.max(0, Math.floor(line.top));
            y < Math.min(surface.height, Math.ceil(line.bottom));
            y++
        ) {
            for (
                let x = Math.max(0, Math.floor(line.left));
                x < Math.min(surface.width, Math.ceil(line.right));
                x++
            ) {
                const index = (y * surface.width + x) * 4;
                if (rgba[index]! > 210 && rgba[index + 1]! > 190 && rgba[index + 2]! > 170) ink++;
            }
        }
        return { width: surface.width, height: surface.height, ink };
    }, first.toString("base64"));
    expect(pixels.width, "the page capture uses the declared width").toBe(1280);
    expect(pixels.height, "the page capture uses the declared height").toBe(720);
    expect(
        pixels.ink,
        "the real page's color-tag description remains visible in the capture",
    ).toBeGreaterThan(20);
});
