import { expect, type Locator, type Page, test } from "@playwright/test";
import { classifyRendered } from "./rendered";

const DEMOS = ["immediate", "retained", "wireframe", "text", "tween"];

const sampleGrid = async (page: Page, canvas: Locator): Promise<number[]> => {
    const screenshot = await canvas.screenshot();
    return page.evaluate(async (base64) => {
        const binary = atob(base64);
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const surface = new OffscreenCanvas(64, 64);
        const context = surface.getContext("2d");
        if (!context) throw new Error("visualization gate: 2D screenshot context unavailable");
        context.drawImage(bitmap, 0, 0, 64, 64);
        bitmap.close();
        const rgba = context.getImageData(0, 0, 64, 64).data;
        const rgb: number[] = [];
        for (let at = 0; at < rgba.length; at += 4) rgb.push(rgba[at], rgba[at + 1], rgba[at + 2]);
        return rgb;
    }, screenshot.toString("base64"));
};

test("visualization showcase — every demo renders a positive canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        const text = message.text();
        if (
            message.type() === "error" ||
            /draw .* skipped|GPUValidationError|WebGPU.*(?:error|failed)/i.test(text)
        ) {
            errors.push(`[console.${message.type()}] ${text}`);
        }
    });

    for (const demo of DEMOS) {
        await page.goto(`/demos/${demo}.html`);
        const canvas = page.locator("canvas");
        await expect(canvas).toBeVisible();
        await page.waitForFunction(() => {
            const canvas = document.querySelector("canvas");
            return canvas instanceof HTMLCanvasElement && canvas.width > 0 && canvas.height > 0;
        });
        let classification = classifyRendered(await sampleGrid(page, canvas));
        await expect
            .poll(
                async () => {
                    classification = classifyRendered(await sampleGrid(page, canvas));
                    return classification.rendered;
                },
                { message: `${demo}: canonical center-vs-corner rendered law`, timeout: 10_000 },
            )
            .toBe(true);
        expect(classification.rendered, `${demo}: ${JSON.stringify(classification)}`).toBe(true);
        expect(errors, errors.join("\n")).toEqual([]);
    }
});
