import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "playwright/test";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const CAPTURE_ENTRY = resolve(ROOT, ".artifacts/capture-page.js");
execFileSync("bun", ["run", "scripts/build-capture-page.ts"], { cwd: ROOT });

test("a browser capture preserves the declared page geometry and shows its color tag", async ({
    page,
}) => {
    await page.goto("/");
    const canvas = page.locator("#scene");
    await expect(canvas).toHaveClass(/visible/);
    expect(page.viewportSize()).toEqual({ width: 1280, height: 720 });
    await page.waitForTimeout(200);

    const first = await page.screenshot();
    const second = await page.screenshot();
    expect(first.equals(second), "two page screenshots of one state are byte-identical").toBe(true);

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
    expect(pixels.width, "the page screenshot uses the declared width").toBe(1280);
    expect(pixels.height, "the page screenshot uses the declared height").toBe(720);
    expect(pixels.ink, "the real page's color-tag description remains visible").toBeGreaterThan(20);

    await page.addScriptTag({ path: CAPTURE_ENTRY });
    await page.addStyleTag({
        content: "#frame { width: 1280px !important; height: 720px !important; }",
    });
    await expect
        .poll(() => canvas.evaluate((element) => (element as HTMLCanvasElement).width))
        .toBe(1280);
    await expect
        .poll(() => canvas.evaluate((element) => (element as HTMLCanvasElement).height))
        .toBe(720);
    const shot = await page.evaluate(async () => {
        const capture = (
            globalThis as typeof globalThis & {
                shallotCaptureFrame: (canvas: HTMLCanvasElement) => Promise<{
                    width: number;
                    height: number;
                    rgba: Uint8ClampedArray;
                    identity: { width: number; height: number; surface: string; encoding: string };
                }>;
            }
        ).shallotCaptureFrame;
        const image = await capture(document.querySelector<HTMLCanvasElement>("#scene")!);
        return {
            width: image.width,
            height: image.height,
            identity: image.identity,
            hasColor: image.rgba.some((value, index) => index % 4 !== 3 && value !== 0),
            byteLength: image.rgba.byteLength,
        };
    });
    expect(shot.width).toBe(1280);
    expect(shot.height).toBe(720);
    expect(shot.identity).toMatchObject({
        width: 1280,
        height: 720,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
    expect(shot.hasColor).toBe(true);
    expect(shot.byteLength).toBe(1280 * 720 * 4);
});
