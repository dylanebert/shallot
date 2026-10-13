import { mkdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { CEILING } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { PALETTES } from "./palette";
import { SCENES } from "./scenes";

interface AdapterIdentity {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
}

async function visiblePixelCount(page: Page): Promise<number> {
    const screenshot = await page.locator("#canvas").screenshot();
    return page.evaluate(async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
        const image = await createImageBitmap(new Blob([bytes]));
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        image.close();

        let count = 0;
        for (let offset = 0; offset < pixels.length; offset += 4) {
            if (
                Math.abs(pixels[offset]! - 28) > 8 ||
                Math.abs(pixels[offset + 1]! - 25) > 8 ||
                Math.abs(pixels[offset + 2]! - 23) > 8
            )
                count++;
        }
        return count;
    }, screenshot.toString("base64"));
}

async function requireRealAdapter(page: Page): Promise<AdapterIdentity> {
    const identity = await page.evaluate(async () => {
        if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU did not provide an adapter");
        const info = adapter.info;
        return {
            vendor: info.vendor,
            architecture: info.architecture,
            device: info.device,
            description: info.description,
        };
    });
    const label = Object.values(identity).join(" ").trim();
    expect(label, "the browser reports its WebGPU adapter identity").not.toBe("");
    expect(label, "captures use a hardware adapter, not a software renderer").not.toMatch(
        /swiftshader|llvmpipe|lavapipe|software rasterizer/i,
    );
    return identity;
}

test("captures every fixed scene and palette on a hardware WebGPU adapter", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/");
    const adapter = await requireRealAdapter(page);
    console.log(`Default-look WebGPU adapter: ${JSON.stringify(adapter)}`);

    const root = dirname(fileURLToPath(import.meta.url));
    for (const paletteName of Object.keys(PALETTES)) {
        for (const scene of SCENES) {
            await page.goto(
                `/?scene=${encodeURIComponent(scene.id)}&palette=${encodeURIComponent(paletteName)}`,
            );
            await expect(page.locator("body")).toHaveAttribute(
                "data-ready",
                `${paletteName}/${scene.id}`,
            );
            await expect
                .poll(() => visiblePixelCount(page), {
                    message: `${scene.id} paints geometry beyond its clear color`,
                    timeout: CEILING.browser,
                })
                .toBeGreaterThan(800);

            const directory = join(root, "captures", paletteName);
            await mkdir(directory, { recursive: true });
            const path = join(directory, `${scene.id}.png`);
            await page.screenshot({ path, animations: "disabled" });
            expect(
                (await stat(path)).size,
                `${paletteName}/${scene.id} PNG is nonempty`,
            ).toBeGreaterThan(1000);
            console.log(`Captured ${paletteName}/${scene.id}: ${path}`);
        }
    }
});
