import { expect, test } from "@playwright/test";

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

test("fountain showcase gate — TGSL integrator and GPU readback", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (message) => {
        const text = message.text();
        if (
            message.type() === "error" ||
            /draw .* skipped|GPUValidationError|WebGPU.*(?:error|failed)/i.test(text)
        ) {
            errors.push(`[console.${message.type()}] ${text}`);
        }
    });

    await page.goto("/");
    await page.waitForFunction(() => typeof window.__fountainGate === "function", null, {
        timeout: 120_000,
    });

    const checks = (await page.evaluate(() =>
        (window as unknown as { __fountainGate: () => Promise<Check[]> }).__fountainGate(),
    )) as Check[];

    expect(errors, errors.join("\n")).toEqual([]);
    expect(checks.length).toBeGreaterThan(0);
    for (const c of checks) {
        expect(c.pass, `${c.name}: ${c.detail}`).toBe(true);
    }

    // The ground and background are neutral gray; only the fountain contributes a high-chroma field
    // above the ground. Decode the actual browser screenshot in-page so the product rung observes what
    // the user sees, not the particle storage buffer the compute rung already covered.
    const screenshot = await page.screenshot();
    const pixels = await page.evaluate(async (base64) => {
        const binary = atob(base64);
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext("2d");
        if (!context) throw new Error("fountain gate: 2D screenshot context unavailable");
        context.drawImage(bitmap, 0, 0);
        const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
        let chromatic = 0;
        const rows = Math.floor(bitmap.height * 0.82);
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < bitmap.width; x++) {
                const at = (y * bitmap.width + x) * 4;
                const r = data[at];
                const g = data[at + 1];
                const b = data[at + 2];
                if (Math.max(r, g, b) - Math.min(r, g, b) >= 36 && Math.max(r, g, b) >= 64) {
                    chromatic++;
                }
            }
        }
        return { chromatic, width: bitmap.width, height: bitmap.height };
    }, screenshot.toString("base64"));
    expect(pixels.chromatic, `fountain product pixels: ${JSON.stringify(pixels)}`).toBeGreaterThan(
        250,
    );
    expect(errors, errors.join("\n")).toEqual([]);
});
