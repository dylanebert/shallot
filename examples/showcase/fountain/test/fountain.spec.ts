import { expect, type Page, test } from "@playwright/test";

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — SwiftShader clears shallot's
// whole base floor (all three `BASE_FEATURES` plus 10 storage buffers per stage, measured under WSL
// 2026-08-10), so nothing about the floor distinguishes it, and it is only the *execution* that dies.
// Bias narrow: an unlisted software adapter re-crashes loudly, while an over-broad pattern would skip this
// gate on real hardware and report green having tested nothing.
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

/** the adapter's self-reported identity, or `""` when the browser hands out none. */
const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

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

    // Full device testing: the TGSL integrator and its readback need a real GPU. Probe before the boot
    // wait, so a GPU-less host skips honestly instead of dying inside `mapAsync` — display-gated the same
    // way `shallot verify`'s callers are.
    const adapter = await adapterName(page);
    console.log(`fountain gate adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

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
