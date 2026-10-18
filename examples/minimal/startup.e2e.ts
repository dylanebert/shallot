import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

const entry = "**/assets/index-*.js";
const loadingOverlay = "body > [data-shallot-loading]";

async function holdEntry(page: Page) {
    let release!: () => void;
    let reached!: () => void;
    const held = new Promise<void>((resolve) => {
        release = resolve;
    });
    const requested = new Promise<void>((resolve) => {
        reached = resolve;
    });
    await page.route(entry, async (route) => {
        reached();
        await held;
        await route.continue();
    });
    await page.goto("/", { waitUntil: "commit" });
    await requested;
    return release;
}

async function labelInkPixels(page: Page): Promise<number> {
    const screenshot = await page.screenshot();
    return page.evaluate(async (base64) => {
        const bootstrap = document.querySelector("[data-shallot-bootstrap]");
        const text = bootstrap?.firstChild;
        if (!text || text.nodeType !== Node.TEXT_NODE) return 0;
        const range = document.createRange();
        range.selectNodeContents(text);
        const bounds = range.getBoundingClientRect();
        if (bounds.width === 0 || bounds.height === 0) return 0;

        const image = await createImageBitmap(
            new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))]),
        );
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        image.close();

        let contrasting = 0;
        for (
            let y = Math.max(0, Math.floor(bounds.top));
            y < Math.min(canvas.height, Math.ceil(bounds.bottom));
            y++
        ) {
            for (
                let x = Math.max(0, Math.floor(bounds.left));
                x < Math.min(canvas.width, Math.ceil(bounds.right));
                x++
            ) {
                const offset = (y * canvas.width + x) * 4;
                if (
                    Math.abs(pixels[offset]! - 40) > 4 &&
                    Math.abs(pixels[offset + 1]! - 38) > 4 &&
                    Math.abs(pixels[offset + 2]! - 34) > 4
                )
                    contrasting++;
            }
        }
        return contrasting;
    }, screenshot.toString("base64"));
}

async function expectPaintedLabel(page: Page) {
    expect(
        await labelInkPixels(page),
        "screenshot has visible label ink against the bootstrap background",
    ).toBeGreaterThan(20);
}

test("minimal paints its bootstrap before the held application entry", async ({ page }) => {
    const release = await holdEntry(page);
    await expectPaintedLabel(page);
    release();
});

test("the screenshot witness rejects a blank bootstrap control", async ({ page }) => {
    const release = await holdEntry(page);
    await page.addStyleTag({
        content: "[data-shallot-bootstrap] { color: transparent !important; }",
    });
    expect(
        await labelInkPixels(page),
        "transparent bootstrap text leaves no contrasting label pixels",
    ).toBe(0);
    release();
});

test("releasing the entry hands off to the loading screen and starts the scene", async ({
    page,
}) => {
    const release = await holdEntry(page);
    await expectPaintedLabel(page);
    release();
    await expect(page.locator("[data-shallot-bootstrap]")).toHaveCount(0);
    await expect(page.locator(loadingOverlay)).toBeVisible();
    await expect(page.locator(loadingOverlay)).toHaveCount(0, { timeout: TIMEOUT.browser });

    const canvas = page.locator("#canvas");
    await expect(canvas).toBeVisible();
    await expect
        .poll(
            async () => {
                const screenshot = await canvas.screenshot();
                return page.evaluate(async (base64) => {
                    const image = await createImageBitmap(
                        new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))]),
                    );
                    const canvas = document.createElement("canvas");
                    canvas.width = image.width;
                    canvas.height = image.height;
                    const context = canvas.getContext("2d")!;
                    context.drawImage(image, 0, 0);
                    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
                    image.close();
                    let visible = 0;
                    for (let i = 0; i < pixels.length; i += 4) {
                        if (
                            pixels[i + 3]! > 0 &&
                            (pixels[i]! > 35 || pixels[i + 1]! > 35 || pixels[i + 2]! > 35)
                        )
                            visible++;
                    }
                    return visible;
                }, screenshot.toString("base64"));
            },
            { timeout: TIMEOUT.browser },
        )
        .toBeGreaterThan(100);
});

test("entry evaluation, dependency rejection, entry load abort, and unhandled rejection settle the bootstrap", async ({
    page,
}) => {
    for (const failure of ["evaluation", "dependency", "load", "rejection"] as const) {
        await page.route(entry, async (route) => {
            if (failure === "load") await route.abort();
            else
                await route.fulfill({
                    status: 200,
                    contentType: "text/javascript",
                    body:
                        failure === "evaluation"
                            ? 'throw new Error("controlled entry evaluation failure");'
                            : failure === "rejection"
                              ? 'void Promise.reject(new Error("controlled entry rejection"));'
                              : 'import "./missing-startup-dependency.js";',
                });
        });
        await page.route("**/assets/missing-startup-dependency.js", (route) => route.abort());
        await page.goto("/", { waitUntil: "commit" });
        const bootstrap = page.locator("[data-shallot-bootstrap]");
        await expect(bootstrap).toHaveAttribute("data-state", "error");
        await expect(bootstrap).toHaveAttribute("role", "alert");
        await expect(bootstrap).toContainText("Unable to start this application");
        await page.unrouteAll({ behavior: "wait" });
        await page.goto("about:blank");
    }
});
