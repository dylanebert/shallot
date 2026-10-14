import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

declare global {
    interface Window {
        __releaseDeviceRequest?: () => void;
        __startupDeviceLoss?: boolean;
    }
}

const entry = "**/assets/index-*.js";
const bootstrapSelector = "[data-shallot-bootstrap]";
const loadingOverlay = "[data-title-card]";
const lightBackground = [28, 25, 23];
const lightTrack = [55, 54, 53];

async function holdEntry(page: Page): Promise<() => void> {
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

async function inspectBootstrap(page: Page): Promise<{ ink: number; background: number[] }> {
    const screenshot = await page.screenshot();
    return page.evaluate(
        async ({ base64, background }) => {
            const label = document.querySelector("[data-shallot-bootstrap] h1")?.firstChild;
            if (!label || label.nodeType !== Node.TEXT_NODE)
                return { ink: 0, background: [] as number[] };
            const range = document.createRange();
            range.selectNodeContents(label);
            const bounds = range.getBoundingClientRect();
            if (!bounds.width || !bounds.height) return { ink: 0, background: [] as number[] };

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

            let count = 0;
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
                        background.every(
                            (channel, index) => Math.abs(pixels[offset + index]! - channel) > 4,
                        )
                    )
                        count++;
                }
            }
            return { ink: count, background: [...pixels.slice(0, 3)] };
        },
        { base64: screenshot.toString("base64"), background: lightBackground },
    );
}

async function loadingTrackPixels(page: Page): Promise<{
    pixels: number;
    width: number;
    height: number;
    background: string;
    track: string;
}> {
    const screenshot = await page.screenshot();
    return page.evaluate(
        async ({ base64, selector, backgroundColor, trackColor }) => {
            const overlay = document.querySelector<HTMLElement>(selector);
            const track = overlay?.querySelector<HTMLElement>(".track");
            if (!overlay || !track)
                return { pixels: 0, width: 0, height: 0, background: "", track: "" };

            const bounds = track.getBoundingClientRect();
            const background = getComputedStyle(document.body).backgroundColor;
            const trackBackground = getComputedStyle(track).backgroundColor;
            const image = await createImageBitmap(
                new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))]),
            );
            const canvas = document.createElement("canvas");
            canvas.width = image.width;
            canvas.height = image.height;
            const context = canvas.getContext("2d")!;
            context.drawImage(image, 0, 0);
            const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
            image.close();
            const parse = (color: string) => (color.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number);
            const body = parse(background);
            const trackRgb = parse(trackBackground);
            const expectedBackground = parse(backgroundColor);
            const expectedTrack = parse(trackColor);
            let pixels = 0;
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
                    const matchesTrack = expectedTrack.every(
                        (channel, index) => Math.abs(rgba[offset + index]! - channel) <= 2,
                    );
                    const contrastsWithBody = expectedBackground.every(
                        (channel, index) => Math.abs(rgba[offset + index]! - channel) > 4,
                    );
                    if (matchesTrack && contrastsWithBody) pixels++;
                }
            }
            return {
                pixels,
                width: bounds.width,
                height: bounds.height,
                background: `rgb(${body.join(", ")})`,
                track: `rgb(${trackRgb.join(", ")})`,
            };
        },
        {
            base64: screenshot.toString("base64"),
            selector: loadingOverlay,
            backgroundColor: `rgb(${lightBackground.join(", ")})`,
            trackColor: `rgb(${lightTrack.join(", ")})`,
        },
    );
}

async function renderedCanvasPixels(page: Page): Promise<number> {
    const screenshot = await page.locator("#canvas").screenshot();
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
}

test("the game title bootstrap hands off without a jump to its own progress track", async ({
    page,
}) => {
    await page.addInitScript(() => {
        const original = GPUAdapter.prototype.requestDevice;
        let release!: () => void;
        const held = new Promise<void>((resolve) => {
            release = resolve;
        });
        window.__releaseDeviceRequest = release;
        GPUAdapter.prototype.requestDevice = function (descriptor?: GPUDeviceDescriptor) {
            return held.then(() => original.call(this, descriptor));
        };
    });

    const releaseEntry = await holdEntry(page);
    const bootstrap = await inspectBootstrap(page);
    expect(bootstrap.background).toEqual(lightBackground);
    expect(
        bootstrap.ink,
        "screenshot has contrasting ink in the custom bootstrap text bounds",
    ).toBeGreaterThan(20);
    await expect(page.locator(`${bootstrapSelector} h1`)).toHaveText("Ember Atlas");
    releaseEntry();
    await expect(page.locator(bootstrapSelector)).toHaveCount(0);
    await expect(page.locator(loadingOverlay)).toBeVisible();

    await page.evaluate(() => {
        document.querySelector<HTMLElement>(".progress")!.style.width = "38%";
    });
    if (process.env.SHALLOT_CAPTURE_DIR)
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/card-mid-load.png` });
    await page.evaluate(() => {
        document.querySelector<HTMLElement>(".progress")!.style.width = "0";
    });
    const runtimeTrack = await loadingTrackPixels(page);
    expect(runtimeTrack.background).toBe("rgb(28, 25, 23)");
    expect(runtimeTrack.track).toBe("rgb(55, 54, 53)");
    expect(runtimeTrack.width).toBe(420);
    expect(runtimeTrack.height).toBe(3);
    expect(
        runtimeTrack.pixels,
        "the held runtime screenshot contains contrasting title-card track pixels inside its bounds",
    ).toBeGreaterThan(700);

    const hiddenOverlay = await page.addStyleTag({
        content: `${loadingOverlay} { visibility: hidden !important; }`,
    });
    expect(
        (await loadingTrackPixels(page)).pixels,
        "the same witness rejects the hidden runtime overlay",
    ).toBe(0);
    await hiddenOverlay.evaluate((element) => element.parentNode?.removeChild(element));
    await page.evaluate(() => window.__releaseDeviceRequest?.());
    await expect(page.locator(loadingOverlay)).toHaveCount(0, { timeout: TIMEOUT.browser });
});

test("the complete card stays after progress reaches one and leaves after its dwell", async ({
    page,
}) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.addInitScript(() => {
        const observer = new MutationObserver(() => {
            const card = document.querySelector<HTMLElement>("[data-title-card]");
            if (card && !card.dataset.observedAt)
                card.dataset.observedAt = String(performance.now());
        });
        observer.observe(document, { subtree: true, attributes: true, childList: true });
    });
    await page.goto("/", { waitUntil: "commit" });
    const card = page.locator(loadingOverlay);
    await expect(card).toHaveAttribute("data-progress", "1");
    await expect(card).toBeVisible();
    const shownAt = Number(await card.getAttribute("data-observed-at"));
    expect((await page.evaluate(() => performance.now())) - shownAt).toBeLessThan(1500);
    await expect(card).toHaveCount(0, { timeout: TIMEOUT.browser });
    expect((await page.evaluate(() => performance.now())) - shownAt).toBeGreaterThanOrEqual(1500);
});

test("the screenshot witness rejects a blank bootstrap text control", async ({ page }) => {
    const releaseEntry = await holdEntry(page);
    await page.addStyleTag({ content: `${bootstrapSelector} { color: transparent !important; }` });
    expect(
        (await inspectBootstrap(page)).ink,
        "transparent custom bootstrap text leaves no contrasting pixels",
    ).toBe(0);
    releaseEntry();
});

test("an entry load abort settles the custom bootstrap to its startup error", async ({ page }) => {
    await page.route(entry, (route) => route.abort());
    await page.goto("/", { waitUntil: "commit" });
    const bootstrap = page.locator(bootstrapSelector);
    await expect(bootstrap).toHaveAttribute("data-state", "error");
    await expect(bootstrap).toHaveAttribute("role", "alert");
    await expect(bootstrap).toContainText("Unable to start this application");
});

test("a destroyed device before its first frame does not reach the failure screen", async ({
    page,
}) => {
    await page.addInitScript(() => {
        window.__startupDeviceLoss = false;
        const original = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = async function (descriptor?: GPUDeviceDescriptor) {
            const device = await original.call(this, descriptor);
            const onSubmittedWorkDone = GPUQueue.prototype.onSubmittedWorkDone;
            let fences = 0;
            GPUQueue.prototype.onSubmittedWorkDone = function (this: GPUQueue): Promise<void> {
                const fence = onSubmittedWorkDone.call(this);
                if (++fences === 2) {
                    window.__startupDeviceLoss = true;
                    device.destroy();
                }
                return fence;
            } as typeof GPUQueue.prototype.onSubmittedWorkDone;
            return device;
        };
    });

    await page.goto("/");
    await page.waitForFunction(() => window.__startupDeviceLoss === true);
    await expect(page.getByText("Something went wrong", { exact: true })).toHaveCount(0);
    await expect(page.getByText(/GPU device lost destroyed:/)).toHaveCount(0);
    await expect(page.locator(".frame-error")).toHaveCount(0);
});

test("a real GPU process crash after presentation reaches the failure screen", async ({
    page,
    browser,
}) => {
    await page.goto("/");
    await expect(page.locator(loadingOverlay)).toHaveCount(0, { timeout: TIMEOUT.browser });
    await expect
        .poll(() => renderedCanvasPixels(page), { timeout: TIMEOUT.browser })
        .toBeGreaterThan(100);
    if (process.env.SHALLOT_CAPTURE_DIR)
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/scene.png` });

    const cdp = await browser.newBrowserCDPSession();
    try {
        await cdp.send("Browser.crashGpuProcess");
        const failure = page.locator(loadingOverlay);
        await page.waitForFunction(
            (selector) =>
                document.querySelector(selector)?.textContent?.includes("Something went wrong"),
            loadingOverlay,
        );
        await expect(failure).toContainText("Something went wrong");
        await expect(failure).toContainText(
            "An error occurred in the app. The details below can help identify the cause.",
        );
        await expect(failure).toContainText("GPU device lost unknown:");
        await expect(failure).toContainText("Reloading the page may resolve this issue.");
    } finally {
        await cdp.detach().catch(() => {});
    }
});

test("a missing browser adapter offers the browser-specific remedy", async ({ page }) => {
    await page.addInitScript(() => {
        Object.defineProperty(navigator, "gpu", {
            configurable: true,
            value: { requestAdapter: async () => null },
        });
    });

    await page.goto("/");
    const failure = page.locator(loadingOverlay);
    await expect(failure).toContainText(
        "No WebGPU adapter is available in Browser. In a browser, enable hardware acceleration or use a browser with WebGPU support.",
    );
    await expect(failure).toContainText("WebGPU not supported.");
});
