import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, releasing, test } from "../tests/browser-fixture";

const CAPTURE_ENTRY = resolve(
    dirname(fileURLToPath(import.meta.url)),
    "../.artifacts/capture-page.js",
);

interface BrowserCapture {
    rgba: Uint8ClampedArray;
    width: number;
    height: number;
    identity: {
        width: number;
        height: number;
        deviceScale: number;
        surface: string;
        encoding: string;
    };
}

declare global {
    interface Window {
        shallotCaptureFrame?: (canvas: HTMLCanvasElement) => Promise<BrowserCapture>;
        __sceneFrames?: Record<string, Uint8ClampedArray>;
        __releaseLoading?: () => void;
        __progressWidths?: number[];
        __releaseReveal?: () => void;
        __revealBeforeFence?: boolean;
        __releaseFirstFrameFence?: () => void;
        __firstFrameFenceReady?: boolean;
        __startupDeviceLoss?: boolean;
        __subjectDeviceRecord?: {
            identity: string;
            requestedFeatures: string[];
            requestedLimits: Record<string, number | undefined>;
            features: string[];
            limits: Record<string, unknown>;
        };
    }
}

const overlaySelector = "#frame > div[style*='z-index: 10000']";

async function openPage(page: Page): Promise<void> {
    await page.goto("/");
    await expect(page.locator("#frame")).toBeVisible();
}

async function waitForScene(page: Page): Promise<void> {
    await openPage(page);
    // A cold first draw on hosted macOS outlasts expect's 5 s default (410e41d0), so the reveal wait is
    // bounded by the test's timeout instead.
    await page.locator("#scene.visible").waitFor();
}

async function inspectStartup(page: Page) {
    const screenshot = await page.screenshot();
    const inspection = await page.evaluate(async (base64) => {
        const image = await createImageBitmap(
            new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))]),
        );
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const rect = (selector: string) => {
            const element = document.querySelector<HTMLElement>(selector);
            if (!element) return null;
            const bounds = element.getBoundingClientRect();
            return {
                left: bounds.left,
                top: bounds.top,
                right: bounds.right,
                bottom: bounds.bottom,
            };
        };
        const frame = rect("#frame");
        const label = rect("[data-bootstrap-label]");
        const track = rect("[data-bootstrap-track]");
        const frameElement = document.querySelector<HTMLElement>("#frame");
        const background = frameElement
            ? getComputedStyle(
                  document.querySelector<HTMLElement>("[data-shallot-bootstrap]") ?? frameElement,
              )
                  .backgroundColor.match(/[\d.]+/g)!
                  .slice(0, 3)
                  .map(Number)
            : [28, 25, 23];
        const countContrasting = (bounds: NonNullable<typeof frame>) => {
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
                        Math.max(
                            ...background.map((channel, index) =>
                                Math.abs(rgba[offset + index]! - channel),
                            ),
                        ) > 4
                    )
                        count++;
                }
            }
            return count;
        };
        const result = {
            width: canvas.width,
            height: canvas.height,
            frame,
            label,
            track,
            labelInkPixels: label ? countContrasting(label) : 0,
            trackContrastPixels: track ? countContrasting(track) : 0,
            description: rect(".description"),
            bootstrap: rect("[data-shallot-bootstrap]"),
        };
        image.close();
        return result;
    }, screenshot.toString("base64"));
    return inspection;
}

function expectStartupPixels(shot: Awaited<ReturnType<typeof inspectStartup>>): void {
    expect(
        shot.labelInkPixels,
        "screenshot contains loading-label ink inside its label bounds",
    ).toBeGreaterThan(8);
    expect(
        shot.trackContrastPixels,
        "screenshot contains the track within its declared bounds",
    ).toBeGreaterThan(100);
}

async function expectPaintedStartup(
    page: Page,
): Promise<Awaited<ReturnType<typeof inspectStartup>>> {
    const shot = await inspectStartup(page);
    expectStartupPixels(shot);
    await expect(page.locator(".description")).toBeVisible();
    await expect(page.locator("#frame")).toBeVisible();
    await expect(page.locator("[data-shallot-bootstrap]")).toBeVisible();
    await expect(page.locator("[data-bootstrap-label]")).toHaveText("Loading…");
    await expect(page.locator("[data-bootstrap-track]")).toBeVisible();
    const descriptionHit = await page.locator(".description").evaluate((element) => {
        const bounds = element.getBoundingClientRect();
        return (
            document.elementFromPoint(
                bounds.left + bounds.width / 2,
                bounds.top + bounds.height / 2,
            ) === element
        );
    });
    expect(
        descriptionHit,
        "the description remains hit-testable while the bootstrap is active",
    ).toBe(true);
    expect(shot.description, "the host description exists before entry").not.toBeNull();
    expect(shot.frame).not.toBeNull();
    const { description, frame, bootstrap } = shot;
    expect(description!.top).toBeGreaterThanOrEqual(0);
    expect(description!.bottom).toBeLessThanOrEqual(frame!.top);
    expect(description!.left).toBeGreaterThanOrEqual(0);
    expect(description!.right).toBeLessThanOrEqual(shot.width);
    expect(bootstrap!.left).toBeCloseTo(frame!.left, 0);
    expect(bootstrap!.top).toBeCloseTo(frame!.top, 0);
    expect(bootstrap!.right).toBeCloseTo(frame!.right, 0);
    expect(bootstrap!.bottom).toBeCloseTo(frame!.bottom, 0);
    expect(frame!.left).toBeGreaterThanOrEqual(0);
    expect(frame!.top).toBeGreaterThanOrEqual(0);
    expect(frame!.right).toBeLessThanOrEqual(shot.width);
    expect(frame!.bottom).toBeLessThanOrEqual(shot.height);
    return shot;
}

test("host paints contained semantic startup feedback before application entry, then hands off", async ({
    page,
}) => {
    let release!: () => void;
    let entryStarted!: () => void;
    let applicationEntry = "";
    const held = new Promise<void>((resolve) => {
        release = resolve;
    });
    const reached = new Promise<void>((resolve) => {
        entryStarted = resolve;
    });
    await page.route("**/assets/index-*.js", async (route) => {
        applicationEntry = route.request().url();
        entryStarted();
        await held;
        await route.continue();
    });
    await page.goto("/", { waitUntil: "commit" });
    await reached;
    expect(applicationEntry).toMatch(/\/assets\/index-[^/]+\.js$/);
    await page.setViewportSize({ width: 1280, height: 720 });
    await expectPaintedStartup(page);
    const evidence = process.env["SHALLOT_STARTUP_EVIDENCE"];
    if (evidence) await page.screenshot({ path: join(evidence, "startup.png") });
    await page.setViewportSize({ width: 390, height: 844 });
    await expectPaintedStartup(page);
    if (evidence) await page.screenshot({ path: join(evidence, "startup-mobile.png") });
    await page.setViewportSize({ width: 820, height: 1180 });
    await expectPaintedStartup(page);
    if (evidence) await page.screenshot({ path: join(evidence, "startup-tablet.png") });
    await page.setViewportSize({ width: 1280, height: 720 });
    const bootstrap = await page.locator("[data-shallot-bootstrap]").elementHandle();
    release();
    await expect(page.locator("[data-shallot-bootstrap]")).toHaveCount(0);
    await expect(page.locator(overlaySelector)).toBeVisible();
    await expect(page.locator(".description")).toHaveCount(1);
    await page.evaluate(() =>
        window.dispatchEvent(new ErrorEvent("error", { message: "late unrelated error" })),
    );
    expect(
        await bootstrap?.evaluate((element) => ({
            connected: element.isConnected,
            text: element.textContent?.trim(),
            state: element.getAttribute("data-state"),
        })),
    ).toEqual({
        connected: false,
        text: "Loading…",
        state: "loading",
    });
});

test("the screenshot witness rejects the held base and a blank-feedback control", async ({
    page,
}) => {
    let release!: () => void;
    let releaseBlank!: () => void;
    let entryStarted!: () => void;
    let blankStarted!: () => void;
    let entryCount = 0;
    const held = new Promise<void>((resolve) => {
        release = resolve;
    });
    const blankHeld = new Promise<void>((resolve) => {
        releaseBlank = resolve;
    });
    const reached = new Promise<void>((resolve) => {
        entryStarted = resolve;
    });
    const blankReached = new Promise<void>((resolve) => {
        blankStarted = resolve;
    });
    await page.route("**/assets/index-*.js", async (route) => {
        if (entryCount++ === 0) {
            entryStarted();
            await held;
        } else {
            blankStarted();
            await blankHeld;
        }
        await route.continue();
    });
    await page.goto("/", { waitUntil: "commit" });
    await reached;
    await page.evaluate(() => document.querySelector("#app")!.replaceChildren());
    const evidence = process.env["SHALLOT_STARTUP_EVIDENCE"];
    if (evidence) await page.screenshot({ path: join(evidence, "startup-base-control.png") });
    const base = await inspectStartup(page);
    expect(
        () => expectStartupPixels(base),
        "held base screenshot fails the same ink-and-track witness",
    ).toThrow();
    release();

    await page.goto("/", { waitUntil: "commit" });
    await blankReached;
    await page.addStyleTag({
        content:
            "[data-bootstrap-label], [data-bootstrap-track] { visibility: hidden !important; }",
    });
    if (evidence) await page.screenshot({ path: join(evidence, "startup-blank-control.png") });
    const blank = await inspectStartup(page);
    expect(
        () => expectStartupPixels(blank),
        "blank screenshot fails the same ink-and-track witness",
    ).toThrow();
    releaseBlank();
});

async function inspectPageScreenshot(page: Page, encoded: string) {
    return page.evaluate((base64) => {
        const image = createImageBitmap(
            new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))], {
                type: "image/png",
            }),
        );
        return image.then((bitmap) => {
            const surface = document.createElement("canvas");
            surface.width = bitmap.width;
            surface.height = bitmap.height;
            const context = surface.getContext("2d")!;
            context.drawImage(bitmap, 0, 0);
            const rgba = context.getImageData(0, 0, surface.width, surface.height).data;
            bitmap.close();
            const channels = (value: string) =>
                (value.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number);
            const frame = document.querySelector<HTMLElement>("#frame")!.getBoundingClientRect();
            const canvas = document.querySelector<HTMLElement>("#scene")!.getBoundingClientRect();
            const description = document
                .querySelector<HTMLElement>(".description")!
                .getBoundingClientRect();
            const overlay = document.querySelector<HTMLElement>(
                "#frame > div[style*='z-index: 10000']",
            );
            const progress = [...(overlay?.querySelectorAll<HTMLElement>("div") ?? [])].find(
                (node) =>
                    node.style.height === "100%" && node.parentElement?.style.overflow === "hidden",
            );
            const pageColor = channels(getComputedStyle(document.body).backgroundColor);
            const frameColor = channels(
                getComputedStyle(overlay ?? document.querySelector<HTMLElement>("#frame")!)
                    .backgroundColor,
            );
            const inkColor = channels(
                getComputedStyle(document.querySelector(".description")!).color,
            );
            const progressColor = progress
                ? channels(getComputedStyle(progress).backgroundColor)
                : [];
            // Non-real adapters add a notice panel: its surface is loading UI too,
            // not a hole exposing the canvas. Count that colour only inside its panel.
            const notice = [...(overlay?.querySelectorAll<HTMLElement>("div") ?? [])].find(
                (node) =>
                    node.childElementCount === 0 && /^\S+ adapter: /.test(node.textContent ?? ""),
            );
            const noticeRect = notice?.getBoundingClientRect();
            const noticeColor = notice ? channels(getComputedStyle(notice).backgroundColor) : [];
            const count = (
                rect: DOMRect,
                color: number[],
                tolerance: number,
                withNotice = false,
            ) => {
                let pixels = 0;
                for (
                    let y = Math.max(0, Math.floor(rect.top));
                    y < Math.min(surface.height, Math.ceil(rect.bottom));
                    y++
                ) {
                    for (
                        let x = Math.max(0, Math.floor(rect.left));
                        x < Math.min(surface.width, Math.ceil(rect.right));
                        x++
                    ) {
                        const offset = (y * surface.width + x) * 4;
                        const matches = color.every(
                            (channel, i) => Math.abs(rgba[offset + i]! - channel) <= tolerance,
                        );
                        const noticeMatches =
                            withNotice &&
                            noticeRect !== undefined &&
                            x >= noticeRect.left &&
                            x < noticeRect.right &&
                            y >= noticeRect.top &&
                            y < noticeRect.bottom &&
                            noticeColor.every(
                                (channel, i) => Math.abs(rgba[offset + i]! - channel) <= tolerance,
                            );
                        if (matches || noticeMatches) pixels++;
                    }
                }
                return pixels;
            };
            const inkPixels = count(description, inkColor, 24);
            const progressPixels = progressColor.length ? count(frame, progressColor, 24) : 0;
            const framePixels = count(canvas, frameColor, 2, true);
            const canvasArea = Math.max(1, Math.round(canvas.width) * Math.round(canvas.height));
            const sample = (x: number, y: number) => {
                const offset = (Math.round(y) * surface.width + Math.round(x)) * 4;
                return [rgba[offset]!, rgba[offset + 1]!, rgba[offset + 2]!];
            };
            const channelError = (actual: number[], expected: number[]) =>
                Math.max(...actual.map((channel, i) => Math.abs(channel - expected[i]!)));
            const pageSample = sample(8, 8);
            const adjacentSample = sample(frame.left - 1, frame.top + frame.height / 2);
            return {
                width: surface.width,
                height: surface.height,
                inkPixels,
                progressPixels,
                frameBackgroundFraction: framePixels / canvasArea,
                pageBackgroundError: channelError(pageSample, pageColor),
                adjacentBackgroundError: channelError(adjacentSample, pageColor),
                progressWidth: progress?.style.width ?? "",
                overlayPresent: overlay?.isConnected === true,
            };
        });
    }, encoded);
}

function oklab(rgb: number[]) {
    const [r, g, b] = rgb.map((v) => {
        const s = v / 255;
        return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    const l = Math.cbrt(0.4122214708 * r! + 0.5363325363 * g! + 0.0514459929 * b!);
    const m = Math.cbrt(0.2119034982 * r! + 0.6806995451 * g! + 0.1073969566 * b!);
    const s = Math.cbrt(0.0883024619 * r! + 0.2817188376 * g! + 0.6299787005 * b!);
    return [
        0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    ];
}

async function inspectScene(page: Page, frameKey: string) {
    const result = await page.evaluate(async (key) => {
        if (!window.shallotCaptureFrame) throw new Error("captureFrame is not installed");
        const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
        const capture = await window.shallotCaptureFrame(canvas);
        (window.__sceneFrames ??= {})[key] = capture.rgba;
        const surface = { width: capture.width, height: capture.height };
        const rgba = capture.rgba;
        const frameColor = (
            getComputedStyle(document.querySelector("#frame")!).backgroundColor.match(/[\d.]+/g) ??
            []
        )
            .slice(0, 3)
            .map(Number);
        const yaw = 0.6;
        const pitch = 0.25;
        const distance = 5;
        const focal = surface.height / (2 * Math.tan((60 * Math.PI) / 360));
        const yawCos = Math.cos(yaw);
        const yawSin = Math.sin(yaw);
        const pitchCos = Math.cos(pitch);
        const pitchSin = Math.sin(pitch);
        const direction = { x: pitchCos * yawSin, y: pitchSin, z: pitchCos * yawCos };
        const right = { x: yawCos, y: 0, z: -yawSin };
        const up = { x: -pitchSin * yawSin, y: pitchCos, z: -pitchSin * yawCos };
        const points: { x: number; y: number }[] = [];
        for (const x of [-0.5, 0.5]) {
            for (const y of [-0.5, 0.5]) {
                for (const z of [-0.5, 0.5]) {
                    const depth = distance - (x * direction.x + y * direction.y + z * direction.z);
                    points.push({
                        x:
                            surface.width / 2 +
                            (focal * (x * right.x + y * right.y + z * right.z)) / depth,
                        y: surface.height / 2 - (focal * (x * up.x + y * up.y + z * up.z)) / depth,
                    });
                }
            }
        }
        const cross = (
            origin: (typeof points)[number],
            a: (typeof points)[number],
            b: (typeof points)[number],
        ) => (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
        const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
        const lower: typeof points = [];
        for (const point of sorted) {
            while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower.at(-1)!, point) <= 0)
                lower.pop();
            lower.push(point);
        }
        const upper: typeof points = [];
        for (const point of sorted.toReversed()) {
            while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper.at(-1)!, point) <= 0)
                upper.pop();
            upper.push(point);
        }
        const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
        let twiceArea = 0;
        for (let i = 0; i < hull.length; i++) {
            const current = hull[i]!;
            const next = hull[(i + 1) % hull.length]!;
            twiceArea += current.x * next.y - current.y * next.x;
        }
        const expectedPopulation = Math.abs(twiceArea) / 2;
        const expectedLeft = Math.min(...hull.map((point) => point.x));
        const expectedRight = Math.max(...hull.map((point) => point.x));
        const expectedTop = Math.min(...hull.map((point) => point.y));
        const expectedBottom = Math.max(...hull.map((point) => point.y));
        const expectedWidth = expectedRight - expectedLeft;
        const expectedHeight = expectedBottom - expectedTop;
        let scenePixels = 0;
        let left = surface.width;
        let top = surface.height;
        let rightEdge = -1;
        let bottom = -1;
        for (let y = 0; y < surface.height; y++) {
            for (let x = 0; x < surface.width; x++) {
                const offset = (y * surface.width + x) * 4;
                const different =
                    rgba[offset]! > rgba[offset + 1]! * 1.25 &&
                    rgba[offset + 1]! > rgba[offset + 2]! * 1.3;
                if (different) {
                    scenePixels++;
                    left = Math.min(left, x);
                    top = Math.min(top, y);
                    rightEdge = Math.max(rightEdge, x);
                    bottom = Math.max(bottom, y);
                }
            }
        }
        const width = rightEdge - left + 1;
        const height = bottom - top + 1;
        const centerX = (left + rightEdge) / 2;
        const centerY = (top + bottom) / 2;
        const expectedCenterX = (expectedLeft + expectedRight) / 2;
        const expectedCenterY = (expectedTop + expectedBottom) / 2;
        const geometry =
            scenePixels >= expectedPopulation * 0.5 &&
            scenePixels <= expectedPopulation * 1.5 &&
            width >= expectedWidth * 0.72 &&
            width <= expectedWidth * 1.28 &&
            height >= expectedHeight * 0.72 &&
            height <= expectedHeight * 1.28 &&
            Math.abs(centerX - expectedCenterX) <= expectedWidth * 0.12 &&
            Math.abs(centerY - expectedCenterY) <= expectedHeight * 0.12;
        const backgroundSum = [0, 0, 0];
        let compared = 0;
        for (let y = 0; y < surface.height; y++) {
            for (let x = 0; x < surface.width; x++) {
                if (y >= expectedTop - 2) continue;
                const px = x + 0.5;
                const py = y + 0.5;
                if (
                    px >= expectedLeft - 1 &&
                    px <= expectedRight + 1 &&
                    py >= expectedTop - 1 &&
                    py <= expectedBottom + 1
                )
                    continue;
                const offset = (y * surface.width + x) * 4;
                for (let channel = 0; channel < 3; channel++) {
                    const value = rgba[offset + channel]!;
                    backgroundSum[channel]! += value;
                }
                compared++;
            }
        }
        const backgroundMean = backgroundSum.map((sum) => sum / compared);
        return {
            width: surface.width,
            height: surface.height,
            scenePixels,
            expectedPopulation,
            geometry,
            backgroundMean,
            frameColor,
            compared,
        };
    }, frameKey);
    const background = oklab(result.backgroundMean);
    const frame = oklab(result.frameColor);
    return {
        ...result,
        backgroundDeltaE: Math.hypot(
            ...background.map((value, channel) => value - frame[channel]!),
        ),
    };
}

test("light document and rendered frame differ by at least 0.6 OKLab lightness", async ({
    page,
}) => {
    await waitForScene(page);
    await page.addScriptTag({ path: CAPTURE_ENTRY });
    await page.addStyleTag({
        content: "#frame { width: 1280px !important; height: 720px !important; }",
    });
    await expect
        .poll(() => page.locator("#scene").evaluate((e) => (e as HTMLCanvasElement).width))
        .toBe(1280);
    const colors = await page.evaluate(async () => {
        const shot = await window.shallotCaptureFrame!(
            document.querySelector<HTMLCanvasElement>("#scene")!,
        );
        const page = getComputedStyle(document.body)
            .backgroundColor.match(/\d+/g)!
            .slice(0, 3)
            .map(Number);
        const clear = Array.from(shot.rgba.slice(0, 3));
        console.log("rendered clear", clear);
        return { page, clear };
    });
    expect(oklab(colors.page)[0]! - oklab(colors.clear)[0]!).toBeGreaterThanOrEqual(0.6);
});

async function compareSceneFrames(page: Page, first: string, later: string) {
    return page.evaluate(
        ({ first, later }) => {
            const frames = window.__sceneFrames ?? {};
            const a = frames[first];
            const b = frames[later];
            if (!a || !b || a.length !== b.length) return { meanError: 255, changedFraction: 1 };
            let totalError = 0;
            let changedPixels = 0;
            const pixelCount = a.length / 4;
            for (let offset = 0; offset < a.length; offset += 4) {
                let largestError = 0;
                for (let channel = 0; channel < 3; channel++) {
                    const error = Math.abs(a[offset + channel]! - b[offset + channel]!);
                    totalError += error;
                    largestError = Math.max(largestError, error);
                }
                if (largestError > 2) changedPixels++;
            }
            return {
                meanError: totalError / (pixelCount * 3),
                changedFraction: changedPixels / pixelCount,
            };
        },
        { first, later },
    );
}

test("embedded-scene progress stays in its owned frame and reaches the bar; its description remains hit-testable", async ({
    page,
}) => {
    await page.addInitScript(() => {
        window.__progressWidths = [];
        new MutationObserver(() => {
            for (const element of document.querySelectorAll<HTMLElement>("#frame *")) {
                const width = Number.parseFloat(element.style.width);
                if (
                    element.style.height === "100%" &&
                    element.parentElement?.style.overflow === "hidden" &&
                    Number.isFinite(width) &&
                    element.style.width.endsWith("%")
                )
                    window.__progressWidths!.push(width);
            }
            const overlay = document.querySelector<HTMLElement>(
                "#frame > div[style*='z-index: 10000']",
            );
            if (!overlay || overlay.dataset.cleanupHeld) return;
            overlay.dataset.cleanupHeld = "true";
            const remove = overlay.remove.bind(overlay);
            overlay.remove = () => {
                window.__releaseLoading = remove;
            };
        }).observe(document, {
            attributes: true,
            childList: true,
            subtree: true,
            attributeFilter: ["style"],
        });

        const original = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = async function (descriptor?: GPUDeviceDescriptor) {
            const info = this.info;
            const device = await original.call(this, descriptor);
            const limits = Object.fromEntries(
                Object.getOwnPropertyNames(Object.getPrototypeOf(device.limits))
                    .filter((name) => name !== "constructor")
                    .map((name) => [name, device.limits[name as keyof GPUSupportedLimits]]),
            );
            window.__subjectDeviceRecord = {
                identity:
                    [info.vendor, info.architecture, info.device, info.description]
                        .filter((value) => typeof value === "string" && value.trim() !== "")
                        .join(" ") || "unidentified",
                requestedFeatures: [...(descriptor?.requiredFeatures ?? [])].sort(),
                requestedLimits: { ...(descriptor?.requiredLimits ?? {}) },
                features: [...device.features]
                    .filter((feature) => feature !== "core-features-and-limits")
                    .sort(),
                limits,
            };
            return device;
        };
    });

    await page.goto("/");
    await page.waitForFunction(() => typeof window.__releaseLoading === "function");
    await releasing(
        async () => {
            const held = await page.evaluate((selector) => {
                const frame = document.querySelector<HTMLElement>("#frame")!;
                const line = document.querySelector<HTMLElement>(".description")!;
                const overlay = document.querySelector<HTMLElement>(selector);
                const rect = overlay?.getBoundingClientRect();
                const bounds = frame.getBoundingClientRect();
                const lineRect = line.getBoundingClientRect();
                return {
                    owned: overlay?.parentElement === frame,
                    inside:
                        rect !== undefined &&
                        rect.left >= bounds.left &&
                        rect.top >= bounds.top &&
                        rect.right <= bounds.right &&
                        rect.bottom <= bounds.bottom,
                    opaqueDark:
                        overlay !== null &&
                        getComputedStyle(overlay).backgroundColor === "rgb(20, 18, 16)",
                    lineHit:
                        document.elementFromPoint(
                            lineRect.left + lineRect.width / 2,
                            lineRect.top + lineRect.height / 2,
                        ) === line,
                    canvasVisible: document.querySelector("#scene")!.classList.contains("visible"),
                    canvasOpacity: getComputedStyle(document.querySelector("#scene")!).opacity,
                    progressBar: [...(overlay?.querySelectorAll<HTMLElement>("div") ?? [])].some(
                        (node) =>
                            node.style.height === "100%" &&
                            node.parentElement?.style.overflow === "hidden",
                    ),
                    device: window.__subjectDeviceRecord,
                };
            }, overlaySelector);
            expect(held.owned, "loading overlay is a child of its scene frame").toBe(true);
            expect(held.inside, "loading overlay remains inside the scene frame").toBe(true);
            expect(held.opaqueDark, "default dark overlay owns an opaque brand background").toBe(
                true,
            );
            expect(held.lineHit, "the description line remains the hit target during loading").toBe(
                true,
            );
            expect(held.canvasVisible, "the canvas is not revealed before its first frame").toBe(
                false,
            );
            expect(held.canvasOpacity, "the canvas remains hidden before its first frame").toBe(
                "0",
            );
            expect(held.progressBar, "the real page owns a progress bar during loading").toBe(true);
            expect(held.device?.identity, "the subject adapter identity was recorded").toBeTruthy();
            const floorFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"];
            expect(held.device?.requestedFeatures).toEqual(floorFeatures);
            expect(held.device?.features).toEqual(floorFeatures);
            expect(held.device?.requestedLimits).not.toHaveProperty(
                "maxStorageBuffersPerShaderStage",
            );
            expect(held.device?.limits.maxStorageBuffersPerShaderStage).toBe(8);
            console.log(`Browser WebGPU floor: ${JSON.stringify(held.device)}`);

            // Chromium 153.0.8010.12 can composite an opacity-0 WebGPU canvas into Linux headless screenshots:
            // this measured 2.18% frame background locally and 2.14% in CI. Omit only the canvas in this
            // page-composition shot; its hidden class and opacity are checked separately above.
            const loadingScreenshot = await page.screenshot({
                style: "#scene { visibility: hidden !important; }",
            });
            const loadingPixels = await inspectPageScreenshot(
                page,
                loadingScreenshot.toString("base64"),
            );
            expect(
                loadingPixels.inkPixels,
                "the held screenshot contains more than 20 ink-coloured description pixels",
            ).toBeGreaterThan(20);
            expect(
                loadingPixels.progressPixels,
                "the screenshot shows the loading bar over the frame void",
            ).toBeGreaterThan(100);
            expect(
                loadingPixels.frameBackgroundFraction,
                "the default loading background and adapter-notice surface fill at least 97 percent of the frame",
            ).toBeGreaterThanOrEqual(0.97);
            expect(
                loadingPixels.pageBackgroundError,
                "the page background matches --bg outside the scene",
            ).toBeLessThanOrEqual(2);
            expect(
                loadingPixels.adjacentBackgroundError,
                "the page immediately outside the frame remains --bg",
            ).toBeLessThanOrEqual(2);
        },
        () => page.evaluate(() => window.__releaseLoading?.()),
    );

    await page.waitForFunction((selector) => !document.querySelector(selector), overlaySelector);
    const widths = await page.evaluate(() => window.__progressWidths ?? []);
    expect(
        widths.some((width) => width > 0 && width < 100),
        "loading progress has an intermediate width",
    ).toBe(true);
    expect(widths.includes(100), "loading progress reaches 100 percent").toBe(true);
});

test("the canvas remains hidden after loading cleanup until its first submitted frame", async ({
    page,
}) => {
    await page.addInitScript(() => {
        const originalFence = GPUQueue.prototype.onSubmittedWorkDone;
        let fences = 0;
        GPUQueue.prototype.onSubmittedWorkDone = function (this: GPUQueue): Promise<void> {
            const fence = originalFence.call(this) as Promise<void>;
            fences++;
            if (fences !== 2) return fence;
            window.__firstFrameFenceReady = true;
            return new Promise<void>((resolve, reject) => {
                window.__releaseFirstFrameFence = () => {
                    fence.then(resolve, reject);
                };
            });
        } as typeof GPUQueue.prototype.onSubmittedWorkDone;

        const observer = new MutationObserver(() => {
            const canvas = document.querySelector<HTMLCanvasElement>("#scene");
            if (!canvas || canvas.dataset.revealHeld) return;
            canvas.dataset.revealHeld = "true";
            const add = canvas.classList.add.bind(canvas.classList);
            canvas.classList.add = (...tokens: string[]) => {
                if (tokens.includes("visible")) {
                    window.__revealBeforeFence = !window.__firstFrameFenceReady;
                    window.__releaseReveal = () => add(...tokens);
                    return;
                }
                add(...tokens);
            };
        });
        observer.observe(document, { childList: true, subtree: true });
    });

    await page.goto("/");
    await page.waitForFunction(() => window.__firstFrameFenceReady || window.__revealBeforeFence);
    await releasing(
        async () => {
            const revealBeforeFence = await page.evaluate(
                () => window.__revealBeforeFence ?? false,
            );
            expect(
                revealBeforeFence,
                "the canvas is not revealed before the submitted-frame fence",
            ).toBe(false);
            await page.waitForFunction(() => typeof window.__releaseFirstFrameFence === "function");
            await page.waitForFunction(
                (selector) => !document.querySelector(selector),
                overlaySelector,
            );
            await page.evaluate(() => window.__releaseFirstFrameFence?.());
            await page.waitForFunction(() => typeof window.__releaseReveal === "function");
            const before = await page.locator("#scene").evaluate((canvas) => ({
                visible: canvas.classList.contains("visible"),
                opacity: getComputedStyle(canvas).opacity,
            }));
            const beforeFrameScreenshot = await page.screenshot();
            const beforeFramePixels = await inspectPageScreenshot(
                page,
                beforeFrameScreenshot.toString("base64"),
            );
            expect(
                before.visible,
                "the canvas remains hidden before its first submitted frame",
            ).toBe(false);
            expect(
                before.opacity,
                "the cleared page shows the frame background before reveal",
            ).toBe("0");
            expect(
                beforeFramePixels.overlayPresent,
                "loading cleanup removes its frame overlay",
            ).toBe(false);
            expect(
                beforeFramePixels.frameBackgroundFraction,
                "the canvas stays hidden over frame --bg2 before reveal",
            ).toBeGreaterThanOrEqual(0.99);
        },
        () =>
            page.evaluate(() => {
                window.__releaseFirstFrameFence?.();
                window.__releaseReveal?.();
            }),
    );
    await page.locator("#scene.visible").waitFor();
});

test("the first and later scene frames match the projected unit cube and remain stable", async ({
    page,
}) => {
    await waitForScene(page);
    await page.addStyleTag({
        content: "#frame { width: 1280px !important; height: 720px !important; }",
    });
    await expect
        .poll(() =>
            page.locator("#scene").evaluate((canvas) => (canvas as HTMLCanvasElement).width),
        )
        .toBe(1280);
    await expect
        .poll(() =>
            page.locator("#scene").evaluate((canvas) => (canvas as HTMLCanvasElement).height),
        )
        .toBe(720);
    await page.waitForTimeout(200);
    const bounds = await page.locator("#scene").boundingBox();
    expect(bounds, "the scene canvas has a page rectangle").not.toBeNull();
    await page.addScriptTag({ path: CAPTURE_ENTRY });
    await page.waitForFunction(() => typeof window.shallotCaptureFrame === "function");
    const first = await inspectScene(page, "first");
    await page.waitForTimeout(100);
    const later = await inspectScene(page, "later");
    console.info("background seam", {
        first: first.backgroundDeltaE,
        later: later.backgroundDeltaE,
    });
    expect(first.width, "captureFrame matches the real canvas width").toBe(
        Math.round(bounds!.width),
    );
    expect(first.height, "captureFrame matches the real canvas height").toBe(
        Math.round(bounds!.height),
    );
    expect(
        first.geometry,
        "the first revealed scene has the orbit-projected unit-cube population and centered bounds",
    ).toBe(true);
    expect(
        first.backgroundDeltaE,
        "the first clear's mean colour is within CSS Color 4's gamut-mapping JND (ΔE_OK 0.02) of the frame",
    ).toBeLessThanOrEqual(0.02);

    expect(
        later.geometry,
        "the later stepped scene has the orbit-projected unit-cube population and centered bounds",
    ).toBe(true);
    const comparison = await compareSceneFrames(page, "first", "later");
    expect(
        comparison.meanError,
        "the first scene region matches the later stepped scene within 0.25 mean RGB",
    ).toBeLessThanOrEqual(0.25);
    expect(
        comparison.changedFraction,
        "the first scene region changes no more than 0.1% of pixels beyond 2 RGB levels",
    ).toBeLessThanOrEqual(0.001);
    expect(
        later.backgroundDeltaE,
        "the later clear's mean colour is within CSS Color 4's gamut-mapping JND (ΔE_OK 0.02) of the frame",
    ).toBeLessThanOrEqual(0.02);
});

test("the description and scene frame stay fitted in six responsive viewports", async ({
    page,
}) => {
    await waitForScene(page);
    const viewports = [
        { width: 390, height: 844 },
        { width: 844, height: 390 },
        { width: 820, height: 1180 },
        { width: 1280, height: 720 },
        { width: 1440, height: 900 },
        { width: 2560, height: 1080 },
    ];
    for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const evidence = process.env["SHALLOT_STARTUP_EVIDENCE"];
        if (evidence && [390, 820, 1280].includes(viewport.width)) {
            await page.waitForTimeout(200);
            await page.screenshot({
                path: join(evidence, `scene-${viewport.width}x${viewport.height}.png`),
            });
        }
        const layout = await page.evaluate(() => {
            const line = document
                .querySelector<HTMLElement>(".description")!
                .getBoundingClientRect();
            const frame = document.querySelector<HTMLElement>("#frame")!.getBoundingClientRect();
            return {
                width: window.innerWidth,
                height: window.innerHeight,
                textEdges: ["h1", ".description"].map((selector) => {
                    const bounds = document.querySelector(selector)!.getBoundingClientRect();
                    return { left: bounds.left, width: bounds.width };
                }),
                lineTop: line.top,
                lineBottom: line.bottom,
                frameLeft: frame.left,
                frameTop: frame.top,
                frameRight: frame.right,
                frameBottom: frame.bottom,
                frameWidth: frame.width,
                frameHeight: frame.height,
                topMargin: document.querySelector("h1")!.getBoundingClientRect().top,
                bottomMargin: window.innerHeight - frame.bottom,
                leftPadding: frame.left,
                rightPadding: window.innerWidth - frame.right,
                documentWidth: document.documentElement.scrollWidth,
                bodyWidth: document.body.scrollWidth,
            };
        });
        for (const edge of layout.textEdges) {
            expect(
                Math.abs(edge.left - layout.frameLeft),
                `text and frame share their left edge at ${viewport.width}x${viewport.height}`,
            ).toBeLessThanOrEqual(1);
            expect(
                Math.abs(edge.width - layout.frameWidth),
                `text and frame share their width at ${viewport.width}x${viewport.height}`,
            ).toBeLessThanOrEqual(1);
        }
        expect(
            layout.lineBottom,
            `description stays above the frame at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(layout.frameTop);
        expect(
            layout.frameLeft,
            `frame stays within the left viewport edge at ${viewport.width}x${viewport.height}`,
        ).toBeGreaterThanOrEqual(0);
        expect(
            layout.frameRight,
            `frame stays within the right viewport edge at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(layout.width);
        expect(
            layout.frameBottom,
            `frame stays within the bottom viewport edge at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(layout.height);
        expect(
            Math.abs(layout.frameWidth - (layout.frameHeight * 16) / 9),
            `frame keeps its 16:9 ratio at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(1);
        expect(
            layout.documentWidth,
            `document has no horizontal overflow at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(layout.width);
        expect(
            layout.bodyWidth,
            `body has no horizontal overflow at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(layout.width);
        expect(
            Math.abs(layout.topMargin - layout.bottomMargin),
            `the description and frame group stays vertically balanced at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(2);
        if (viewport.width <= 480) {
            expect(
                layout.leftPadding,
                `the frame keeps mobile left padding at ${viewport.width}x${viewport.height}`,
            ).toBeGreaterThanOrEqual(16);
            expect(
                layout.rightPadding,
                `the frame keeps mobile right padding at ${viewport.width}x${viewport.height}`,
            ).toBeGreaterThanOrEqual(16);
        }
        expect(
            layout.frameWidth,
            `frame stays below its maximum width at ${viewport.width}x${viewport.height}`,
        ).toBeLessThanOrEqual(880);
    }
});
