import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "playwright/test";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const CAPTURE_ENTRY = resolve(ROOT, ".artifacts/capture-page.js");
execFileSync("bun", ["run", "scripts/build-capture-page.ts"], { cwd: ROOT });

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
        __subjectAdapterIdentity?: string;
    }
}

const overlaySelector = "#frame > div[style*='z-index: 10000']";

async function openPage(page: Page): Promise<void> {
    await page.goto("/");
    await expect(page.locator("#frame")).toBeVisible();
}

async function waitForScene(page: Page): Promise<void> {
    await openPage(page);
    await expect(page.locator("#scene")).toHaveClass(/visible/);
}

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
                getComputedStyle(document.querySelector<HTMLElement>("#frame")!).backgroundColor,
            );
            const inkColor = channels(
                getComputedStyle(document.querySelector(".description")!).color,
            );
            const progressColor = progress
                ? channels(getComputedStyle(progress).backgroundColor)
                : [];
            const count = (rect: DOMRect, color: number[], tolerance: number) => {
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
                        if (matches) pixels++;
                    }
                }
                return pixels;
            };
            const inkPixels = count(description, inkColor, 24);
            const progressPixels = progressColor.length ? count(frame, progressColor, 24) : 0;
            const framePixels = count(canvas, frameColor, 2);
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

async function inspectScene(page: Page, frameKey: string) {
    return page.evaluate(async (key) => {
        if (!window.shallotCaptureFrame) throw new Error("captureFrame is not installed");
        const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
        const capture = await window.shallotCaptureFrame(canvas);
        (window.__sceneFrames ??= {})[key] = capture.rgba;
        const surface = { width: capture.width, height: capture.height };
        const rgba = capture.rgba;
        const clear = [rgba[0]!, rgba[1]!, rgba[2]!];
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
                    Math.max(
                        Math.abs(rgba[offset]! - clear[0]!),
                        Math.abs(rgba[offset + 1]! - clear[1]!),
                        Math.abs(rgba[offset + 2]! - clear[2]!),
                    ) > 24;
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
        let maxBackgroundError = 0;
        let compared = 0;
        for (let y = 0; y < surface.height; y++) {
            for (let x = 0; x < surface.width; x++) {
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
                maxBackgroundError = Math.max(
                    maxBackgroundError,
                    Math.abs(rgba[offset]! - frameColor[0]!),
                    Math.abs(rgba[offset + 1]! - frameColor[1]!),
                    Math.abs(rgba[offset + 2]! - frameColor[2]!),
                );
                compared++;
            }
        }
        return {
            width: surface.width,
            height: surface.height,
            scenePixels,
            expectedPopulation,
            geometry,
            maxBackgroundError,
            compared,
        };
    }, frameKey);
}

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

test("loading progress stays in its owned frame and reaches the bar; its description remains hit-testable", async ({
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
            window.__subjectAdapterIdentity =
                [info.vendor, info.architecture, info.device, info.description]
                    .filter((value) => typeof value === "string" && value.trim() !== "")
                    .join(" ") || "unidentified";
            return original.call(this, descriptor);
        };
    });

    await page.goto("/");
    await page.waitForFunction(() => typeof window.__releaseLoading === "function");
    try {
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
                transparent:
                    overlay !== null &&
                    getComputedStyle(overlay).backgroundColor === "rgba(0, 0, 0, 0)",
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
                adapterIdentity: window.__subjectAdapterIdentity,
            };
        }, overlaySelector);
        expect(held.owned, "loading overlay is a child of its scene frame").toBe(true);
        expect(held.inside, "loading overlay remains inside the scene frame").toBe(true);
        expect(held.transparent, "minimal loading overlay remains transparent").toBe(true);
        expect(held.lineHit, "the description line remains the hit target during loading").toBe(
            true,
        );
        expect(held.canvasVisible, "the canvas is not revealed before its first frame").toBe(false);
        expect(held.canvasOpacity, "the canvas remains hidden before its first frame").toBe("0");
        expect(held.progressBar, "the real page owns a progress bar during loading").toBe(true);
        expect(held.adapterIdentity, "the subject adapter identity was recorded").toBeTruthy();
        console.log(`Browser WebGPU adapter: ${held.adapterIdentity}`);
        expect(held.adapterIdentity, "the subject adapter is present rather than absent").not.toBe(
            "none",
        );

        const loadingScreenshot = await page.screenshot();
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
            "the hidden canvas leaves at least 97 percent of its frame at --bg2",
        ).toBeGreaterThanOrEqual(0.97);
        expect(
            loadingPixels.pageBackgroundError,
            "the page background matches --bg outside the scene",
        ).toBeLessThanOrEqual(2);
        expect(
            loadingPixels.adjacentBackgroundError,
            "the page immediately outside the frame remains --bg",
        ).toBeLessThanOrEqual(2);
    } finally {
        await page.evaluate(() => window.__releaseLoading?.());
    }

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
    try {
        const revealBeforeFence = await page.evaluate(() => window.__revealBeforeFence ?? false);
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
        expect(before.visible, "the canvas remains hidden before its first submitted frame").toBe(
            false,
        );
        expect(before.opacity, "the cleared page shows the frame background before reveal").toBe(
            "0",
        );
        expect(beforeFramePixels.overlayPresent, "loading cleanup removes its frame overlay").toBe(
            false,
        );
        expect(
            beforeFramePixels.frameBackgroundFraction,
            "the canvas stays hidden over frame --bg2 before reveal",
        ).toBeGreaterThanOrEqual(0.99);
    } finally {
        await page.evaluate(() => {
            window.__releaseFirstFrameFence?.();
            window.__releaseReveal?.();
        });
    }
    await expect(page.locator("#scene")).toHaveClass(/visible/);
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
    await page.addScriptTag({ path: CAPTURE_ENTRY, type: "module" });
    const first = await inspectScene(page, "first");
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
        first.maxBackgroundError,
        "outside the projected cube, the revealed scene matches frame --bg2 within 2 RGB levels per channel",
    ).toBeLessThanOrEqual(2);

    await page.waitForTimeout(100);
    const later = await inspectScene(page, "later");
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
        later.maxBackgroundError,
        "the later revealed scene matches frame --bg2 outside the projected cube",
    ).toBeLessThanOrEqual(2);
});

test("the description and scene frame stay fitted in six responsive viewports", async ({
    page,
}) => {
    await openPage(page);
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
        const layout = await page.evaluate(() => {
            const line = document
                .querySelector<HTMLElement>(".description")!
                .getBoundingClientRect();
            const frame = document.querySelector<HTMLElement>("#frame")!.getBoundingClientRect();
            return {
                width: window.innerWidth,
                height: window.innerHeight,
                lineTop: line.top,
                lineBottom: line.bottom,
                frameLeft: frame.left,
                frameTop: frame.top,
                frameRight: frame.right,
                frameBottom: frame.bottom,
                frameWidth: frame.width,
                frameHeight: frame.height,
                topMargin: line.top,
                bottomMargin: window.innerHeight - frame.bottom,
                leftPadding: frame.left,
                rightPadding: window.innerWidth - frame.right,
                documentWidth: document.documentElement.scrollWidth,
                bodyWidth: document.body.scrollWidth,
            };
        });
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
