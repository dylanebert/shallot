import { build, Compute, type Loading, minimalDark } from "@dylanebert/shallot";
import { mountHost } from "../src/host";
import { revealAfterFirstFrame } from "../src/reveal";
import { SCENE } from "../src/scene";

type Progress = { value: number; width: string };
type Adapter = { class: string; identity: string };
type Snapshot = {
    frameOwnsOverlay: boolean;
    transparentOverlay: boolean;
    posterVisible: boolean;
    canvasHidden: boolean;
    overlayPresent: boolean;
    progress: Progress[];
    pageActionResult: string;
    cleaned: boolean;
    adapter: Adapter | null;
};
type Bounds = { left: number; top: number; right: number; bottom: number };
type ScreenshotObservation = {
    posterTitlePixels: number;
    sceneColorPixels: number;
    sceneBounds: Bounds | null;
    canvas: { x: number; y: number; width: number; height: number };
    canvasPixels: Uint8Array;
    progressColorPixelsInFrame: number;
    progressColorPixelsOutsideFrame: number;
    pageBackground: [number, number, number];
};

// Mirrors the static fixture: a unit cube at the origin, a 60° camera at z=4.5, and a 1280×720 target.
// Its nearest face is 4 world units away, projecting to about 107 CSS pixels square in this frame.
const CAMERA_FOV = 60;
const CAMERA_DISTANCE = 4.5;
const PART_HALF_EXTENT = 0.5;
const RENDER_WIDTH = 1280;
const RENDER_HEIGHT = 720;
const FRAME_DIFFERENCE_MEAN_LIMIT = 0.25;
const FRAME_DIFFERENCE_PIXEL_LIMIT = 0.001;
const CHANNEL_TOLERANCE = 2;
const BOX_POPULATION_MIN_FACTOR = 0.2;
const BOX_POPULATION_MAX_FACTOR = 1.5;
const BOX_SIZE_MIN_FACTOR = 0.72;
const BOX_SIZE_MAX_FACTOR = 1.28;
const BOX_CENTER_TOLERANCE_FACTOR = 0.12;

function sceneGeometry(image: ScreenshotObservation | undefined) {
    if (!image)
        return {
            ok: false,
            population: 0,
            expectedPopulation: 0,
            width: 0,
            height: 0,
            expectedWidth: 0,
            expectedHeight: 0,
            centerOffsetX: 0,
            centerOffsetY: 0,
            boundsLeft: -1,
            boundsTop: -1,
            boundsRight: -1,
            boundsBottom: -1,
        };
    const focalPixels = RENDER_HEIGHT / (2 * Math.tan((CAMERA_FOV * Math.PI) / 360));
    const nearDistance = CAMERA_DISTANCE - PART_HALF_EXTENT;
    const expectedWidth =
        ((focalPixels * (2 * PART_HALF_EXTENT)) / nearDistance) *
        (image.canvas.width / RENDER_WIDTH);
    const expectedHeight =
        ((focalPixels * (2 * PART_HALF_EXTENT)) / nearDistance) *
        (image.canvas.height / RENDER_HEIGHT);
    const expectedPopulation = expectedWidth * expectedHeight;
    const bounds = image.sceneBounds;
    const width = bounds ? bounds.right - bounds.left + 1 : 0;
    const height = bounds ? bounds.bottom - bounds.top + 1 : 0;
    const centerOffsetX = bounds
        ? Math.abs((bounds.left + bounds.right) / 2 - (image.canvas.width - 1) / 2)
        : image.canvas.width;
    const centerOffsetY = bounds
        ? Math.abs((bounds.top + bounds.bottom) / 2 - (image.canvas.height - 1) / 2)
        : image.canvas.height;
    return {
        ok: Boolean(
            bounds &&
                image.sceneColorPixels >= expectedPopulation * BOX_POPULATION_MIN_FACTOR &&
                image.sceneColorPixels <= expectedPopulation * BOX_POPULATION_MAX_FACTOR &&
                width >= expectedWidth * BOX_SIZE_MIN_FACTOR &&
                width <= expectedWidth * BOX_SIZE_MAX_FACTOR &&
                height >= expectedHeight * BOX_SIZE_MIN_FACTOR &&
                height <= expectedHeight * BOX_SIZE_MAX_FACTOR &&
                centerOffsetX <= expectedWidth * BOX_CENTER_TOLERANCE_FACTOR &&
                centerOffsetY <= expectedHeight * BOX_CENTER_TOLERANCE_FACTOR,
        ),
        population: image.sceneColorPixels,
        expectedPopulation,
        width,
        height,
        expectedWidth,
        expectedHeight,
        centerOffsetX,
        centerOffsetY,
        boundsLeft: bounds?.left ?? -1,
        boundsTop: bounds?.top ?? -1,
        boundsRight: bounds?.right ?? -1,
        boundsBottom: bounds?.bottom ?? -1,
    };
}

function compareSceneFrames(
    first: ScreenshotObservation | undefined,
    later: ScreenshotObservation | undefined,
): { ok: boolean; meanAbsoluteChannelError: number; changedPixelFraction: number } {
    if (
        !first ||
        !later ||
        first.canvas.width !== later.canvas.width ||
        first.canvas.height !== later.canvas.height ||
        first.canvasPixels.length !== later.canvasPixels.length
    )
        return { ok: false, meanAbsoluteChannelError: 255, changedPixelFraction: 1 };
    let totalError = 0;
    let changedPixels = 0;
    const pixelCount = first.canvasPixels.length / 3;
    for (let i = 0; i < first.canvasPixels.length; i += 3) {
        let largestChannelError = 0;
        for (let channel = 0; channel < 3; channel++) {
            const error = Math.abs(
                first.canvasPixels[i + channel]! - later.canvasPixels[i + channel]!,
            );
            totalError += error;
            largestChannelError = Math.max(largestChannelError, error);
        }
        if (largestChannelError > CHANNEL_TOLERANCE) changedPixels++;
    }
    const meanAbsoluteChannelError = totalError / first.canvasPixels.length;
    const changedPixelFraction = changedPixels / pixelCount;
    return {
        ok:
            meanAbsoluteChannelError <= FRAME_DIFFERENCE_MEAN_LIMIT &&
            changedPixelFraction <= FRAME_DIFFERENCE_PIXEL_LIMIT,
        meanAbsoluteChannelError,
        changedPixelFraction,
    };
}

function deferred(): { promise: Promise<void>; resolve(): void } {
    let resolve!: () => void;
    return { promise: new Promise<void>((done) => (resolve = done)), resolve: () => resolve() };
}

const host = mountHost();
const firstFrameReady = deferred();
const revealObserver = new MutationObserver(() => {
    if (host.poster.hidden && host.canvas.classList.contains("visible")) firstFrameReady.resolve();
});
revealObserver.observe(host.frame, {
    attributes: true,
    attributeFilter: ["class", "hidden"],
    subtree: true,
});
const completionEntered = deferred();
const releaseBuild = deferred();
const buildCleaned = deferred();
const buildReturned = deferred();
const buildFailed = deferred();
const barProgress: Progress[] = [];
const screenshots = new Map<string, ScreenshotObservation>();
const observedStates = new Map<string, Snapshot>();
const screen = minimalDark({ container: host.frame });
let overlay: HTMLElement | null = null;
let app: Awaited<ReturnType<typeof build>> | null = null;
let loadingAtCompletion: Snapshot | undefined;
let buildError: unknown;
let began = false;
let cleaned = false;

function progressElement(): HTMLElement | null {
    return overlay?.firstElementChild?.firstElementChild?.firstElementChild as HTMLElement | null;
}

function snapshot(): Snapshot {
    const canvasStyle = getComputedStyle(host.canvas);
    return {
        frameOwnsOverlay: overlay?.parentElement === host.frame,
        transparentOverlay:
            overlay !== null && getComputedStyle(overlay).backgroundColor === "rgba(0, 0, 0, 0)",
        posterVisible: !host.poster.hidden,
        canvasHidden: !host.canvas.classList.contains("visible") && canvasStyle.opacity === "0",
        overlayPresent: overlay?.isConnected === true,
        progress: [...barProgress],
        pageActionResult:
            document.querySelector<HTMLElement>("#page-action-result")?.textContent ?? "",
        cleaned,
        adapter: Compute.adapter
            ? { class: Compute.adapter.class, identity: Compute.adapter.identity }
            : null,
    };
}

const loading: Loading = {
    show() {
        const cleanup = screen.show();
        overlay = host.frame.lastElementChild as HTMLElement | null;
        return () => {
            cleanup?.();
            cleaned = true;
            buildCleaned.resolve();
        };
    },
    update(value) {
        screen.update(value);
        barProgress.push({ value, width: progressElement()?.style.width ?? "" });
    },
    complete() {
        loadingAtCompletion = snapshot();
        completionEntered.resolve();
        return releaseBuild.promise;
    },
    notice(verdict) {
        screen.notice?.(verdict);
    },
    error(error) {
        screen.error?.(error);
    },
};

function waitForBuild(): Promise<void> {
    return Promise.race([
        buildReturned.promise,
        buildFailed.promise.then(() => {
            throw new Error(`the example build failed: ${String(buildError)}`);
        }),
    ]);
}

async function inspectScreenshot(encoded: string): Promise<ScreenshotObservation> {
    const binary = atob(encoded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes.buffer], { type: "image/png" }));
    const image = document.createElement("canvas");
    image.width = bitmap.width;
    image.height = bitmap.height;
    const context = image.getContext("2d");
    if (!context) throw new Error("could not inspect the page screenshot");
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const pixels = context.getImageData(0, 0, image.width, image.height).data;
    const rect = (element: Element) => {
        const box = element.getBoundingClientRect();
        return {
            x: Math.round(box.left + window.scrollX),
            y: Math.round(box.top + window.scrollY),
            width: Math.round(box.width),
            height: Math.round(box.height),
        };
    };
    const posterTitle = rect(host.poster.querySelector(".poster-title")!);
    const frame = rect(host.frame);
    const canvas = rect(host.canvas);
    const inside = (
        box: ReturnType<typeof rect>,
        predicate: (r: number, g: number, b: number) => boolean,
    ) => {
        let count = 0;
        const left = Math.max(0, box.x);
        const top = Math.max(0, box.y);
        const right = Math.min(image.width, box.x + box.width);
        const bottom = Math.min(image.height, box.y + box.height);
        for (let y = top; y < bottom; y++) {
            for (let x = left; x < right; x++) {
                const index = (y * image.width + x) * 4;
                if (predicate(pixels[index]!, pixels[index + 1]!, pixels[index + 2]!)) count++;
            }
        }
        return count;
    };
    const posterTitlePixels = inside(
        posterTitle,
        (r, g, b) => r > 150 && g > 155 && b > 145 && Math.max(r, g, b) - Math.min(r, g, b) < 45,
    );
    const isSceneColor = (r: number, g: number, b: number) =>
        r > 90 && r > g * 1.25 && g > b * 1.05;
    const canvasPixels = new Uint8Array(canvas.width * canvas.height * 3);
    let sceneColorPixels = 0;
    let left = canvas.width;
    let top = canvas.height;
    let right = -1;
    let bottom = -1;
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const source = ((canvas.y + y) * image.width + canvas.x + x) * 4;
            const target = (y * canvas.width + x) * 3;
            const r = pixels[source]!;
            const g = pixels[source + 1]!;
            const b = pixels[source + 2]!;
            canvasPixels[target] = r;
            canvasPixels[target + 1] = g;
            canvasPixels[target + 2] = b;
            if (!isSceneColor(r, g, b)) continue;
            sceneColorPixels++;
            left = Math.min(left, x);
            top = Math.min(top, y);
            right = Math.max(right, x);
            bottom = Math.max(bottom, y);
        }
    }
    const sceneBounds = sceneColorPixels === 0 ? null : { left, top, right, bottom };
    const isProgressColor = (r: number, g: number, b: number) =>
        Math.abs(r - 212) < 24 && Math.abs(g - 149) < 24 && Math.abs(b - 96) < 24;
    const progressColorPixelsInFrame = inside(frame, isProgressColor);
    const progressColorPixelsOutsideFrame = (() => {
        let count = 0;
        for (let y = 0; y < image.height; y++) {
            for (let x = 0; x < image.width; x++) {
                if (
                    x >= frame.x &&
                    x < frame.x + frame.width &&
                    y >= frame.y &&
                    y < frame.y + frame.height
                )
                    continue;
                const index = (y * image.width + x) * 4;
                if (isProgressColor(pixels[index]!, pixels[index + 1]!, pixels[index + 2]!))
                    count++;
            }
        }
        return count;
    })();
    const pageBackground = [
        pixels[(8 * image.width + 8) * 4]!,
        pixels[(8 * image.width + 8) * 4 + 1]!,
        pixels[(8 * image.width + 8) * 4 + 2]!,
    ] as [number, number, number];
    return {
        posterTitlePixels,
        sceneColorPixels,
        sceneBounds,
        canvas,
        canvasPixels,
        progressColorPixelsInFrame,
        progressColorPixelsOutsideFrame,
        pageBackground,
    };
}

const checkWindow = window as unknown as {
    __loadingCheck?: {
        begin(): void;
        waitForCompletion(): Promise<void>;
        snapshot(): Snapshot;
        releaseBuild(): void;
        waitForCleanup(): Promise<void>;
        waitForBuild(): Promise<void>;
        stepFirstFrame(): Promise<void>;
        stepLaterFrame(): Promise<void>;
        recordSnapshot(name: string): void;
        recordScreenshot(name: string, encoded: string): Promise<void>;
    };
    __harness?: {
        ready: boolean;
        run(): Promise<{
            ok: boolean;
            checks: { name: string; ok: boolean; detail?: string; data?: Record<string, number> }[];
            hardware?: string;
        }>;
    };
};

checkWindow.__loadingCheck = {
    begin() {
        if (began) throw new Error("the example check already started the build");
        began = true;
        void build({
            plugins: [revealAfterFirstFrame(host)],
            scene: SCENE,
            loading,
            pixelRatio: 1,
        }).then(
            (built) => {
                app = built;
                buildReturned.resolve();
            },
            (error: unknown) => {
                buildError = error;
                buildFailed.resolve();
                host.fail(error);
            },
        );
    },
    waitForCompletion() {
        return Promise.race([
            completionEntered.promise,
            buildFailed.promise.then(() => {
                throw new Error(
                    `the example never reached loading completion: ${String(buildError)}`,
                );
            }),
        ]);
    },
    snapshot,
    releaseBuild: () => releaseBuild.resolve(),
    waitForCleanup: () => buildCleaned.promise,
    waitForBuild,
    recordSnapshot(name) {
        observedStates.set(name, snapshot());
    },
    async stepFirstFrame() {
        await waitForBuild();
        if (!app) throw new Error("the example build returned no app");
        app.state.step(1 / 60);
        await firstFrameReady.promise;
    },
    async stepLaterFrame() {
        await firstFrameReady.promise;
        if (!app) throw new Error("the example build returned no app");
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        app.state.step(1 / 60);
        await Compute.device.queue.onSubmittedWorkDone();
    },
    async recordScreenshot(name, encoded) {
        screenshots.set(name, await inspectScreenshot(encoded));
    },
};

checkWindow.__harness = {
    ready: true,
    async run() {
        const held = loadingAtCompletion;
        const beforeFrame = observedStates.get("before-frame");
        const firstFrame = observedStates.get("first-frame");
        const heldImage = screenshots.get("held");
        const beforeFrameImage = screenshots.get("before-frame");
        const firstFrameImage = screenshots.get("first-frame");
        const laterFrameImage = screenshots.get("later-frame");
        const firstGeometry = sceneGeometry(firstFrameImage);
        const laterGeometry = sceneGeometry(laterFrameImage);
        const frameComparison = compareSceneFrames(firstFrameImage, laterFrameImage);
        const checks: {
            name: string;
            ok: boolean;
            data?: Record<string, number>;
        }[] = [
            {
                name: "build completion kept transparent loading inside the frame",
                ok: Boolean(
                    held?.frameOwnsOverlay && held.transparentOverlay && held.overlayPresent,
                ),
            },
            {
                name: "build progress reached completion and drove an intermediate frame-local bar",
                ok: Boolean(
                    held?.progress.some(
                        (step) =>
                            step.value > 0 &&
                            step.value < 1 &&
                            step.width === `${step.value * 100}%`,
                    ) && held.progress.some((step) => step.value === 1 && step.width === "100%"),
                ),
            },
            {
                name: "the screenshot confines progress to the frame and leaves the page background intact",
                ok: Boolean(
                    heldImage &&
                        heldImage.progressColorPixelsInFrame > 100 &&
                        heldImage.progressColorPixelsOutsideFrame === 0 &&
                        heldImage.pageBackground.every(
                            (channel, index) => Math.abs(channel - [244, 242, 237][index]!) <= 12,
                        ),
                ),
                data: {
                    progressPixelsInFrame: heldImage?.progressColorPixelsInFrame ?? 0,
                    progressPixelsOutsideFrame: heldImage?.progressColorPixelsOutsideFrame ?? 0,
                },
            },
            {
                name: "the host poster stayed visible through build cleanup until the first stepped frame",
                ok: Boolean(
                    held?.posterVisible &&
                        held.canvasHidden &&
                        beforeFrame?.posterVisible &&
                        beforeFrame.canvasHidden &&
                        beforeFrame.cleaned &&
                        beforeFrameImage &&
                        beforeFrameImage.posterTitlePixels > 20,
                ),
                data: { posterTitlePixels: beforeFrameImage?.posterTitlePixels ?? 0 },
            },
            {
                name: "the first revealed scene has the projected unit-cube population and centered bounds",
                ok: Boolean(
                    firstFrame &&
                        !firstFrame.posterVisible &&
                        !firstFrame.canvasHidden &&
                        firstGeometry.ok,
                ),
                data: {
                    scenePixels: firstGeometry.population,
                    expectedScenePixels: firstGeometry.expectedPopulation,
                    boundsLeft: firstGeometry.boundsLeft,
                    boundsTop: firstGeometry.boundsTop,
                    boundsRight: firstGeometry.boundsRight,
                    boundsBottom: firstGeometry.boundsBottom,
                    expectedWidth: firstGeometry.expectedWidth,
                    expectedHeight: firstGeometry.expectedHeight,
                    centerOffsetX: firstGeometry.centerOffsetX,
                    centerOffsetY: firstGeometry.centerOffsetY,
                },
            },
            {
                name: "the later stepped scene has the projected unit-cube population and centered bounds",
                ok: laterGeometry.ok,
                data: {
                    scenePixels: laterGeometry.population,
                    expectedScenePixels: laterGeometry.expectedPopulation,
                    boundsLeft: laterGeometry.boundsLeft,
                    boundsTop: laterGeometry.boundsTop,
                    boundsRight: laterGeometry.boundsRight,
                    boundsBottom: laterGeometry.boundsBottom,
                    expectedWidth: laterGeometry.expectedWidth,
                    expectedHeight: laterGeometry.expectedHeight,
                    centerOffsetX: laterGeometry.centerOffsetX,
                    centerOffsetY: laterGeometry.centerOffsetY,
                },
            },
            {
                name: "the first scene region matches the later stepped scene within 0.25 mean RGB and 0.1% changed-pixel tolerance",
                ok: frameComparison.ok,
                data: {
                    meanAbsoluteChannelError: frameComparison.meanAbsoluteChannelError,
                    changedPixelFraction: frameComparison.changedPixelFraction,
                    channelTolerance: CHANNEL_TOLERANCE,
                },
            },
            {
                name: "the surrounding page action worked while build completion was held",
                ok: beforeFrame?.pageActionResult === "Page action works.",
            },
        ];
        const adapter = held?.adapter ?? null;
        checks.push({
            name: "the subject adapter identity was recorded",
            ok: Boolean(adapter && adapter.class !== "absent" && adapter.identity !== "none"),
        });
        return {
            ok: checks.every((check) => check.ok),
            checks,
            hardware: adapter
                ? `${adapter.class} adapter ${adapter.identity}`
                : "adapter unavailable",
        };
    },
};
