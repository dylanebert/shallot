import { build, Compute, type Loading, minimalDark } from "@dylanebert/shallot";
import { DARK } from "@dylanebert/shallot/brand";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { mountHost } from "../src/host";
import { revealAfterFirstFrame } from "../src/reveal";
import { SCENE } from "../src/scene";

type Progress = { value: number; width: string };
type Adapter = { class: string; identity: string };
type Snapshot = {
    frameOwnsOverlay: boolean;
    overlayRectInsideFrame: boolean;
    transparentOverlay: boolean;
    canvasHidden: boolean;
    overlayPresent: boolean;
    progress: Progress[];
    cleaned: boolean;
    adapter: Adapter | null;
};
type Bounds = { left: number; top: number; right: number; bottom: number };
type ScreenshotObservation = {
    sceneColorPixels: number;
    descriptionInkPixels: number;
    sceneBounds: Bounds | null;
    canvas: { x: number; y: number; width: number; height: number };
    canvasPixels: Uint8Array;
    progressColorPixelsInFrame: number;
    sceneBackgroundFraction: number;
    clearBackground: [number, number, number];
    adjacentBackground: [number, number, number];
    pageBackground: [number, number, number];
};

type ViewportLayout = {
    found: boolean;
    width: number;
    height: number;
    lineLeft: number;
    lineTop: number;
    lineRight: number;
    lineBottom: number;
    frameLeft: number;
    frameTop: number;
    frameRight: number;
    frameBottom: number;
    frameWidth: number;
    frameHeight: number;
    documentScrollWidth: number;
    bodyScrollWidth: number;
};
type Point = { x: number; y: number };

// The scaffold's orbit pose (distance 5, yaw 0.6, pitch 0.25) exposes three cube faces.
// Project all eight corners through that perspective pose; the hull is the expected silhouette.
const CAMERA_FOV = 60;
const CAMERA_DISTANCE = 5;
const ORBIT_YAW = 0.6;
const ORBIT_PITCH = 0.25;
// A scene pixel differs when its largest RGB-channel delta from the clear corner exceeds 24.
const SCENE_BACKGROUND_DELTA = 24;
const DESCRIPTION_INK_MIN_PIXELS = 20;
const DESCRIPTION_INK_TOLERANCE = 24;
const PART_HALF_EXTENT = 0.5;
const FRAME_DIFFERENCE_MEAN_LIMIT = 0.25;
const FRAME_DIFFERENCE_PIXEL_LIMIT = 0.001;
const CHANNEL_TOLERANCE = 2;
const SCENE_BACKGROUND_TOLERANCE = 2;
const PROJECTED_BOUNDS_PADDING = 1;
const BOX_POPULATION_MIN_FACTOR = 0.5;
const BOX_POPULATION_MAX_FACTOR = 1.5;
const BOX_SIZE_MIN_FACTOR = 0.72;
const BOX_SIZE_MAX_FACTOR = 1.28;
const BOX_CENTER_TOLERANCE_FACTOR = 0.12;
const parseColor = (hex: string): [number, number, number] => [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
];
const PAGE_BACKGROUND = parseColor(DARK.bg);
const DESCRIPTION_INK = parseColor(DARK.ink);
const PROGRESS_GOLD = parseColor(DARK.gold);

function convexHull(points: Point[]): Point[] {
    const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (origin: Point, a: Point, b: Point) =>
        (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
    const lower: Point[] = [];
    for (const point of sorted) {
        while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower.at(-1)!, point) <= 0)
            lower.pop();
        lower.push(point);
    }
    const upper: Point[] = [];
    for (const point of sorted.toReversed()) {
        while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper.at(-1)!, point) <= 0)
            upper.pop();
        upper.push(point);
    }
    return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function polygonArea(points: Point[]): number {
    let twiceArea = 0;
    for (let i = 0; i < points.length; i++) {
        const current = points[i]!;
        const next = points[(i + 1) % points.length]!;
        twiceArea += current.x * next.y - current.y * next.x;
    }
    return Math.abs(twiceArea) / 2;
}

function projectUnitCube(canvas: ScreenshotObservation["canvas"]) {
    const yawCos = Math.cos(ORBIT_YAW);
    const yawSin = Math.sin(ORBIT_YAW);
    const pitchCos = Math.cos(ORBIT_PITCH);
    const pitchSin = Math.sin(ORBIT_PITCH);
    const direction = {
        x: pitchCos * yawSin,
        y: pitchSin,
        z: pitchCos * yawCos,
    };
    const right = { x: yawCos, y: 0, z: -yawSin };
    const up = { x: -pitchSin * yawSin, y: pitchCos, z: -pitchSin * yawCos };
    const focalPixels = canvas.height / (2 * Math.tan((CAMERA_FOV * Math.PI) / 360));
    const points: Point[] = [];
    for (const x of [-PART_HALF_EXTENT, PART_HALF_EXTENT]) {
        for (const y of [-PART_HALF_EXTENT, PART_HALF_EXTENT]) {
            for (const z of [-PART_HALF_EXTENT, PART_HALF_EXTENT]) {
                const depth =
                    CAMERA_DISTANCE - (x * direction.x + y * direction.y + z * direction.z);
                const viewX = x * right.x + y * right.y + z * right.z;
                const viewY = x * up.x + y * up.y + z * up.z;
                points.push({
                    x: canvas.width / 2 + (focalPixels * viewX) / depth,
                    y: canvas.height / 2 - (focalPixels * viewY) / depth,
                });
            }
        }
    }
    const hull = convexHull(points);
    const left = Math.min(...hull.map((point) => point.x));
    const rightEdge = Math.max(...hull.map((point) => point.x));
    const top = Math.min(...hull.map((point) => point.y));
    const bottom = Math.max(...hull.map((point) => point.y));
    return {
        expectedWidth: rightEdge - left,
        expectedHeight: bottom - top,
        expectedPopulation: polygonArea(hull),
        expectedCenterX: (left + rightEdge) / 2,
        expectedCenterY: (top + bottom) / 2,
        expectedLeft: left,
        expectedTop: top,
        expectedRight: rightEdge,
        expectedBottom: bottom,
    };
}

function sceneGeometry(image: ScreenshotObservation | undefined) {
    const empty = {
        ok: false,
        population: 0,
        expectedPopulation: 0,
        width: 0,
        height: 0,
        expectedWidth: 0,
        expectedHeight: 0,
        centerOffsetX: 0,
        centerOffsetY: 0,
        expectedCenterX: 0,
        expectedCenterY: 0,
        expectedLeft: 0,
        expectedTop: 0,
        expectedRight: 0,
        expectedBottom: 0,
        clearBackground: [0, 0, 0] as [number, number, number],
        boundsLeft: -1,
        boundsTop: -1,
        boundsRight: -1,
        boundsBottom: -1,
    };
    if (!image) return empty;
    const expected = projectUnitCube(image.canvas);
    const bounds = image.sceneBounds;
    const width = bounds ? bounds.right - bounds.left + 1 : 0;
    const height = bounds ? bounds.bottom - bounds.top + 1 : 0;
    const centerOffsetX = bounds
        ? Math.abs((bounds.left + bounds.right) / 2 - expected.expectedCenterX)
        : image.canvas.width;
    const centerOffsetY = bounds
        ? Math.abs((bounds.top + bounds.bottom) / 2 - expected.expectedCenterY)
        : image.canvas.height;
    return {
        ok: Boolean(
            bounds &&
                image.sceneColorPixels >= expected.expectedPopulation * BOX_POPULATION_MIN_FACTOR &&
                image.sceneColorPixels <= expected.expectedPopulation * BOX_POPULATION_MAX_FACTOR &&
                width >= expected.expectedWidth * BOX_SIZE_MIN_FACTOR &&
                width <= expected.expectedWidth * BOX_SIZE_MAX_FACTOR &&
                height >= expected.expectedHeight * BOX_SIZE_MIN_FACTOR &&
                height <= expected.expectedHeight * BOX_SIZE_MAX_FACTOR &&
                centerOffsetX <= expected.expectedWidth * BOX_CENTER_TOLERANCE_FACTOR &&
                centerOffsetY <= expected.expectedHeight * BOX_CENTER_TOLERANCE_FACTOR,
        ),
        population: image.sceneColorPixels,
        clearBackground: image.clearBackground,
        ...expected,
        width,
        height,
        centerOffsetX,
        centerOffsetY,
        boundsLeft: bounds?.left ?? -1,
        boundsTop: bounds?.top ?? -1,
        boundsRight: bounds?.right ?? -1,
        boundsBottom: bounds?.bottom ?? -1,
    };
}

function compareSceneBackground(
    image: ScreenshotObservation | undefined,
    geometry: ReturnType<typeof sceneGeometry>,
): { ok: boolean; maxChannelError: number; pixelsCompared: number } {
    if (!image) return { ok: false, maxChannelError: 255, pixelsCompared: 0 };
    let maxChannelError = 0;
    let pixelsCompared = 0;
    for (let y = 0; y < image.canvas.height; y++) {
        for (let x = 0; x < image.canvas.width; x++) {
            const pixelX = x + 0.5;
            const pixelY = y + 0.5;
            if (
                pixelX >= geometry.expectedLeft - PROJECTED_BOUNDS_PADDING &&
                pixelX <= geometry.expectedRight + PROJECTED_BOUNDS_PADDING &&
                pixelY >= geometry.expectedTop - PROJECTED_BOUNDS_PADDING &&
                pixelY <= geometry.expectedBottom + PROJECTED_BOUNDS_PADDING
            )
                continue;
            const index = (y * image.canvas.width + x) * 3;
            for (let channel = 0; channel < 3; channel++) {
                maxChannelError = Math.max(
                    maxChannelError,
                    Math.abs(
                        image.canvasPixels[index + channel]! - image.adjacentBackground[channel]!,
                    ),
                );
            }
            pixelsCompared++;
        }
    }
    return {
        ok: pixelsCompared > 0 && maxChannelError <= SCENE_BACKGROUND_TOLERANCE,
        maxChannelError,
        pixelsCompared,
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
    if (host.canvas.classList.contains("visible")) firstFrameReady.resolve();
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
const viewportLayouts: ViewportLayout[] = [];
const screen = minimalDark({ container: host.frame });
let overlay: HTMLElement | null = null;
let app: Awaited<ReturnType<typeof build>> | null = null;
let loadingAtCompletion: Snapshot | undefined;
let buildError: unknown;
let began = false;
let cleaned = false;
let pageLineHitTarget = false;

function progressElement(): HTMLElement | null {
    return overlay?.firstElementChild?.firstElementChild?.firstElementChild as HTMLElement | null;
}

function snapshot(): Snapshot {
    const canvasStyle = getComputedStyle(host.canvas);
    const frameRect = host.frame.getBoundingClientRect();
    const overlayRect = overlay?.getBoundingClientRect();
    return {
        frameOwnsOverlay: overlay?.parentElement === host.frame,
        overlayRectInsideFrame: Boolean(
            overlayRect &&
                overlayRect.left >= frameRect.left &&
                overlayRect.top >= frameRect.top &&
                overlayRect.right <= frameRect.right &&
                overlayRect.bottom <= frameRect.bottom,
        ),
        transparentOverlay:
            overlay !== null && getComputedStyle(overlay).backgroundColor === "rgba(0, 0, 0, 0)",
        canvasHidden: !host.canvas.classList.contains("visible") && canvasStyle.opacity === "0",
        overlayPresent: overlay?.isConnected === true,
        progress: [...barProgress],
        cleaned,
        adapter: Compute.adapter
            ? { class: Compute.adapter.class, identity: Compute.adapter.identity }
            : null,
    };
}

const loading: Loading = {
    show() {
        const cleanup = screen.show();
        overlay =
            [...document.querySelectorAll<HTMLElement>("*")].find(
                (element) => element.style.zIndex === "10000",
            ) ?? null;
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
    const frame = rect(host.frame);
    const canvas = rect(host.canvas);
    const descriptionElement = document.querySelector<HTMLElement>(".description");
    const description = descriptionElement ? rect(descriptionElement) : null;
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
    const clearIndex = (canvas.y * image.width + canvas.x) * 4;
    const clearBackground = [
        pixels[clearIndex]!,
        pixels[clearIndex + 1]!,
        pixels[clearIndex + 2]!,
    ] as [number, number, number];
    const isSceneColor = (r: number, g: number, b: number) =>
        Math.max(
            Math.abs(r - clearBackground[0]),
            Math.abs(g - clearBackground[1]),
            Math.abs(b - clearBackground[2]),
        ) > SCENE_BACKGROUND_DELTA;
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
    const isDescriptionInk = (r: number, g: number, b: number) =>
        Math.abs(r - DESCRIPTION_INK[0]) <= DESCRIPTION_INK_TOLERANCE &&
        Math.abs(g - DESCRIPTION_INK[1]) <= DESCRIPTION_INK_TOLERANCE &&
        Math.abs(b - DESCRIPTION_INK[2]) <= DESCRIPTION_INK_TOLERANCE;
    const descriptionInkPixels = description ? inside(description, isDescriptionInk) : 0;
    const isProgressColor = (r: number, g: number, b: number) =>
        Math.abs(r - PROGRESS_GOLD[0]) < 24 &&
        Math.abs(g - PROGRESS_GOLD[1]) < 24 &&
        Math.abs(b - PROGRESS_GOLD[2]) < 24;
    const isPageBackground = (r: number, g: number, b: number) =>
        Math.abs(r - PAGE_BACKGROUND[0]) <= 2 &&
        Math.abs(g - PAGE_BACKGROUND[1]) <= 2 &&
        Math.abs(b - PAGE_BACKGROUND[2]) <= 2;
    const progressColorPixelsInFrame = inside(frame, isProgressColor);
    const sceneBackgroundFraction =
        inside(canvas, isPageBackground) / (canvas.width * canvas.height);
    const sample = (x: number, y: number): [number, number, number] => {
        const index = (y * image.width + x) * 4;
        return [pixels[index]!, pixels[index + 1]!, pixels[index + 2]!];
    };
    const adjacentBackground = sample(frame.x - 1, frame.y + Math.floor(frame.height / 2));
    const pageBackground = sample(8, 8);
    return {
        sceneColorPixels,
        descriptionInkPixels,
        sceneBounds,
        canvas,
        canvasPixels,
        progressColorPixelsInFrame,
        sceneBackgroundFraction,
        clearBackground,
        adjacentBackground,
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
        recordPageLineHitTest(hitTarget: boolean): void;
        recordViewport(layout: ViewportLayout): void;
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
            plugins: [OrbitPlugin, revealAfterFirstFrame(host, loading)],
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
    recordPageLineHitTest(hitTarget) {
        pageLineHitTarget = hitTarget;
    },
    recordViewport(layout) {
        viewportLayouts.push(layout);
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
        const backgroundComparison = compareSceneBackground(firstFrameImage, firstGeometry);
        const frameComparison = compareSceneFrames(firstFrameImage, laterFrameImage);
        const responsiveChecks = viewportLayouts.map((layout) => {
            const aspectError = Math.abs(layout.frameWidth - (layout.frameHeight * 16) / 9);
            const topMargin = layout.lineTop;
            const bottomMargin = layout.height - layout.frameBottom;
            const marginDifference = Math.abs(topMargin - bottomMargin);
            const lineAboveFrame = layout.lineTop >= 0 && layout.lineBottom <= layout.frameTop;
            const frameInsideViewport =
                layout.frameLeft >= 0 &&
                layout.frameTop >= 0 &&
                layout.frameRight <= layout.width &&
                layout.frameBottom <= layout.height;
            const noHorizontalOverflow =
                layout.documentScrollWidth <= layout.width &&
                layout.bodyScrollWidth <= layout.width;
            const sidePadding =
                layout.width > 480 ||
                (layout.frameLeft >= 16 && layout.width - layout.frameRight >= 16);
            const ok = Boolean(
                layout.found &&
                    lineAboveFrame &&
                    frameInsideViewport &&
                    aspectError <= 1 &&
                    noHorizontalOverflow &&
                    marginDifference <= 2 &&
                    sidePadding &&
                    layout.frameWidth <= 880,
            );
            return {
                name: `${layout.width}x${layout.height} keeps the scene fitted and group vertically balanced`,
                ok,
                data: {
                    found: Number(layout.found),
                    viewportWidth: layout.width,
                    viewportHeight: layout.height,
                    lineLeft: layout.lineLeft,
                    lineTop: layout.lineTop,
                    lineRight: layout.lineRight,
                    lineBottom: layout.lineBottom,
                    frameLeft: layout.frameLeft,
                    frameTop: layout.frameTop,
                    frameRight: layout.frameRight,
                    frameBottom: layout.frameBottom,
                    frameWidth: layout.frameWidth,
                    frameHeight: layout.frameHeight,
                    aspectError,
                    documentScrollWidth: layout.documentScrollWidth,
                    bodyScrollWidth: layout.bodyScrollWidth,
                    topMargin,
                    bottomMargin,
                    marginDifference,
                    sidePaddingLeft: layout.frameLeft,
                    sidePaddingRight: layout.width - layout.frameRight,
                },
            };
        });
        const intermediateSteps =
            held?.progress.filter((step) => step.value > 0 && step.value < 1) ?? [];
        const completeSteps = held?.progress.filter((step) => step.value === 1) ?? [];
        const widthPercent = (width: string | undefined) => Number.parseFloat(width ?? "") || 0;
        const widthMatchesValue = (step: Progress) =>
            Math.abs(widthPercent(step.width) - step.value * 100) <= 0.001;
        const intermediateWidthMatches = intermediateSteps.filter(widthMatchesValue);
        const completeWidthMatches = completeSteps.filter(widthMatchesValue);
        const checks: {
            name: string;
            ok: boolean;
            data?: Record<string, number>;
        }[] = [
            {
                name: "build completion kept transparent loading inside its owned frame bounds",
                ok: Boolean(
                    held?.frameOwnsOverlay &&
                        held.overlayRectInsideFrame &&
                        held.transparentOverlay &&
                        held.overlayPresent,
                ),
            },
            {
                name: "build progress reached completion and drove an intermediate frame-local bar",
                ok: intermediateWidthMatches.length > 0 && completeWidthMatches.length > 0,
                data: {
                    progressSamples: held?.progress.length ?? 0,
                    intermediateSamples: intermediateSteps.length,
                    intermediateWidthMatches: intermediateWidthMatches.length,
                    firstIntermediateValue: intermediateSteps[0]?.value ?? 0,
                    firstIntermediateWidthPercent: widthPercent(intermediateSteps[0]?.width),
                    expectedIntermediateWidthPercent: (intermediateSteps[0]?.value ?? 0) * 100,
                    completeSamples: completeSteps.length,
                    completeWidthMatches: completeWidthMatches.length,
                    firstCompleteWidthPercent: widthPercent(completeSteps[0]?.width),
                },
            },
            {
                name: "the description line remains hit-testable during loading",
                ok: pageLineHitTarget,
                data: { hitTarget: Number(pageLineHitTarget) },
            },
            {
                name: "the held screenshot contains more than 20 ink-coloured description pixels",
                ok: Boolean(
                    heldImage && heldImage.descriptionInkPixels > DESCRIPTION_INK_MIN_PIXELS,
                ),
                data: {
                    inkPixels: heldImage?.descriptionInkPixels ?? 0,
                    minimumInkPixels: DESCRIPTION_INK_MIN_PIXELS,
                    channelTolerance: DESCRIPTION_INK_TOLERANCE,
                },
            },
            {
                name: "the screenshot shows the loading bar over the page-background void",
                ok: Boolean(
                    heldImage &&
                        heldImage.progressColorPixelsInFrame > 100 &&
                        heldImage.sceneBackgroundFraction >= 0.97 &&
                        heldImage.pageBackground.every(
                            (channel, index) => Math.abs(channel - PAGE_BACKGROUND[index]!) <= 2,
                        ),
                ),
                data: {
                    progressPixelsInFrame: heldImage?.progressColorPixelsInFrame ?? 0,
                    sceneBackgroundFraction: heldImage?.sceneBackgroundFraction ?? 0,
                },
            },
            {
                name: "the canvas stays hidden over the page-background void until its first stepped frame",
                ok: Boolean(
                    held?.canvasHidden &&
                        held.overlayPresent &&
                        heldImage &&
                        heldImage.sceneBackgroundFraction >= 0.97 &&
                        beforeFrame?.canvasHidden &&
                        beforeFrame.cleaned &&
                        !beforeFrame.overlayPresent &&
                        beforeFrameImage &&
                        beforeFrameImage.sceneBackgroundFraction >= 0.99,
                ),
                data: {
                    heldBackgroundFraction: heldImage?.sceneBackgroundFraction ?? 0,
                    beforeFrameBackgroundFraction: beforeFrameImage?.sceneBackgroundFraction ?? 0,
                },
            },
            {
                name: "the first revealed scene has the orbit-projected unit-cube population and centered bounds",
                ok: Boolean(firstFrame && !firstFrame.canvasHidden && firstGeometry.ok),
                data: {
                    scenePixels: firstGeometry.population,
                    expectedScenePixels: firstGeometry.expectedPopulation,
                    populationThreshold: SCENE_BACKGROUND_DELTA,
                    clearRed: firstGeometry.clearBackground[0],
                    clearGreen: firstGeometry.clearBackground[1],
                    clearBlue: firstGeometry.clearBackground[2],
                    boundsLeft: firstGeometry.boundsLeft,
                    boundsTop: firstGeometry.boundsTop,
                    boundsRight: firstGeometry.boundsRight,
                    boundsBottom: firstGeometry.boundsBottom,
                    expectedWidth: firstGeometry.expectedWidth,
                    expectedHeight: firstGeometry.expectedHeight,
                    expectedCenterX: firstGeometry.expectedCenterX,
                    expectedCenterY: firstGeometry.expectedCenterY,
                    centerOffsetX: firstGeometry.centerOffsetX,
                    centerOffsetY: firstGeometry.centerOffsetY,
                },
            },
            {
                name: "outside the projected cube, the revealed scene matches adjacent page background within 2 RGB levels per channel",
                ok: backgroundComparison.ok,
                data: {
                    maxChannelError: backgroundComparison.maxChannelError,
                    tolerance: SCENE_BACKGROUND_TOLERANCE,
                    pixelsCompared: backgroundComparison.pixelsCompared,
                    pageRed: firstFrameImage?.adjacentBackground[0] ?? 0,
                    pageGreen: firstFrameImage?.adjacentBackground[1] ?? 0,
                    pageBlue: firstFrameImage?.adjacentBackground[2] ?? 0,
                },
            },
            {
                name: "the later stepped scene has the orbit-projected unit-cube population and centered bounds",
                ok: laterGeometry.ok,
                data: {
                    scenePixels: laterGeometry.population,
                    expectedScenePixels: laterGeometry.expectedPopulation,
                    populationThreshold: SCENE_BACKGROUND_DELTA,
                    clearRed: laterGeometry.clearBackground[0],
                    clearGreen: laterGeometry.clearBackground[1],
                    clearBlue: laterGeometry.clearBackground[2],
                    boundsLeft: laterGeometry.boundsLeft,
                    boundsTop: laterGeometry.boundsTop,
                    boundsRight: laterGeometry.boundsRight,
                    boundsBottom: laterGeometry.boundsBottom,
                    expectedWidth: laterGeometry.expectedWidth,
                    expectedHeight: laterGeometry.expectedHeight,
                    expectedCenterX: laterGeometry.expectedCenterX,
                    expectedCenterY: laterGeometry.expectedCenterY,
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
        ];
        checks.push({
            name: "all six responsive viewports were measured",
            ok: viewportLayouts.length === 6,
            data: { measuredViewports: viewportLayouts.length, expectedViewports: 6 },
        });
        checks.push(...responsiveChecks);
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
