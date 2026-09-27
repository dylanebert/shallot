import { build, Compute, type Loading, minimalDark } from "@dylanebert/shallot";
import { mountHost } from "./host";
import { revealAfterFirstFrame } from "./reveal";
import { SCENE } from "./scene";

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
type ScreenshotObservation = {
    width: number;
    height: number;
    posterTitlePixels: number;
    sceneColorPixels: number;
    progressColorPixelsInFrame: number;
    progressColorPixelsOutsideFrame: number;
    pageBackground: [number, number, number];
};

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
    const sceneColorPixels = inside(canvas, (r, g, b) => r > 90 && r > g * 1.25 && g > b * 1.05);
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
        width: image.width,
        height: image.height,
        posterTitlePixels,
        sceneColorPixels,
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
                name: "the first revealed canvas screenshot contains the rendered scene",
                ok: Boolean(
                    firstFrame &&
                        !firstFrame.posterVisible &&
                        !firstFrame.canvasHidden &&
                        firstFrameImage &&
                        firstFrameImage.sceneColorPixels > 100,
                ),
                data: { sceneColorPixels: firstFrameImage?.sceneColorPixels ?? 0 },
            },
            {
                name: "a later stepped frame contains the same scene",
                ok: Boolean(laterFrameImage && laterFrameImage.sceneColorPixels > 100),
                data: { sceneColorPixels: laterFrameImage?.sceneColorPixels ?? 0 },
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
