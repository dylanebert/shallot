import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness/browser";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "the embedded scene reveals only after its first stepped frame",
    {
        claim: "loading confinement or progress, description-line hit testing, first-frame reveal, static scene appearance, or responsive layout breaks",
        size: "integration",
        requires: ["browser"],
        subject: [
            "examples/loading-screen",
            "src/core/rendering",
            "src/engine/app",
            "src/extras/orbit",
            "src/harness/browser.ts",
            "src/project/build.ts",
            "src/standard/loading",
            "src/standard/rendering",
            "src/transitional/part",
            "src/transitional/transforms",
        ],
        budget: 20_000,
    },
    () =>
        runBrowserCheck(resolve(import.meta.dir, "../fixtures/check.html"), async (page) => {
            await page.evaluate(() => window.__loadingCheck!.begin());
            await page.evaluate(() => window.__loadingCheck!.waitForCompletion());

            const capture = async (name: string): Promise<void> => {
                const screenshot = await page.screenshot({ fullPage: true });
                await page.evaluate(
                    ({ label, encoded }) => window.__loadingCheck!.recordScreenshot(label, encoded),
                    { label: name, encoded: screenshot.toString("base64") },
                );
            };
            await capture("held");
            await page.evaluate(() => window.scrollTo(0, 0));
            const lineHitTarget = await page.evaluate(() => {
                const line = document.querySelector<HTMLParagraphElement>(".description");
                if (!line) return false;
                const range = document.createRange();
                range.selectNodeContents(line);
                const rect = range.getBoundingClientRect();
                return (
                    document.elementFromPoint(
                        rect.left + rect.width / 2,
                        rect.top + rect.height / 2,
                    ) === line
                );
            });
            await page.evaluate(
                (hitTarget) => window.__loadingCheck!.recordPageLineHitTest(hitTarget),
                lineHitTarget,
            );
            await page.evaluate(() => window.__loadingCheck!.releaseBuild());
            await page.evaluate(() => window.__loadingCheck!.waitForCleanup());
            await page.evaluate(() => window.__loadingCheck!.waitForBuild());
            await page.evaluate(() => window.__loadingCheck!.recordSnapshot("before-frame"));
            await capture("before-frame");

            await page.evaluate(() => window.__loadingCheck!.stepFirstFrame());
            await page.evaluate(() => window.__loadingCheck!.recordSnapshot("first-frame"));
            await capture("first-frame");

            await page.evaluate(() => window.__loadingCheck!.stepLaterFrame());
            await page.evaluate(() => window.__loadingCheck!.recordSnapshot("later-frame"));
            await capture("later-frame");

            for (const { width, height } of [
                { width: 390, height: 844 },
                { width: 844, height: 390 },
                { width: 820, height: 1180 },
                { width: 1280, height: 720 },
                { width: 1440, height: 900 },
                { width: 2560, height: 1080 },
            ]) {
                await page.setViewportSize({ width, height });
                await page.evaluate(() => window.scrollTo(0, 0));
                const layout = await page.evaluate(() => {
                    const line = document.querySelector<HTMLElement>(".description");
                    const frame = document.querySelector<HTMLElement>("#frame");
                    const lineRect = line?.getBoundingClientRect();
                    const frameRect = frame?.getBoundingClientRect();
                    return {
                        found: Boolean(lineRect && frameRect),
                        width: window.innerWidth,
                        height: window.innerHeight,
                        lineLeft: lineRect?.left ?? -1,
                        lineTop: lineRect?.top ?? -1,
                        lineRight: lineRect?.right ?? -1,
                        lineBottom: lineRect?.bottom ?? -1,
                        frameLeft: frameRect?.left ?? -1,
                        frameTop: frameRect?.top ?? -1,
                        frameRight: frameRect?.right ?? -1,
                        frameBottom: frameRect?.bottom ?? -1,
                        frameWidth: frameRect?.width ?? 0,
                        frameHeight: frameRect?.height ?? 0,
                        documentScrollWidth: document.documentElement.scrollWidth,
                        bodyScrollWidth: document.body.scrollWidth,
                    };
                });
                await page.evaluate(
                    (observed) => window.__loadingCheck!.recordViewport(observed),
                    layout,
                );
            }
        }),
);

declare global {
    interface Window {
        __loadingCheck: {
            begin(): void;
            waitForCompletion(): Promise<void>;
            snapshot(): {
                frameOwnsOverlay: boolean;
                overlayRectInsideFrame: boolean;
                transparentOverlay: boolean;
                canvasHidden: boolean;
                overlayPresent: boolean;
                progress: { value: number; width: string }[];
                cleaned: boolean;
            };
            releaseBuild(): void;
            waitForCleanup(): Promise<void>;
            waitForBuild(): Promise<void>;
            recordSnapshot(name: string): void;
            recordPageLineHitTest(hitTarget: boolean): void;
            recordViewport(layout: {
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
            }): void;
            recordScreenshot(name: string, encoded: string): Promise<void>;
            stepFirstFrame(): Promise<void>;
            stepLaterFrame(): Promise<void>;
        };
    }
}
