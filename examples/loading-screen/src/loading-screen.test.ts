import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness/browser";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "the embedded scene reveals only after its first stepped frame",
    {
        claim: "frame-local loading or page usability breaks, the scene appears before its first stepped frame, or the first static scene frame is blank or geometrically wrong",
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
            const linkPoint = await page.evaluate(() => {
                const link = document.querySelector<HTMLAnchorElement>("#section-link");
                if (!link) return null;
                const rect = link.getBoundingClientRect();
                const x = rect.left + rect.width / 2;
                const y = rect.top + rect.height / 2;
                return { x, y, hitTarget: document.elementFromPoint(x, y) === link };
            });
            if (linkPoint) await page.mouse.click(linkPoint.x, linkPoint.y);
            const hashAfter = await page.evaluate(() => window.location.hash);
            await page.evaluate(
                ({ hitTarget, hashChanged }) =>
                    window.__loadingCheck!.recordLinkInteraction(hitTarget, hashChanged),
                {
                    hitTarget: linkPoint?.hitTarget ?? false,
                    hashChanged: linkPoint !== null && hashAfter === "#details",
                },
            );
            await page.evaluate(() => window.scrollTo(0, 0));
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
            recordLinkInteraction(hitTarget: boolean, hashChanged: boolean): void;
            recordScreenshot(name: string, encoded: string): Promise<void>;
            stepFirstFrame(): Promise<void>;
            stepLaterFrame(): Promise<void>;
        };
    }
}
