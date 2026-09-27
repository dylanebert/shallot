import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness/browser";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "the embedded scene reveals only after its first stepped frame",
    {
        claim: "frame-local loading or page usability breaks, the poster leaves at build completion, or the host reveals a wrong or blank first frame before a later correct frame",
        size: "integration",
        requires: ["browser"],
        subject: [
            "examples/loading-screen/fixtures/check.html",
            "examples/loading-screen/index.html",
            "examples/loading-screen/fixtures/browser-check.ts",
            "examples/loading-screen/src/host.ts",
            "examples/loading-screen/src/main.ts",
            "examples/loading-screen/src/reveal.ts",
            "examples/loading-screen/src/scene.ts",
            "examples/loading-screen/src/style.css",
            "src/standard/loading/index.ts",
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
            await page.getByRole("button", { name: /Still usable/ }).click();
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
                transparentOverlay: boolean;
                posterVisible: boolean;
                canvasHidden: boolean;
                overlayPresent: boolean;
                progress: { value: number; width: string }[];
                pageActionResult: string;
                cleaned: boolean;
            };
            releaseBuild(): void;
            waitForCleanup(): Promise<void>;
            waitForBuild(): Promise<void>;
            recordSnapshot(name: string): void;
            recordScreenshot(name: string, encoded: string): Promise<void>;
            stepFirstFrame(): Promise<void>;
            stepLaterFrame(): Promise<void>;
        };
    }
}
