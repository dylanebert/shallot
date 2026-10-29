import { test } from "playwright/test";
import { issueScenes } from "../src/issues";
import { capture } from "./capture";

for (const width of [1280, 390])
    for (const id of Object.keys(issueScenes)) {
        test(`${id} Issues look review at ${width}`, async ({ page }) => {
            page.on("pageerror", (error) => {
                throw error;
            });
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            await page.goto(`/tests/issues-capture.html?scene=${id}`);
            await page.waitForFunction(() => "issuesCaptureReady" in window);
            const frames = {
                run: () =>
                    page.evaluate(async () => {
                        await (
                            window as unknown as { drawIssuesCapture(): Promise<void> }
                        ).drawIssuesCapture();
                    }),
            };
            await capture(page, `${id}-${width}-default`, frames);
            await page.evaluate(async () => {
                await (
                    window as unknown as { advanceIssuesCapture(): Promise<void> }
                ).advanceIssuesCapture();
            });
            await capture(page, `${id}-${width}-step-120`, frames);
        });
    }
