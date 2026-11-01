import { expect, test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390])
    for (const scene of [
        "spinningCapsule",
        "sleep",
        "spinner",
        "benchmarkMeshDrop",
        "benchmarkVillage",
    ])
        test(`${scene} re-pin look at ${width}`, async ({ browser }, info) => {
            const context = await browser.newContext({
                viewport: { width, height: width === 390 ? 844 : 720 },
                hasTouch: width === 390,
            });
            try {
                const page = await context.newPage();
                await page.goto(`${info.project.use.baseURL}/?scene=${scene}`);
                await expect(
                    page.getByRole("button", { name: "Scene", exact: true }),
                ).toBeEnabled();
                await capture(page, `${scene}-${width}`);
            } finally {
                await context.close();
            }
        });
