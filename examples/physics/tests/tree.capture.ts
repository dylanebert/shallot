import { test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390])
    test(`Tree Benchmark look review at ${width}`, async ({ page }) => {
        page.on("pageerror", (error) => {
            throw error;
        });
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/?scene=treeBenchmark");
        await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
        await page.waitForTimeout(2000);
        await capture(page, `treeBenchmark-${width}-default`);
        for (const [name, suffix] of [
            ["Ray Cast", "ray"],
            ["Overlap", "overlap"],
            ["Closest Point", "closest"],
        ] as const) {
            const toggle = page.getByLabel(name, { exact: true });
            await toggle.check();
            await capture(page, `treeBenchmark-${width}-${suffix}`);
            await toggle.uncheck();
        }
        for (const name of ["Ray Cast", "Overlap", "Closest Point"])
            await page.getByLabel(name, { exact: true }).check();
        await capture(page, `treeBenchmark-${width}-queries`);
        await page.getByLabel("Level", { exact: true }).fill("3");
        await capture(page, `treeBenchmark-${width}-level3`);
    });
