import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "playwright/test";
import { scenes } from "../src/scenes";

for (const scene of Object.keys(scenes)) {
    test(scene, async ({ page }) => {
        const directory = process.env.SHALLOT_CAPTURE_DIR;
        if (!directory) throw new Error("set SHALLOT_CAPTURE_DIR to the capture destination");
        await mkdir(directory, { recursive: true });
        await page.goto(`/?scene=${scene}`);
        await expect(page.getByRole("button", { name: "Scene", exact: true })).toHaveText(scene);
        await page.waitForFunction(() =>
            performance
                .getEntriesByType("measure")
                .some((entry) => entry.name.endsWith(":first-update")),
        );
        await page.locator("canvas").screenshot({ path: join(directory, `${scene}.png`) });
    });
}
