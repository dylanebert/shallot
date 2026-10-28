import { test } from "playwright/test";
import { collisionNames } from "../src/collision";
import { capture } from "./capture";

for (const width of [1280, 390])
    for (const id of collisionNames) {
        test(`${id} Collision look review at ${width}`, async ({ page }) => {
            page.on("pageerror", (error) => {
                throw error;
            });
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            await page.goto(`/?scene=${id}`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            if (id === "castWorld")
                for (const name of ["Spheres", "Capsules", "Hulls", "Meshes", "Height Field"])
                    await page.getByRole("button", { name, exact: true }).click();
            await page.waitForTimeout(1000);
            await capture(page, `${id}-${width}-default`);
        });
    }
