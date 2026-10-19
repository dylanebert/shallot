import { readFileSync } from "node:fs";
import { test } from "playwright/test";
import { capture } from "./capture";

for (const [id, name, last] of [
    ["revolute", "Revolute", "Rotation"],
    ["prismatic", "Prismatic", "Translation"],
    ["wheel", "Wheel", "Max Degrees"],
    ["filter", "Filter", ""],
    ["parallel", "Parallel Spring", "Damping"],
]) {
    test(`${name} look review`, async ({ page }) => {
        await page.goto("/");
        await page.getByRole("button", { name: "Scene", exact: true }).click();
        await page.getByRole("button", { name, exact: true }).click();
        await capture(page, `${id}-default`);
        if (id === "filter") return;
        const sequence = readFileSync(new URL(`./${id}.controls.txt`, import.meta.url), "utf8")
            .trim()
            .split("\n");
        for (const line of sequence) {
            const [, control, value] = line.split("\t");
            const slider = page.locator(`[id=${JSON.stringify(`control-${control}`)}]`);
            if (await slider.count()) await slider.fill(value!);
            else
                await page
                    .getByRole("checkbox", { name: control, exact: true })
                    .setChecked(value === "1");
            if (control === last) {
                await capture(page, `${id}-controls`);
                break;
            }
        }
    });
}
