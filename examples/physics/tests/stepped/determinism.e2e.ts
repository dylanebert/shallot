import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Determinism readouts and its category-scoped Mesh Drop fit at ${width}`, async ({
        page,
    }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const panel = page.locator(".controls");
        for (const name of ["Falling Ragdolls", "Wave Pile", "Query Spawn", "Mesh Drop"]) {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Determinism", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
            await expect(panel.getByText("Determinism", { exact: true })).toBeVisible();
            await expect(panel.getByRole("slider")).toHaveCount(0);
            await expect(panel.getByLabel("Hash", { exact: true })).toHaveText("0x00000000");
            await page.evaluate(() =>
                (window as unknown as { controlsKey(): number }).controlsKey(),
            );
            await expect(panel.getByLabel("Hash", { exact: true })).toHaveText(/^0x[0-9A-F]{8}$/);
            if (name === "Query Spawn") {
                await expect(panel.getByLabel("Spawned", { exact: true })).toHaveText("1");
                await expect(panel.getByLabel("Query hash", { exact: true })).toHaveText(
                    /^0x[0-9A-F]{8}$/,
                );
            }
            expect(
                await panel
                    .locator("span, output")
                    .evaluateAll((elements) =>
                        elements.every((el) => el.scrollWidth <= el.clientWidth),
                    ),
            ).toBe(true);
        }
    });
