import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Robustness recovery formats and constraint-graph readouts fit at ${width}`, async ({
        page,
    }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const select = async (name: string) => {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Robustness", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
        };
        await select("Overlap Recovery");
        const panel = page.locator(".controls");
        for (const [name, value, output] of [
            ["Extent", "0.7", "0.7"],
            ["Base Count", "7", "7"],
            ["Overlap", "0.75", "0.75"],
            ["Speed", "10", "10.0"],
            ["Hertz", "120", "120"],
            ["Damping Ratio", "3.4", "3.4"],
        ] as const) {
            await panel.getByRole("slider", { name, exact: true }).fill(value);
            await expect(panel.locator(`output[for='control-${name}']`)).toHaveText(output);
            await bodyHasFocus(page);
        }
        await pointerControl(page, panel.getByRole("button", { name: "Reset Scene", exact: true }));
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        await select("Overflow Color Pile");
        await page.evaluate(() => (window as unknown as { controlsKey(): number }).controlsKey());
        const section = panel.getByRole("region", { name: "Constraint graph", exact: true });
        await expect(section.locator("output")).toHaveCount(3);
        await expect(section.getByLabel("Neighbors", { exact: true })).toHaveText("25");
        await expect(section.getByLabel("Overflow contacts", { exact: true })).toHaveText(/^\d+$/);
        await expect(section.getByLabel("Total contacts", { exact: true })).toHaveText(/^\d+$/);
        await expect(panel.getByRole("slider")).toHaveCount(0);
        expect(
            await panel
                .locator("span, output")
                .evaluateAll((elements) =>
                    elements.every((el) => el.scrollWidth <= el.clientWidth),
                ),
        ).toBe(true);
        for (const name of ["Tiny Pyramid", "HighMassRatio1"]) {
            await select(name);
            await expect(panel).toHaveCount(0);
        }
    });
