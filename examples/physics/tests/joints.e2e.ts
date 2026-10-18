import { readFileSync } from "node:fs";
import { expect, test } from "playwright/test";

const samples = [
    ["revolute", "Revolute"],
    ["prismatic", "Prismatic"],
    ["wheel", "Wheel"],
    ["filter", "Filter"],
    ["parallel", "Parallel Spring"],
] as const;

for (const [id, name] of samples) {
    test(`${name} has its own controls and a captured scene`, async ({ page }) => {
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto("/");
        const picker = page.getByRole("button", { name: "Scene", exact: true });
        await expect(picker).toBeEnabled();
        await picker.click();
        await page
            .getByRole("region", { name: "Joints", exact: true })
            .getByRole("button", { name, exact: true })
            .click();
        await expect(picker).toHaveText(name);
        await expect(picker).toBeEnabled();
        await expect(page.locator("#canvas")).toBeVisible();
        const panel = page.getByRole("region", { name: `${name} controls`, exact: true });
        if (id === "filter") await expect(panel).toBeHidden();
        else await expect(panel).toBeVisible();
        async function capture(suffix: string) {
            const directory = process.env.SHALLOT_CAPTURE_DIR;
            if (!directory) return;
            await page.addScriptTag({ type: "module", url: "/tests/capture-page.ts" });
            await page.waitForFunction(() => "capturePhysicsFrame" in window);
            await page.waitForFunction(() => {
                const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
                return (
                    canvas.width === Math.round(canvas.clientWidth * devicePixelRatio) &&
                    canvas.height === Math.round(canvas.clientHeight * devicePixelRatio)
                );
            });
            await page.evaluate(async () => {
                await (
                    window as unknown as { capturePhysicsFrame(): Promise<void> }
                ).capturePhysicsFrame();
            });
            await page.screenshot({ path: `${directory}/${id}-${suffix}.png` });
            await page.locator("#captured-frame").evaluate((element) => element.remove());
        }
        await capture("default");
        if (id !== "filter") {
            const sequence = readFileSync(new URL(`./${id}.controls.txt`, import.meta.url), "utf8")
                .trim()
                .split("\n");
            for (const line of sequence) {
                const [, control, value] = line.split("\t");
                const slider = page.locator(`[id=${JSON.stringify(`control-${control}`)}]`);
                if (await slider.count()) await slider.fill(value!);
                else
                    await panel
                        .getByRole("checkbox", { name: control, exact: true })
                        .setChecked(value === "1");
                // Capture before the final disabling sequence, with all dependent controls exposed.
                if (
                    control ===
                    (id === "revolute"
                        ? "Rotation"
                        : id === "prismatic"
                          ? "Translation"
                          : id === "wheel"
                            ? "Max Degrees"
                            : "Damping")
                )
                    await capture("controls");
            }
            if (id === "wheel") {
                await expect(
                    panel.getByRole("checkbox", { name: "Steering Limit", exact: true }),
                ).toBeHidden();
                await expect(
                    panel.getByRole("slider", { name: "Min Degrees", exact: true }),
                ).toBeHidden();
            }
        }
        await picker.click();
        await page.getByRole("button", { name: "Box Stack", exact: true }).click();
        await expect(picker).toHaveText("Box Stack");
        await expect(picker).toBeEnabled();
        await expect(panel).toBeHidden();
        expect(errors).toEqual([]);
    });

    test(`${name} can be picked and grabbed`, async ({ page }) => {
        await page.goto(`/tests/stepped.html?scene=${id}`);
        await page.waitForFunction(() => (window as unknown as { ready: boolean }).ready);
        const result = await page.evaluate(async () => {
            const sample = window as unknown as {
                drag(move: boolean): Promise<{
                    body: number;
                    picked: number;
                    mouse: number;
                    joint: number;
                    markerDuring: boolean;
                }>;
                finishDrag(): Promise<{ mouse: number; joint: number; marker: number }>;
            };
            const drag = await sample.drag(false);
            const released = await sample.finishDrag();
            return { drag, released };
        });
        expect(result.drag.picked).toBe(result.drag.body);
        expect(result.drag.mouse).toBeGreaterThan(0);
        expect(result.drag.joint).toBeGreaterThan(0);
        expect(result.drag.markerDuring).toBe(true);
        expect(result.released).toMatchObject({ mouse: 0, joint: 0, marker: 0 });
    });
}
