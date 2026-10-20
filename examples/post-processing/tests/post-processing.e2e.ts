import { mkdir, writeFile } from "node:fs/promises";
import { expect, type Page, test } from "playwright/test";

type SteppedWindow = Window & {
    captureScene?: () => Promise<number[]>;
    baseline?: number[];
    combined?: number[];
};
const names = ["Lens distortion", "Chromatic aberration", "Vignette"];

async function capture(page: Page, name: string) {
    const save =
        Boolean(process.env.SHALLOT_CAPTURE_DIR) &&
        ["none", "lens-distortion", "chromatic-aberration", "vignette", "all"].includes(name);
    const result = await page.evaluate(
        async ({ name, save }) => {
            const win = window as SteppedWindow;
            const rgba = await win.captureScene!();
            win.baseline ??= rgba;
            function difference(reference: number[]): number {
                let total = 0;
                for (let i = 0; i < rgba.length; i += 4)
                    for (let channel = 0; channel < 3; channel++)
                        total += Math.abs(rgba[i + channel] - reference[i + channel]);
                return total / (rgba.length * 0.75);
            }
            let png = "";
            if (save) {
                const source = document.querySelector("canvas")!;
                const canvas = document.createElement("canvas");
                canvas.width = source.width;
                canvas.height = source.height;
                canvas
                    .getContext("2d")!
                    .putImageData(
                        new ImageData(new Uint8ClampedArray(rgba), canvas.width, canvas.height),
                        0,
                        0,
                    );
                png = canvas.toDataURL("image/png").split(",")[1];
            }
            if (name === "all") win.combined = rgba;
            return {
                baseline: difference(win.baseline),
                combined: win.combined ? difference(win.combined) : 0,
                colors: new Set(rgba).size,
                png,
            };
        },
        { name, save },
    );
    if (save) {
        await mkdir(process.env.SHALLOT_CAPTURE_DIR!, { recursive: true });
        await writeFile(
            `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
            Buffer.from(result.png, "base64"),
        );
    }
    return result;
}

test("the built page presents the scene and keeps its HUD inside the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    const panel = page.getByRole("region", { name: "Post-processing controls" });
    await panel.waitFor();
    const canvas = (await page.locator("canvas").boundingBox())!;
    const bounds = (await panel.boundingBox())!;
    expect(bounds.x).toBeGreaterThanOrEqual(canvas.x);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(canvas.x + canvas.width);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(canvas.y + canvas.height);
    for (const name of names)
        await expect(page.getByRole("button", { name, exact: true })).toHaveAttribute(
            "aria-pressed",
            "false",
        );
    expect(errors).toEqual([]);
});

test("each HUD toggle changes the captured scene alone, reversibly and in the combined stack", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:4266/tests/stepped.html");
    await page.waitForFunction(() => (window as SteppedWindow).captureScene !== undefined);
    expect((await capture(page, "none")).colors).toBeGreaterThan(100);
    const panel = page.getByRole("region", { name: "Post-processing controls" });
    const bounds = await panel.boundingBox();
    for (const [index, name] of names.entries()) {
        const toggle = page.getByRole("button", { name, exact: true });
        await toggle.click();
        await expect(toggle).toHaveAttribute("aria-pressed", "true");
        const changed = await capture(
            page,
            ["lens-distortion", "chromatic-aberration", "vignette"][index],
        );
        expect(changed.baseline, `${name} must change scene pixels`).toBeGreaterThan(0.25);
        expect(await panel.boundingBox()).toEqual(bounds);
        await toggle.click();
        await expect(toggle).toHaveAttribute("aria-pressed", "false");
        // Tonemapping dithering changes sub-byte averages between otherwise identical frames.
        expect((await capture(page, "reset")).baseline).toBeLessThan(0.1);
    }
    for (const name of names) await page.getByRole("button", { name, exact: true }).click();
    expect((await capture(page, "all")).baseline).toBeGreaterThan(0.25);
    for (const name of names) {
        const toggle = page.getByRole("button", { name, exact: true });
        await toggle.click();
        expect((await capture(page, "without")).combined, name).toBeGreaterThan(0.25);
        await toggle.click();
    }
    expect(errors).toEqual([]);
});
