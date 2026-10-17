import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        voxel: {
            step(ticks: number): void;
            capture(
                pixels?: boolean,
            ): Promise<{ png: string; rgba: number[]; width: number; height: number }>;
        };
    }
}
test("the stepped scrolling terrain covers the ground and retains a visible view-distance edge", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
    });
    await page.goto("http://127.0.0.1:4206/tests/stepped.html");
    await page.waitForFunction(() => window.voxel);
    await page.evaluate(() => window.voxel.step(240));
    const image = await page.evaluate(() => window.voxel.capture(true));
    const { rgba, width, height } = image;
    // Grass and earth have less blue than red/green; beyond the footprint is daytime sky.
    const land = (x: number, y: number) => {
        const i = (Math.floor(y * height) * width + Math.floor(x * width)) * 4;
        return rgba[i] > rgba[i + 2] * 1.1 && rgba[i + 1] > rgba[i + 2] * 1.1;
    };
    let covered = 0,
        total = 0;
    for (let y = 0.4; y < 0.65; y += 0.01)
        for (let x = 0.35; x < 0.65; x += 0.01) {
            covered += Number(land(x, y));
            total++;
        }
    expect(covered / total).toBeGreaterThan(0.95);
    expect(land(0.05, 0.05)).toBe(false);
    expect(land(0.95, 0.05)).toBe(false);
    expect(errors).toEqual([]);
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (directory) {
        mkdirSync(directory, { recursive: true });
        writeFileSync(`${directory}/frame.png`, Buffer.from(image.png.split(",")[1], "base64"));
        for (let i = 0; i < 90; i++) {
            await page.evaluate(() => window.voxel.step(4));
            const frame = await page.evaluate(() => window.voxel.capture());
            writeFileSync(
                `${directory}/scroll-${String(i).padStart(3, "0")}.png`,
                Buffer.from(frame.png.split(",")[1], "base64"),
            );
        }
    }
});
