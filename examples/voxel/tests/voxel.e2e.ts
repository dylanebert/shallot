import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        voxel: {
            step(ticks: number): void;
            capture(
                evidence?: boolean,
            ): Promise<{ png: string; coverage: number; corners: boolean[] }>;
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
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    await page.goto(`http://127.0.0.1:4206/tests/stepped.html${directory ? "?shipped" : ""}`);
    await page.waitForFunction(() => window.voxel);
    await page.evaluate(() => window.voxel.step(240));
    const image = await page.evaluate(() => window.voxel.capture(true));
    expect(image.coverage).toBeGreaterThan(0.95);
    expect(image.corners).toEqual([false, false]);
    expect(errors).toEqual([]);
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
