import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        city: {
            capture(): Promise<{ png: string; rgba: number[]; width: number; height: number }>;
        };
    }
}
test("the stepped city's buildings fill the centre of the view", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
    });
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => window.city);
    const image = await page.evaluate(() => window.city.capture());
    let covered = 0,
        accents = 0,
        total = 0;
    for (let y = 0.33; y < 0.61; y += 0.005)
        for (let x = 0.31; x < 0.69; x += 0.005) {
            const i =
                (Math.floor(y * image.height) * image.width + Math.floor(x * image.width)) * 4;
            const [r, g, b] = image.rgba.slice(i, i + 3);
            covered += Number(Math.abs(r - 25) + Math.abs(g - 24) + Math.abs(b - 22) > 25);
            accents += Number(Math.max(r, g, b) - Math.min(r, g, b) > 35);
            total++;
        }
    expect(covered / total).toBeGreaterThan(0.8);
    expect(accents / total).toBeGreaterThan(0.01);
    expect(errors).toEqual([]);
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (directory) {
        mkdirSync(directory, { recursive: true });
        writeFileSync(`${directory}/frame.png`, Buffer.from(image.png.split(",")[1], "base64"));
    }
});
