import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

type Frame = { hour: number; width: number; height: number; rgba: number[]; png: string };
declare global {
    interface Window {
        frame: (hour?: number) => Promise<Frame>;
    }
}

function mean(frame: Frame, x0: number, y0: number, x1: number, y1: number) {
    let sum = 0,
        count = 0;
    for (let y = Math.floor(y0 * frame.height); y < y1 * frame.height; y++) {
        for (let x = Math.floor(x0 * frame.width); x < x1 * frame.width; x++) {
            const i = (y * frame.width + x) * 4;
            sum += 0.2126 * frame.rgba[i] + 0.7152 * frame.rgba[i + 1] + 0.0722 * frame.rgba[i + 2];
            count++;
        }
    }
    return sum / count;
}

test("the slider drives the hour, noon sky is brighter than midnight, and shadows cross the shapes", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
    await page.waitForFunction(() => !!window.frame);
    const frames: Record<string, Frame> = {};
    for (const [name, hour] of Object.entries({
        dawn: 6,
        morning: 9,
        noon: 12,
        afternoon: 15,
        dusk: 18,
        midnight: 0,
    })) {
        await page.getByRole("slider", { name: "Hour" }).fill(String(hour));
        frames[name] = await page.evaluate(() => window.frame());
        expect(frames[name].hour).toBe(hour);
    }
    expect(mean(frames.noon, 0.1, 0.05, 0.9, 0.25)).toBeGreaterThan(
        mean(frames.midnight, 0.1, 0.05, 0.9, 0.25) + 30,
    );
    // Mirrored patches beside the central pillar exclude its silhouette and compare the ground's shadow.
    const contrast = (f: Frame) =>
        mean(f, 0.44, 0.495, 0.47, 0.505) - mean(f, 0.53, 0.495, 0.56, 0.505);
    if (process.env.SHALLOT_CAPTURE_DIR) {
        mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
        const save = (name: string, frame: Frame) =>
            writeFileSync(
                `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
                Buffer.from(frame.png.split(",")[1], "base64"),
            );
        for (const [name, frame] of Object.entries(frames)) save(name, frame);
        for (let i = 0; i < 12; i++)
            save(
                `low-sun-${String(i).padStart(2, "0")}`,
                await page.evaluate((hour) => window.frame(hour), 7 + i / 150),
            );
    }
    expect(contrast(frames.morning) * contrast(frames.afternoon)).toBeLessThan(-4);
    expect(errors).toEqual([]);
});
