import { mkdirSync, writeFileSync } from "node:fs";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        snow: {
            step(n?: number): void;
            position(): number[];
            point(): [number, number];
            capture(): Promise<{ png: string; rgba: number[]; width: number; height: number }>;
        };
    }
}
test.use(pipelineWarnings);
for (const phone of [false, true]) {
    test.describe(phone ? "phone" : "desktop", () => {
        test.use({
            viewport: phone ? { width: 390, height: 844 } : { width: 1280, height: 720 },
            hasTouch: phone,
        });
        test("the first frame has no late-pipeline warning", async ({ page }) => {
            await page.goto("/");
            await page.waitForFunction(() =>
                performance
                    .getEntriesByType("measure")
                    .some((e) => e.name.endsWith(":first-update")),
            );
        });
        test("the stepped snow covers the centre and the boulder can be dragged", async ({
            page,
        }) => {
            await page.goto("/tests/stepped.html");
            await page.waitForFunction(() => window.snow);
            const frame = await page.evaluate(() => window.snow.capture());
            let covered = 0,
                total = 0;
            for (let y = 0.4; y < 0.6; y += 0.01)
                for (let x = 0.4; x < 0.6; x += 0.01) {
                    const i =
                        (Math.floor(y * frame.height) * frame.width + Math.floor(x * frame.width)) *
                        4;
                    covered += Number(frame.rgba[i] + frame.rgba[i + 1] + frame.rgba[i + 2] > 180);
                    total++;
                }
            expect(covered / total).toBeGreaterThan(0.9);
            const directory = process.env.SHALLOT_CAPTURE_DIR;
            if (directory) {
                mkdirSync(directory, { recursive: true });
                writeFileSync(
                    `${directory}/${phone ? "390x844" : "desktop"}.png`,
                    Buffer.from(frame.png.split(",")[1], "base64"),
                );
            }
            const before = await page.evaluate(() => window.snow.position());
            const [x, y] = await page.evaluate(() => window.snow.point());
            if (phone) {
                const session = await page.context().newCDPSession(page);
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchStart",
                    touchPoints: [{ x, y }],
                });
                await page.evaluate(() => window.snow.step());
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchMove",
                    touchPoints: [{ x: x + 65, y: y - 30 }],
                });
                await page.evaluate(() => window.snow.step(45));
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchEnd",
                    touchPoints: [],
                });
            } else {
                await page.mouse.move(x, y);
                await page.mouse.down();
                await page.evaluate(() => window.snow.step());
                await page.mouse.move(x + 100, y - 30);
                await page.evaluate(() => window.snow.step(45));
                await page.mouse.up();
            }
            const after = await page.evaluate(() => window.snow.position());
            expect(Math.hypot(after[0] - before[0], after[2] - before[2])).toBeGreaterThan(1);
        });
    });
}
