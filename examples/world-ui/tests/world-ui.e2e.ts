import { mkdir, writeFile } from "node:fs/promises";
import { expect, type Page, test } from "playwright/test";

type Frame = {
    adapter: unknown;
    png: string;
    icon: number;
    iconWidth: number;
    iconVisible: number;
    iconCenter: number[];
    iconProjection: number[];
    fill: number;
    name: number;
    barFilled: number[];
    barEmpty: number;
    emptyRgb: number[];
    besideTip: number[];
    trail: number[];
    restored?: boolean;
};
declare global {
    interface Window {
        deviceRequests: GPUDeviceDescriptor[];
        delivery: {
            capture(png?: boolean): Promise<Frame>;
            background(): Promise<Frame>;
            advance(seconds: number, pitch: number): void;
        };
    }
}
test.describe("stepped delivery", () => {
    test.describe.configure({ mode: "serial" });
    let page: Page;
    let spawn: Frame;
    let low: Frame;
    let earlyExit: Frame;
    let gone: Frame;
    let spawnBehind: Frame;
    let lowBehind: Frame;
    let requests: GPUDeviceDescriptor[];
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        page.on("pageerror", (error) => errors.push(error.message));
        await page.addInitScript(() => {
            window.deviceRequests = [];
            const requestDevice = GPUAdapter.prototype.requestDevice;
            GPUAdapter.prototype.requestDevice = function (descriptor = {}) {
                window.deviceRequests.push(descriptor);
                return requestDevice.call(this, descriptor);
            };
        });
        await page.goto("http://127.0.0.1:4194/tests/stepped.html");
        await page.waitForFunction(() => window.delivery !== undefined);
        const png = !!process.env.SHALLOT_CAPTURE_DIR;
        spawn = await page.evaluate((png) => window.delivery.capture(png), png);
        spawnBehind = await page.evaluate(() => window.delivery.background());
        await page.evaluate(() => window.delivery.advance(3, 0.2));
        low = await page.evaluate((png) => window.delivery.capture(png), png);
        lowBehind = await page.evaluate(() => window.delivery.background());
        await page.evaluate(() => window.delivery.advance(3.14, 0.85));
        earlyExit = await page.evaluate((png) => window.delivery.capture(png), png);
        await page.evaluate(() => window.delivery.advance(0.26, 0.85));
        gone = await page.evaluate(() => window.delivery.capture());
        requests = await page.evaluate(() => window.deviceRequests);
        console.log("world-ui adapter", spawn.adapter);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
            for (const [name, frame] of [
                ["spawn", spawn],
                ["mid-drive-low", low],
                ["mid-bounce", earlyExit],
            ] as const) {
                await writeFile(
                    `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
                    Buffer.from(frame.png.split(",")[1], "base64"),
                );
            }
        }
    });
    test.afterAll(async () => {
        await page?.close();
    });
    test.afterEach(() => {
        expect(errors).toEqual([]);
    });
    test("a captured frame shows the building icon and the car's registered-font name at their projections", () => {
        expect(spawn.icon).toBeGreaterThan(100);
        for (let axis = 0; axis < 2; axis++)
            expect(
                Math.abs(spawn.iconCenter[axis] - spawn.iconProjection[axis]),
            ).toBeLessThanOrEqual(1);
        expect(spawn.name).toBeGreaterThan(30);
    });
    test("the car's cyan horizontal fill shows progress without colouring the empty track", () => {
        expect(spawn.fill).toBe(0);
        for (const sample of spawn.barFilled) expect(sample).toBe(0);
        expect(low.fill).toBeCloseTo(0.5, 6);
        for (const sample of low.barFilled) expect(sample).toBeGreaterThan(1);
        for (const frame of [spawn, low]) expect(frame.barEmpty).toBe(0);
    });
    test("the empty neutral track separates from the ground at spawn and the clear at low orbit", () => {
        for (const [frame, behind] of [
            [spawn, spawnBehind],
            [low, lowBehind],
        ]) {
            const difference = Math.max(
                ...frame.emptyRgb.map((channel, i) => Math.abs(channel - behind.emptyRgb[i])),
            );
            expect(difference).toBeGreaterThan(20);
            expect(Math.max(...frame.emptyRgb) - Math.min(...frame.emptyRgb)).toBeLessThan(20);
        }
    });
    test("showing the backing again restores every authored Sprite value", () => {
        for (const behind of [spawnBehind, lowBehind]) expect(behind.restored).toBe(true);
    });
    test("the trail reaches the car's rear as a plain line, without an arrowhead", () => {
        for (const frame of [low, earlyExit]) expect(frame.besideTip).toEqual([0, 0]);
    });
    test("zero-delta captured frames show the trail between road start and car", () => {
        for (const frame of [low, earlyExit])
            for (const sample of frame.trail) expect(sample).toBeGreaterThan(3);
    });
    test("early in the back-in exit the icon is at least as large as at spawn, at the same centre", () => {
        expect(earlyExit.iconWidth).toBeGreaterThanOrEqual(spawn.iconWidth);
        expect(earlyExit.icon).toBeGreaterThanOrEqual(spawn.icon);
        for (let axis = 0; axis < 2; axis++) {
            expect(
                Math.abs(earlyExit.iconCenter[axis] - spawn.iconCenter[axis]),
            ).toBeLessThanOrEqual(1);
            expect(
                Math.abs(earlyExit.iconCenter[axis] - earlyExit.iconProjection[axis]),
            ).toBeLessThanOrEqual(1);
        }
    });
    test("after the 0.35-second exit no icon pixels remain in its unscaled screen region", () => {
        expect(spawn.iconVisible).toBeGreaterThan(100);
        expect(gone.iconVisible).toBe(0);
    });
    test("the device requests no requiredLimits, absent or empty", () => {
        expect(requests.length).toBeGreaterThan(0);
        for (const request of requests)
            expect(Object.keys(request.requiredLimits ?? {})).toEqual([]);
    });
});
test("the copied production page presents the delivery scene", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await expect(page.locator("canvas")).toBeVisible();
    await expect
        .poll(() =>
            page.locator("canvas").evaluate((canvas) => (canvas as HTMLCanvasElement).width),
        )
        .toBeGreaterThan(1);
    expect(errors).toEqual([]);
});
