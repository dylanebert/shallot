import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        ready: boolean;
        view(x?: number, y?: number, z?: number, yaw?: number, pitch?: number): void;
        regions(): Promise<{
            sunlitFloor: number;
            darkFloor: number;
            pools: number[];
        }>;
        shadow(
            kind: "block" | "bulbs",
        ): Promise<{ darkened: number; darkenedFraction: number; maxDelta: number }>;
        appearance(): Promise<{
            window: number;
            sunlitFloor: number;
            wallRgb: number[];
            wallL: number;
            clearL: number;
            ceilingL: number;
            floorL: number;
        }>;
        capture(): Promise<string>;
        device: { requests: { requiredLimits?: unknown }[]; adapter: unknown; limits: unknown };
    }
}

test("the sunlit floor and both pendant pools outshine the dark stretch", async ({ page }) => {
    const errors: string[] = [],
        warnings: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "warning" || m.type() === "error") warnings.push(m.text());
    });
    await page.goto("http://127.0.0.1:4216/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    await test.step("daylight through the window outshines its floor patch, and the dark room stays legible against the clear", async () => {
        const appearance = await page.evaluate(() => window.appearance());
        console.log("window and dark room", appearance);
        expect.soft(appearance.window).toBeGreaterThan(appearance.sunlitFloor + 20);
        expect.soft(appearance.wallL - appearance.clearL).toBeGreaterThan(0.03);
        expect.soft(appearance.wallL - appearance.ceilingL).toBeGreaterThan(0.015);
        expect.soft(appearance.ceilingL - appearance.floorL).toBeGreaterThan(0.015);
    });
    const lit = await page.evaluate(() => window.regions());
    console.log("sunlit floor and pendant pool luminance", lit);
    expect.soft(lit.sunlitFloor - lit.darkFloor).toBeGreaterThan(30);
    for (const pool of lit.pools) expect.soft(pool - lit.darkFloor).toBeGreaterThan(15);
    const device = await page.evaluate(() => window.device);
    console.log("lit-interior device", JSON.stringify(device));
    expect(device.requests).toHaveLength(1);
    expect(device.requests[0]!.requiredLimits ?? {}).toEqual({});
    expect(warnings.filter((s) => /overflow|pipeline|refus|GPUValidationError/i.test(s))).toEqual(
        [],
    );
    expect(errors).toEqual([]);

    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    const dir = process.env.SHALLOT_CAPTURE_DIR;
    mkdirSync(dir, { recursive: true });
    await page.goto("http://127.0.0.1:4216/tests/stepped.html?captures");
    await page.waitForFunction(() => window.ready);
    console.log("capture adapter", await page.evaluate(() => window.device));
    async function capture(name: string) {
        const data = await page.evaluate(() => window.capture());
        writeFileSync(`${dir}/${name}.png`, Buffer.from(data.split(",")[1]!, "base64"));
    }
    await capture("spawn");
    await page.evaluate(() => window.view(1, 1, -7, Math.PI, -0.1));
    await capture("shaft-toward-pendants");
    await page.evaluate(() => window.view(1, 1, 5, 0, 0));
    await capture("pendants-toward-window");
});

test("a pendant's block shadows visible floor in front of it, but its emissive bulbs do not", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4216/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    const block = await page.evaluate(() => window.shadow("block"));
    const bulbs = await page.evaluate(() => window.shadow("bulbs"));
    console.log("floor shadows", { block, bulbs });
    expect(bulbs.darkened).toBe(0);
    expect(bulbs.maxDelta).toBeLessThan(1);
    expect(block.darkenedFraction).toBeGreaterThan(10 / (640 * 360));
    expect(block.maxDelta).toBeGreaterThan(15);
});
