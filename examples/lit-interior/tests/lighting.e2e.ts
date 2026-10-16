import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        ready: boolean;
        view(x?: number, y?: number, z?: number, yaw?: number, pitch?: number): void;
        switchView(): void;
        key(down: boolean): void;
        turn(delta: number): void;
        floor(): Promise<number>;
        lamp(): number;
        shadow(kind: string): Promise<{ darkened: number; delta: number }>;
        spot(): Promise<{ x: number; y: number; sum: number }>;
        flashlightWall(): void;
        capture(): Promise<string>;
        device: { requests: { requiredLimits?: unknown }[]; adapter: unknown; limits: unknown };
    }
}
test("E dims its ceiling lamp's floor; carried spot moves with input; objects shadow the floor but the shade does not, on the default device", async ({
    page,
}) => {
    const errors: string[] = [],
        warnings: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "warning" || m.type() === "error") warnings.push(m.text());
    });
    await page.goto("http://127.0.0.1:4216/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    const shadeAtSpawn = await page.evaluate(() => window.shadow("shade"));
    expect(shadeAtSpawn.darkened).toBe(0);
    const device = await page.evaluate(() => window.device);
    console.log("lit-interior device", JSON.stringify(device));
    expect(device.requests.length).toBe(1);
    expect(device.requests[0].requiredLimits ?? {}).toEqual({});
    await page.evaluate(() => window.switchView());
    const on = await page.evaluate(() => window.floor());
    await page.evaluate(() => window.key(true));
    expect(await page.evaluate(() => window.lamp())).toBe(0);
    const off = await page.evaluate(() => window.floor());
    console.log("floor", { on, off });
    expect(on - off).toBeGreaterThan(3);
    await page.evaluate(() => {
        window.key(false);
        window.key(true);
        window.key(false);
    });
    expect(await page.evaluate(() => window.lamp())).toBe(80000);
    const object = await page.evaluate(() => window.shadow("object"));
    console.log("floor shadows", { object, shade: shadeAtSpawn });
    expect(object.darkened).toBeGreaterThan(20);
    await page.evaluate(() => window.flashlightWall());
    const before = await page.evaluate(() => window.spot());
    await page.evaluate(() => window.turn(300));
    const after = await page.evaluate(() => window.spot());
    console.log("flashlight spot", { before, after });
    expect(before.sum).toBeGreaterThan(100);
    expect(after.sum).toBeGreaterThan(100);
    expect(Math.abs(after.x - before.x)).toBeGreaterThan(15);
    expect(warnings.filter((s) => /overflow|pipeline|refus/i.test(s))).toEqual([]);
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
    await capture("spawn-both-lit");
    await page.evaluate(() => {
        window.switchView();
        window.key(true);
        window.key(false);
        window.view();
    });
    await capture("spawn-switched-off");
    await page.evaluate(() => window.flashlightWall());
    await capture("flashlight-wall");
});
