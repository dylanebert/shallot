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
        playerWall(): void;
        ambient(value: number): void;
        fixtures(): Promise<number[]>;
        capture(): Promise<string>;
        device: { requests: { requiredLimits?: unknown }[]; adapter: unknown; limits: unknown };
    }
}
test("E dims its ceiling lamp's floor; objects shadow the floor but the shade does not, on the default device", async ({
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
    const shade = await page.evaluate(() => window.shadow("shade"));
    expect(shade.darkened).toBe(0);
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
    console.log("floor shadows", { object, shade });
    expect(object.darkened).toBeGreaterThan(20);
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
    await page.evaluate(() => window.playerWall());
    await capture("flashlight-wall");
    await page.evaluate(() => window.ambient(300));
    await capture("flashlight-wall-ambient-300");
});

test("turning the player with core input moves the carried flashlight's bright spot across a stationary view", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("http://127.0.0.1:4216/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    await page.evaluate(() => window.flashlightWall());
    const before = await page.evaluate(() => window.spot());
    await page.evaluate(() => window.turn(300));
    const after = await page.evaluate(() => window.spot());
    console.log("flashlight spot", { before, after });
    expect(before.sum).toBeGreaterThan(100);
    expect(after.sum).toBeGreaterThan(100);
    expect(Math.abs(after.x - before.x)).toBeGreaterThan(15);
    expect(errors).toEqual([]);
});

test("both fixtures show bright sources; E extinguishes only its bulb and restores its emission", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4216/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    const on = await page.evaluate(() => window.fixtures());
    await page.evaluate(() => {
        window.switchView();
        window.key(true);
        window.key(false);
    });
    const off = await page.evaluate(() => window.fixtures());
    await page.evaluate(() => {
        window.switchView();
        window.key(true);
        window.key(false);
    });
    const restored = await page.evaluate(() => window.fixtures());
    console.log("fixture sources", { on, off, restored });
    expect(on[0]).toBeGreaterThan(180);
    expect(on[1]).toBeGreaterThan(180);
    expect(on[0]! - off[0]!).toBeGreaterThan(100);
    expect(Math.abs(on[1]! - off[1]!)).toBeLessThan(5);
    expect(Math.abs(on[0]! - restored[0]!)).toBeLessThan(5);
});
