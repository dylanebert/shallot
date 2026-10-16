import { mkdirSync, writeFileSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";

type State = {
    targets: { outlined: boolean; active: number; height: number }[];
    prompt: number;
    ticks: number;
};
declare global {
    interface Window {
        ready: boolean;
        aim(index: number, distance?: number, yaw?: number, pitch?: number): void;
        key(down: boolean): void;
        advance(count?: number): State;
        state(): State;
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        turnAway(): void;
        overview(): void;
        evidence(): Promise<{ outline: number; prompt: number }>;
        promptEvidence(): Promise<number[]>;
        lampEvidence(): Promise<{ changed: number; increase: number }>;
        holdEvidence(): Promise<{
            beforePosition: number[];
            afterPosition: number[];
            changed: number;
        }>;
    }
}
const STEPPED = `http://127.0.0.1:4186/tests/stepped.html${process.env.SHALLOT_CAPTURE_DIR ? "?captures" : ""}`;

async function capture(page: Page, name: string) {
    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    const data = await page.evaluate(() => window.capture());
    if (process.env.SHALLOT_CAPTURE_DIR) {
        mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
        writeFileSync(
            `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
            Buffer.from(data.split(",")[1]!, "base64"),
        );
    }
    return data;
}

test("aim is sampled before E; in-reach lamp and prompt render, E lights the wall, and far aim clears both", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    console.log("interact adapter", await page.evaluate(() => window.adapter));
    await page.evaluate(() => {
        window.aim(0);
        window.turnAway();
        window.key(true);
        window.advance();
    });
    const turned = await page.evaluate(() => window.state());
    expect(turned.prompt).toBe(0);
    expect(turned.targets[0]!.active).toBe(0);
    await page.evaluate(() => {
        window.key(false);
        window.advance();
        window.aim(0);
    });
    const focused = await page.evaluate(() => window.state());
    expect(focused.targets[0].outlined).toBe(true);
    expect(focused.prompt).toBe(1);
    const evidence = await page.evaluate(() => window.evidence());
    expect(evidence.outline).toBeGreaterThan(15);
    expect(evidence.prompt).toBeGreaterThan(15);
    await capture(page, "lamp-focused");
    const lighting = await page.evaluate(() => window.lampEvidence());
    console.log("lamp wall", lighting);
    expect(lighting.changed).toBeGreaterThan(0.14);
    expect(lighting.increase).toBeGreaterThan(8);
    const switched = (await page.evaluate(() => window.state())).targets[0]!;
    expect(switched.active).toBe(1);
    await page.evaluate(() => {
        window.key(false);
        window.advance();
        window.aim(0, 6);
    });
    const far = await page.evaluate(() => window.state());
    expect(far.prompt).toBe(0);
    expect(far.targets.every((t) => !t.outlined)).toBe(true);
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await page.evaluate(() => window.overview());
        await capture(page, "room-unfocused");
    }
    expect(errors).toEqual([]);
});

test("every prompt part renders at near, far and oblique views during a hold", async ({ page }) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    for (const [distance, yaw, pitch] of [
        [1.2, 0, 0],
        [1.8, -0.45, 0.25],
        [2.6, 0.45, -0.25],
    ]) {
        await page.evaluate(
            ([d, y, p]) => {
                window.aim(2, d, y, p);
                window.key(true);
                window.advance(30);
            },
            [distance, yaw, pitch],
        );
        const parts = await page.evaluate(() => window.promptEvidence());
        console.log("prompt view", distance, yaw, pitch, parts);
        await capture(page, `hold-view-${distance}`);
        for (const pixels of parts) expect(pixels).toBeGreaterThan(2);
        await page.evaluate(() => {
            window.key(false);
            window.advance();
        });
    }
});

test("a completed hold changes the block colour without moving it", async ({ page }) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    const evidence = await page.evaluate(() => window.holdEvidence());
    console.log("hold colour", evidence);
    await capture(page, "hold-complete");
    expect(evidence.afterPosition).toEqual(evidence.beforePosition);
    expect(evidence.changed).toBeGreaterThan(0.9);
});

test("each prompt faces the player, with a half-filled ring during the one-second hold", async ({
    page,
}) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    for (const [index, name] of ["lamp", "button", "hold"].entries()) {
        await page.evaluate((i) => window.aim(i), index);
        const state = await page.evaluate(() => window.state());
        expect(state.targets[index]!.outlined).toBe(true);
        expect(state.prompt).toBe(1);
        await capture(page, `${name}-focused`);
    }
    await page.evaluate(() => {
        window.key(true);
        window.advance(30);
    });
    expect((await page.evaluate(() => window.state())).ticks).toBe(30);
    await capture(page, "hold-mid-fill");
    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    await page.evaluate(() => window.aim(2, 1.8));
    await capture(page, "hold-close-up");
    await page.evaluate(() => {
        window.key(false);
        window.advance();
        window.aim(0, 1.8);
    });
    await capture(page, "prompt-close-up");
    await page.evaluate(() => window.aim(0));
    await capture(page, "lamp-off");
    await page.evaluate(() => {
        window.key(true);
        window.advance();
    });
    await capture(page, "lamp-on");
});
