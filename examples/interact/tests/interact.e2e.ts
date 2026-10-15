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
        aim(index: number, distance?: number): void;
        key(down: boolean): void;
        advance(count?: number): State;
        state(): State;
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        turnAway(): void;
        overview(): void;
        evidence(): Promise<{ outline: number; prompt: number }>;
        lampEvidence(): Promise<{ changed: number; increase: number }>;
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

test("aiming within reach frames the outlined lamp and prompt; E switches it; beyond reach clears both", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    console.log("interact adapter", await page.evaluate(() => window.adapter));
    await page.evaluate(() => {
        window.aim(0);
    });
    const focused = await page.evaluate(() => window.state());
    expect(focused.targets[0].outlined).toBe(true);
    expect(focused.prompt).toBe(1);
    const evidence = await page.evaluate(() => window.evidence());
    expect(evidence.outline).toBeGreaterThan(15);
    expect(evidence.prompt).toBeGreaterThan(15);
    await capture(page, "lamp-focused");
    await page.evaluate(() => {
        window.key(true);
        window.advance();
    });
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
    await page.evaluate(() => window.overview());
    await capture(page, "room-unfocused");
    expect(errors).toEqual([]);
});

test("switching the lamp visibly lights the bare back wall", async ({ page }) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    await page.evaluate(() => window.aim(0));
    const evidence = await page.evaluate(() => window.lampEvidence());
    console.log("lamp wall", evidence);
    expect(evidence.changed).toBeGreaterThan(0.14);
    expect(evidence.increase).toBeGreaterThan(8);
});

test("a fast turn samples the new aim before E's fixed action", async ({ page }) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => window.ready);
    await page.evaluate(() => {
        window.aim(0);
        window.turnAway();
        window.key(true);
        window.advance();
    });
    const state = await page.evaluate(() => window.state());
    expect(state.prompt).toBe(0);
    expect(state.targets[0]!.active).toBe(0);
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
