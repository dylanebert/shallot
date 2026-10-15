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
    }
}
async function capture(page: Page, name: string) {
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

test("aiming within reach frames the outlined door and prompt; E opens it; beyond reach clears both", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    console.log("interact adapter", await page.evaluate(() => window.adapter));
    await page.evaluate(() => {
        window.aim(0);
        window.aim(1);
    });
    const focused = await page.evaluate(() => window.state());
    expect(focused.targets[1].outlined).toBe(true);
    expect(focused.prompt).toBe(1);
    const evidence = await page.evaluate(() => window.evidence());
    expect(evidence.outline).toBeGreaterThan(300);
    expect(evidence.prompt).toBeGreaterThan(300);
    await capture(page, "door-focused");
    await page.evaluate(() => {
        window.key(true);
        window.advance();
    });
    const opened = (await page.evaluate(() => window.state())).targets[1]!;
    expect(opened.active).toBe(1);
    expect(opened.height).toBe(4);
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

test("a fast turn samples the new aim before E's fixed action", async ({ page }) => {
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
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
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    for (const [index, name] of ["lamp", "door", "button", "hold"].entries()) {
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
});
