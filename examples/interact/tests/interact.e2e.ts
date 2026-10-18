import { mkdirSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";

type State = {
    targets: { outlined: boolean; active: number; height: number }[];
    prompt: number;
    ticks: number;
};
declare global {
    interface Window {
        ready: boolean;
        view(pixels: boolean): void;
        aim(index: number, distance?: number, yaw?: number, pitch?: number): void;
        key(down: boolean): void;
        advance(count?: number): State;
        state(): State;
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        turnAway(): void;
        overview(): void;
        evidence(): Promise<{ outline: number }>;
        lampEvidence(): Promise<{ changed: number; increase: number }>;
        holdEvidence(): Promise<{
            beforePosition: number[];
            afterPosition: number[];
            changed: number;
        }>;
    }
}
const STEPPED = "http://127.0.0.1:4186/tests/stepped.html";
const PIXELS = `${STEPPED}${process.env.SHALLOT_CAPTURE_DIR ? "?captures" : ""}`;

async function capture(page: Page, name: string) {
    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
    await page.evaluate(() => window.capture());
    await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png` });
}

test.describe("one interaction room", () => {
    test.describe.configure({ mode: "serial" });
    let page: Page;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        const session = await page.context().newCDPSession(page);
        await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto(PIXELS);
        await page.waitForFunction(() => window.ready);
    });
    test.afterAll(() => page.close());
    test.beforeEach(async () => {
        await page.evaluate(() => {
            window.key(false);
            window.advance();
            for (const index of [0, 2]) {
                if (!window.state().targets[index]!.active) continue;
                window.aim(index);
                window.key(true);
                window.advance(index === 2 ? 60 : 1);
                window.key(false);
                window.advance();
            }
            window.view(false);
        });
    });
    test.afterEach(() => expect(errors).toEqual([]));

    test("aim is sampled before E; in-reach lamp and prompt render, E lights the wall, and far aim clears both", async () => {
        await page.evaluate(() => window.view(true));
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
        await expect(page.locator(".hint")).toBeVisible();
        await expect(page.locator(".action")).toHaveText("Switch on");
        await expect(page.locator("kbd")).toHaveText("E");
        await capture(page, "lamp-focused");
        const lighting = await page.evaluate(() => window.lampEvidence());
        console.log("lamp wall", lighting);
        expect(lighting.changed).toBeGreaterThan(0.14);
        expect(lighting.increase).toBeGreaterThan(8);
        const switched = (await page.evaluate(() => window.state())).targets[0]!;
        expect(switched.active).toBe(1);
        await expect(page.locator(".action")).toHaveText("Switch off");
        await expect(page.locator(".hold-ring")).toBeHidden();
        await page.evaluate(() => {
            window.key(false);
            window.advance();
            window.aim(0, 6);
        });
        const far = await page.evaluate(() => window.state());
        expect(far.prompt).toBe(0);
        await expect(page.locator(".hint")).toBeHidden();
        expect(far.targets.every((t) => !t.outlined)).toBe(true);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await page.evaluate(() => window.overview());
            await capture(page, "room-unfocused");
        }
        expect(errors).toEqual([]);
    });

    test("HUD hint stays visible at near, far and oblique views with its hold ring centred on the key", async () => {
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
            await expect(page.locator(".action")).toHaveText("Hold to change");
            await expect(page.locator(".hold-ring")).toBeVisible();
            const key = await page.locator("kbd").boundingBox();
            const ring = await page.locator(".hold-ring").boundingBox();
            expect(key).not.toBeNull();
            expect(ring).not.toBeNull();
            expect(
                Math.abs(key!.x + key!.width / 2 - ring!.x - ring!.width / 2),
            ).toBeLessThanOrEqual(1);
            expect(
                Math.abs(key!.y + key!.height / 2 - ring!.y - ring!.height / 2),
            ).toBeLessThanOrEqual(1);
            console.log("ring/key centres", {
                x: key!.x + key!.width / 2 - ring!.x - ring!.width / 2,
                y: key!.y + key!.height / 2 - ring!.y - ring!.height / 2,
            });
            const action = await page.locator(".action").boundingBox();
            expect(action!.x).toBeGreaterThan(ring!.x + ring!.width);
            expect(Math.hypot(key!.width, key!.height) / 2).toBeLessThan(21);
            await expect(page.locator(".progress")).toHaveCSS("stroke-dasharray", "0.5px, 1px");
            await capture(page, `hold-view-${distance}`);

            await page.evaluate(() => {
                window.key(false);
                window.advance();
            });
        }
        await test.step("a completed hold changes the block colour without moving it", async () => {
            const evidence = await page.evaluate(() => window.holdEvidence());
            console.log("hold colour", evidence);
            await capture(page, "hold-complete");
            expect(evidence.afterPosition).toEqual(evidence.beforePosition);
            expect(evidence.changed).toBeGreaterThan(0.9);
        });
    });

    test("reticle fills only on focus and the hint begins 96px below its centre", async () => {
        // The stepped fixture mounts the game's HUD; also check the shipped shell for stray UI.
        const shell = await (await page.request.get("http://127.0.0.1:4186/")).text();
        const stray = await page.evaluate((html) => {
            const doc = new DOMParser().parseFromString(html, "text/html");
            return [...doc.body.children].filter((el) => !["CANVAS", "SCRIPT"].includes(el.tagName))
                .length;
        }, shell);
        expect(stray).toBe(0);
        await expect(page.locator(".reticle")).toHaveCount(1);
        await expect(page.locator(".reticle-dot")).toHaveCount(1);
        await expect(page.locator(".reticle")).toHaveCSS("border-radius", "50%");
        await expect(page.locator(".reticle-dot")).toHaveCSS("border-radius", "50%");
        for (const [index, name] of ["lamp", "button", "hold"].entries()) {
            await page.evaluate((i) => window.aim(i), index);
            const reticle = page.locator(".reticle");
            const dot = page.locator(".reticle-dot");
            await expect(dot).toBeVisible();
            await expect(reticle).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
            const centre = await reticle.boundingBox();
            const inner = await dot.boundingBox();
            expect(
                Math.abs(inner!.x + inner!.width / 2 - centre!.x - centre!.width / 2),
            ).toBeLessThanOrEqual(1);
            expect(
                Math.abs(inner!.y + inner!.height / 2 - centre!.y - centre!.height / 2),
            ).toBeLessThanOrEqual(1);
            const hint = await page.locator(".hint").boundingBox();
            const bounds = await page.locator(".interact-hud").boundingBox();
            expect(
                Math.abs(centre!.x + centre!.width / 2 - bounds!.x - bounds!.width / 2),
            ).toBeLessThanOrEqual(1);
            expect(
                Math.abs(centre!.y + centre!.height / 2 - bounds!.y - bounds!.height / 2),
            ).toBeLessThanOrEqual(1);
            expect(Math.abs(hint!.y - centre!.y - centre!.height / 2 - 96)).toBeLessThanOrEqual(1);
            await capture(page, `${name}-focused`);
            await page.evaluate((i) => window.aim(i, 6), index);
            await expect(dot).toBeHidden();
            await expect(reticle).toBeVisible();
            await expect(page.locator(".hint")).toBeHidden();
            await capture(page, `${name}-unfocused`);
        }
    });

    test("each HUD hint shows its action, with a half-filled ring during the one-second hold", async () => {
        for (const [index, name] of ["lamp", "button", "hold"].entries()) {
            await page.evaluate((i) => window.aim(i), index);
            const state = await page.evaluate(() => window.state());
            expect(state.targets[index]!.outlined).toBe(true);
            expect(state.prompt).toBe(1);
            await expect(page.locator(".action")).toHaveText(
                ["Switch on", "Press", "Hold to change"][index]!,
            );
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
});
