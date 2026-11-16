import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

test.use(pipelineWarnings);
type State = {
    latency?: number;
    loss?: number;
    slot: number;
    confirmed: number;
    tick: number;
    position: number[];
    remote: number[];
    box: number[];
    lift: number[];
    yaw: number;
    jumps: number;
    predictedBox: boolean;
};
type Game = {
    frames(count?: number): Promise<State>;
    state(): State;
    key(key: string, down: boolean): void;
    touch(id: number, x: number, y: number, held?: boolean): void;
    cancel(id: number): void;
    avatarPixels(): Promise<number>;
};
declare global {
    interface Window {
        game: Game;
        ready: boolean;
    }
}
async function frames(page: Page, count = 1) {
    return page.evaluate((count) => window.game.frames(count), count);
}
async function open(page: Page, url = "") {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`/tests/stepped.html${url}`);
    await page.waitForFunction(() => window.ready);
    await expect.poll(async () => (await frames(page, 2)).slot).toBeGreaterThanOrEqual(0);
    return errors;
}

test.beforeEach(async ({ page }) => {
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
});

test("an unanswered server address explains how to start the server", async ({ page }) => {
    await page.goto("/?server=ws://127.0.0.1:4432");
    await expect(
        page.getByText("No server answered at ws://127.0.0.1:4432.", { exact: false }),
    ).toBeVisible();
    await expect(
        page.getByText("bun server.ts --host 127.0.0.1 --port 3001", { exact: false }),
    ).toBeVisible();
});

for (const protocol of ["ws", "wss"]) {
    test(`an unreachable remote ${protocol} server does not recommend a local server`, async ({
        page,
    }) => {
        const address = `${protocol}://multiplayer.invalid`;
        await page.goto(`/?server=${address}`);
        await expect(
            page.getByText(`No server answered at ${address}.`, { exact: false }),
        ).toBeVisible();
        await expect(page.getByText("bun server.ts", { exact: false })).toHaveCount(0);
        await expect(
            page.getByText("Check that its server is running and reachable", { exact: false }),
        ).toBeVisible();
        await expect(page.getByText("RUNNING.md", { exact: false })).toBeVisible();
    });
}

test("a server closing before the first image explains the failed connection", async ({ page }) => {
    await page.routeWebSocket("ws://127.0.0.1:4432", (socket) => socket.close());
    await page.goto("/?server=ws://127.0.0.1:4432");
    await expect(
        page.getByText("No server answered at ws://127.0.0.1:4432.", { exact: false }),
    ).toBeVisible();
    await expect(page.getByText("RUNNING.md", { exact: false })).toBeVisible();
});

test("the page entry starts its in-tab game and shared bootstrap", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const held = await holdFrames(page, "[aria-label='Network simulator']");
    const response = await page.goto("/");
    expect(response?.status()).toBe(200);
    expect(await page.title()).toBe("multiplayer");
    await expect(page.locator("[data-shallot-bootstrap], [data-shallot-loading]")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Network simulator" })).toBeVisible();
    await held.run(15);
    expect(errors).toEqual([]);
});

test("in-tab latency and loss are live, a thin client moves visibly, and local input pushes the shared box", async ({
    page,
}) => {
    const errors = await open(page);
    const start = await frames(page, 30);
    expect(await page.evaluate(() => window.game.avatarPixels())).toBeGreaterThan(5);
    const later = await frames(page, 30);
    // Client prediction dilates its own ticks, never the server or transport clock.
    expect(later.confirmed - start.confirmed).toBeGreaterThanOrEqual(29);
    expect(later.confirmed - start.confirmed).toBeLessThanOrEqual(31);
    expect(Math.hypot(...later.remote.map((v, i) => v - start.remote[i]!))).toBeGreaterThan(0.3);
    expect(later.lift[1]).not.toBe(start.lift[1]);
    const latency = page.getByLabel("Round trip", { exact: true });
    await latency.evaluate((input: HTMLInputElement) => {
        input.value = "200";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.locator('output[for="latency"]')).toHaveText("200ms");
    await page.getByLabel("Packet loss", { exact: true }).evaluate((input: HTMLInputElement) => {
        input.value = "4";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.locator('output[for="loss"]')).toHaveText("4%");
    const configured = await frames(page);
    expect(configured.latency).toBe(200);
    expect(configured.loss).toBe(0.04);
    await page.evaluate(() => window.game.key("KeyW", true));
    const pushed = await frames(page, 130);
    await page.evaluate(() => window.game.key("KeyW", false));
    expect(pushed.position[2]).toBeLessThan(start.position[2] - 2);
    expect(pushed.box[2]).toBeLessThan(start.box[2] - 1);
    expect(pushed.predictedBox).toBe(true);
    expect(pushed.confirmed).toBeGreaterThan(start.confirmed + 90);
    expect(errors).toEqual([]);
});

test("two pages share the Bun server: one player's input moves the remote avatar and pushes the same box", async ({
    page,
    context,
}) => {
    const other = await context.newPage();
    const server = "?server=ws://127.0.0.1:4431";
    const errors = await open(page, server);
    const otherErrors = await open(other, server);
    const start = await frames(page, 2),
        remoteStart = await frames(other, 2);
    expect([start.slot, remoteStart.slot].sort()).toEqual([0, 1]);
    await expect(page.getByLabel("Round trip", { exact: true })).toBeDisabled();
    await page.evaluate(() => window.game.key("KeyW", true));
    let own = start,
        remote = remoteStart;
    await expect
        .poll(async () => {
            own = await frames(page, 2);
            remote = await frames(other, 2);
            return Math.abs(remote.remote[2]! - remoteStart.remote[2]!);
        })
        .toBeGreaterThan(2);
    await page.evaluate(() => window.game.key("KeyW", false));
    await expect
        .poll(async () => {
            own = await frames(page, 2);
            remote = await frames(other, 2);
            return Math.max(own.box[2]!, remote.box[2]!);
        })
        .toBeLessThan(start.box[2]! - 1);
    expect(remote.position[0]).toBeCloseTo(remoteStart.position[0]!, 2);
    expect(errors.concat(otherErrors)).toEqual([]);
});

test.describe("phone controls", () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });
    test("floating stick, look and Jump coexist with the network panel at 390×844", async ({
        page,
    }) => {
        const errors = await open(page);
        const initial = await frames(page, 45);
        await page.evaluate(() => {
            window.game.touch(1, 80, 620);
            window.game.touch(1, 80, 550);
            window.game.touch(2, 280, 450);
            window.game.touch(2, 310, 450);
        });
        const moved = await frames(page, 15);
        expect(moved.position[2]).toBeLessThan(initial.position[2]! - 0.2);
        expect(moved.yaw).not.toBe(initial.yaw);
        await expect(page.locator("[data-touch-stick]")).toBeVisible();
        const jump = page.getByRole("button", { name: "Jump", exact: true });
        await expect(jump).toBeVisible();
        const bounds = await jump.boundingBox(),
            panel = await page.getByRole("region", { name: "Network simulator" }).boundingBox();
        expect(bounds!.y).toBeGreaterThan(panel!.y + panel!.height);
        expect(panel!.x).toBeGreaterThanOrEqual(0);
        expect(panel!.x + panel!.width).toBeLessThanOrEqual(390);
        await page.evaluate(() => {
            window.game.cancel(1);
            window.game.cancel(2);
        });
        await frames(page, 30);
        const beforeJump = await frames(page, 1);
        await jump.dispatchEvent("pointerdown", { pointerId: 3 });
        const jumped = await frames(page, 8);
        await jump.dispatchEvent("pointerup", { pointerId: 3 });
        expect(jumped.jumps).toBeGreaterThan(beforeJump.jumps);
        expect(jumped.position[1]).toBeGreaterThan(beforeJump.position[1]! + 0.2);
        await expect(page.locator("[data-touch-stick]")).toBeHidden();
        expect(errors).toEqual([]);
    });
});
