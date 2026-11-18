import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
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
    removeNetworkDiagnostics(): void;
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
// Socket prediction holds at maxAhead while stepped frames outrun the server's clock, so a
// subject times motion in client ticks, not frames.
async function ticks(page: Page, count: number) {
    const { state, end } = await page.evaluate(async (count) => {
        let state = window.game.state();
        const end = state.tick + count;
        for (let frame = 0; frame < count * 20 && state.tick < end; frame++)
            state = await window.game.frames();
        return { state, end };
    }, count);
    expect(state.tick).toBeGreaterThanOrEqual(end);
    return state;
}
const servers: ChildProcess[] = [];
test.afterEach(() => {
    for (const server of servers.splice(0)) server.kill();
});
/** Starts a Bun server on a free port, so a subject that moves the scene starts from its route. */
async function serve() {
    const server = spawn("bun", ["server.ts", "--port", "0", "--host", "127.0.0.1"], {
        cwd: import.meta.dirname + "/..",
        stdio: ["ignore", "pipe", "inherit"],
    });
    servers.push(server);
    let output = "";
    const port = await new Promise<string>((resolve, reject) => {
        server.stdout!.on("data", (chunk: Buffer) => {
            output += chunk;
            const port = /ws:\/\/localhost:(\d+)/.exec(output)?.[1];
            if (port) resolve(port);
        });
        server.on("exit", (code) => reject(new Error(`Bun server exited ${code}: ${output}`)));
    });
    return `ws://127.0.0.1:${port}`;
}
async function open(page: Page, url = "") {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`/tests/stepped.html${url}`);
    await page.waitForFunction(() => window.ready);
    await expect.poll(async () => (await frames(page, 2)).slot).toBeGreaterThanOrEqual(0);
    return errors;
}

for (const server of [false, true]) {
    test(`F3+R saves replayable ${server ? "socket" : "in-tab"} recordings`, async ({ page }) => {
        await page.setViewportSize({ width: 640, height: 360 });
        await open(page, `?record${server ? `&server=${await serve()}` : ""}`);
        const start = await frames(page, 10);
        await page.evaluate(() => window.game.key("KeyW", true));
        const pushed = await ticks(page, 130);
        await page.evaluate(() => window.game.key("KeyW", false));
        expect(pushed.box[2]).toBeLessThan(start.box[2]! - 1);
        expect(pushed.predictedBox).toBe(true);
        const yawCount = await page.evaluate(async (socket) => {
            window.game.key("KeyW", true);
            window.game.touch(91, innerWidth * 0.8, innerHeight * 0.4);
            const yaws = new Set<number>();
            for (let frame = 0; frame < 600; frame++) {
                window.game.touch(91, innerWidth * 0.8 + (frame + 1) * 0.73, innerHeight * 0.4);
                yaws.add((await window.game.frames()).yaw);
                // Only the external server advances on wall time.
                if (socket)
                    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            }
            window.game.cancel(91);
            window.game.key("KeyW", false);
            return yaws.size;
        }, server);
        expect(yawCount).toBeGreaterThan(500);
        console.log(`yaw sweep ${server ? "socket" : "in-tab"}: ${yawCount} distinct angles`);
        const download = page.waitForEvent("download");
        await page.keyboard.down("F3");
        await page.keyboard.press("r");
        await page.keyboard.up("F3");
        const path = await (await download).path();
        const recordings = JSON.parse(await readFile(path!, "utf8"));
        expect(recordings).toHaveLength(server ? 1 : 2);
        for (const recording of recordings) {
            expect(recording.javascript.engine).toBe("V8");
            expect(recording.javascript.version).toContain("Chrome/");
        }
        execFileSync("bun", ["tests/replay.ts", path!], { cwd: import.meta.dirname + "/.." });
        execFileSync("bun", ["tests/replay.ts", path!, "--drop-receive"], {
            cwd: import.meta.dirname + "/..",
        });
    });
}

test("without record F3+R downloads nothing", async ({ page }) => {
    await open(page);
    const downloads: unknown[] = [];
    page.on("download", (download) => downloads.push(download));
    await page.keyboard.down("F3");
    await page.keyboard.press("r");
    await page.keyboard.up("F3");
    await frames(page, 3);
    expect(downloads).toEqual([]);
});

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

test("F3 has no network section without registered network diagnostics", async ({ page }) => {
    await open(page);
    await page.keyboard.press("F3");
    const title = page.locator("[data-shallot-profile]").getByText("▸ network", { exact: true });
    await expect
        .poll(async () => {
            await frames(page);
            return title.isVisible();
        })
        .toBe(true);
    await page.evaluate(() => window.game.removeNetworkDiagnostics());
    await expect
        .poll(async () => {
            await frames(page);
            return title.isVisible();
        })
        .toBe(false);
});

for (const server of ["", "?server=ws://127.0.0.1:4431"]) {
    for (const viewport of [
        { width: 1280, height: 800 },
        { width: 390, height: 844 },
    ]) {
        test(`measured diagnostics ${server ? "Bun" : "in-tab"} at ${viewport.width}×${viewport.height}`, async ({
            page,
        }, info) => {
            await page.setViewportSize(viewport);
            await page.goto(`/${server}`);
            await expect(page.locator("[data-measured-rtt]")).toHaveText(/^[0-9]+(?:\.[0-9]+)?ms/);
            await page.keyboard.press("F3");
            const overlay = page.locator("[data-shallot-profile]");
            await expect(overlay).toBeVisible();
            await overlay.getByText("▸ network", { exact: true }).click();
            await expect(overlay.getByText("round trip", { exact: true })).toBeVisible();
            await expect(overlay.getByText("time since receive", { exact: true })).toBeVisible();
            await expect
                .poll(() =>
                    page.evaluate(() => {
                        const label = [
                            ...document.querySelectorAll("[data-shallot-profile] span"),
                        ].find((node) => node.textContent === "round trip");
                        return (
                            label?.nextElementSibling?.textContent?.replace(" ms", "ms") ===
                            document.querySelector("[data-measured-rtt]")?.textContent
                        );
                    }),
                )
                .toBe(true);
            if (viewport.width === 1280) {
                const bounds = await overlay.boundingBox();
                const panel = await page
                    .getByRole("region", { name: "Network simulator" })
                    .boundingBox();
                expect(
                    bounds!.x + bounds!.width <= panel!.x || bounds!.y >= panel!.y + panel!.height,
                ).toBe(true);
            }
            await page.screenshot({ path: info.outputPath("network.png") });
        });
    }
}

test("in-tab latency and loss are live, a thin client moves visibly, and local input pushes the shared box", async ({
    page,
}) => {
    const errors = await open(page);
    // Isolate clock pacing from loss/retransmission bursts before testing live faults below.
    await page.getByLabel("Packet loss", { exact: true }).evaluate((input: HTMLInputElement) => {
        input.value = "0";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
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
    const server = `?server=${await serve()}`;
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
