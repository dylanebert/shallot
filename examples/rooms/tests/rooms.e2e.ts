import { spawn } from "node:child_process";
import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Locator, type Page, test } from "playwright/test";

test.use(pipelineWarnings);

const players = (page: Page) => page.locator("#players li");
const action = (page: Page, name: string) => page.getByRole("button", { name, exact: true });
const plus = (page: Page, seat: number) => action(page, `Open actions for seat ${seat}`);
const box = async (locator: Locator) => (await locator.boundingBox())!;
const seatBoxes = async (page: Page) => Promise.all((await players(page).all()).map(box));
const noFocus = async (page: Page) => expect(page.locator(".ec :focus")).toHaveCount(0);
const raised = "rgb(60, 50, 44)";
const line = "rgb(42, 36, 32)";
const muted = "rgb(160, 140, 120)";

async function capture(page: Page, name: string) {
    const directory = process.env.SHALLOT_CONTROLS_CAPTURE_DIR;
    if (!directory) return;
    await page.evaluate(() => document.fonts.ready);
    await page.mouse.move(0, 0);
    await page.screenshot({ path: `${directory}/${name}.png`, scale: "css" });
}

async function server() {
    const child = spawn("bun", ["server.ts", "--port", "0"], {
        cwd: new URL("..", import.meta.url),
        stdio: ["ignore", "pipe", "inherit"],
    });
    const exit = new Promise<void>((resolve) => {
        child.once("close", () => resolve());
    });
    const stop = async () => {
        child.kill();
        await exit;
    };
    try {
        const port = await new Promise<string>((resolve, reject) => {
            let output = "";
            child.stdout.setEncoding("utf8");
            child.stdout.on("data", (chunk: string) => {
                output += chunk;
                const port = /Rooms server listening on port (\d+)/.exec(output)?.[1];
                if (port) resolve(port);
            });
            child.once("error", reject);
            child.once("exit", (code) => reject(new Error(`Bun server exited: ${code}`)));
        });
        return {
            address: `http://127.0.0.1:${port}/`,
            query: `/?server=ws://127.0.0.1:${port}`,
            stop,
        };
    } catch (error) {
        await stop();
        throw error;
    }
}

async function touchTargets(page: Page) {
    for (const target of await page.locator(".ec :is(button, input):visible").all()) {
        const { x, y, width, height } = await box(target);
        expect(width).toBeGreaterThanOrEqual(44);
        expect(height).toBeGreaterThanOrEqual(44);
        expect(x).toBeGreaterThanOrEqual(16);
        expect(y).toBeGreaterThanOrEqual(16);
        expect(x + width).toBeLessThanOrEqual(390 - 16);
        expect(y + height).toBeLessThanOrEqual(844 - 16);
    }
}

for (const phone of [false, true]) {
    const size = phone ? "phone" : "desktop";
    test.describe(size, () => {
        test.use({
            viewport: phone ? { width: 390, height: 844 } : { width: 1280, height: 800 },
            hasTouch: phone,
            isMobile: phone,
        });
        const use = (target: Locator) => (phone ? target.tap() : target.click());

        test("entry discloses a padded code field; local party fills fixed seats", async ({
            page,
        }) => {
            const frames = await holdFrames(page, "#players .member");
            await page.goto("/");
            const lobby = page.getByRole("region", { name: "Join a room" });
            const input = page.getByRole("textbox", { name: "Room code" });
            const enter = action(page, "Enter code");
            await expect(lobby.getByRole("button")).toHaveText([
                "Quick join",
                "Create private room",
                "Enter code",
            ]);
            await expect(input).toBeHidden();
            await expect(enter).toHaveAttribute("aria-expanded", "false");
            const trigger = await box(enter);
            const label = await box(enter.getByText("Enter code"));
            const chevron = await box(enter.locator("svg[data-icon='chevron']"));
            expect(label.x - trigger.x).toBeGreaterThanOrEqual(8);
            expect(trigger.x + trigger.width - chevron.x - chevron.width).toBeGreaterThanOrEqual(8);
            expect(chevron.x - label.x - label.width).toBeGreaterThanOrEqual(8);
            if (phone) await touchTargets(page);
            await capture(page, `entry-closed-${size}`);

            await expect(enter).toHaveCSS("background-color", raised);
            if (!phone) {
                await enter.hover();
                await expect(enter).toHaveCSS("border-top-color", muted);
                await page.mouse.down();
                await expect(enter).toHaveCSS("background-color", line);
                expect(await box(enter)).toEqual(trigger);
                await page.mouse.up();
            } else await use(enter);
            await expect(input).toBeVisible();
            await expect(enter).toHaveAttribute("aria-expanded", "true");
            expect(await box(enter)).toEqual(trigger);
            await expect(enter.locator("svg")).toHaveCSS("rotate", "180deg");
            await noFocus(page);
            await page.keyboard.press("Tab");
            await noFocus(page);
            if (phone) await touchTargets(page);
            await capture(page, `entry-code-${size}`);

            await use(action(page, "Quick join"));
            await expect(players(page)).toHaveText(["You", "", "", ""]);
            await expect(page.getByRole("img", { name: "Human player" })).toHaveAttribute(
                "data-icon",
                "user",
            );
            await expect(page.getByRole("img", { name: "Party leader" })).toHaveAttribute(
                "data-icon",
                "star",
            );
            await expect(page.locator("#status")).toBeHidden();
            await expect(action(page, "Copy room code")).toBeHidden();
            await noFocus(page);
            const before = await seatBoxes(page);
            const [first, second] = before;
            expect(first!.x).toBe(phone ? 16 : 24);
            for (const [index, rect] of before.entries()) {
                expect([rect.x, rect.width, rect.height]).toEqual([
                    first!.x,
                    first!.width,
                    first!.height,
                ]);
                expect(rect.y).toBe(first!.y + index * (first!.height + 8));
            }
            await expect(players(page).nth(1)).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
            await expect(players(page).first().locator(".member")).toHaveCSS(
                "background-color",
                raised,
            );
            for (const seat of [2, 3, 4]) {
                await expect(plus(page, seat).locator("svg[data-icon='plus']")).toHaveCount(1);
                await expect(plus(page, seat)).toHaveAttribute(
                    "title",
                    `Open actions for seat ${seat}`,
                );
            }
            const leave = action(page, "Leave");
            await expect(leave.locator("svg[data-icon='leave']")).toHaveCount(1);
            await expect(leave).toHaveAttribute("title", "Leave");
            if (!phone) {
                await leave.hover();
                await expect(leave).toHaveCSS("cursor", "pointer");
                await expect(leave).toHaveCSS("border-top-color", muted);
            }
            const right = await box(page.locator("#room-actions"));
            expect(right.x).toBeGreaterThan(first!.x + first!.width);
            expect(right.x + right.width).toBe(phone ? 390 - 16 : 1280 - 24);

            await use(plus(page, 3));
            await noFocus(page);
            await expect(plus(page, 3)).toHaveAttribute("aria-expanded", "true");
            const add = action(page, "Add bot");
            await expect(
                players(page).nth(2).getByRole("button", { name: "Add bot" }),
            ).toBeVisible();
            await expect(action(page, "Copy code")).toBeHidden();
            expect(await seatBoxes(page)).toEqual(before);
            await use(add);
            await expect
                .poll(async () => {
                    await frames.run(4);
                    return players(page).allInnerTexts();
                })
                .toEqual(["You", "", "Bot 1", ""]);
            await expect(players(page).nth(2).getByRole("img", { name: "Bot" })).toHaveAttribute(
                "data-icon",
                "bot",
            );
            await expect(action(page, "Remove Bot 1")).toHaveAttribute("title", "Remove Bot 1");
            await expect(page.getByRole("img", { name: "Party leader" })).toHaveCount(1);
            expect(await seatBoxes(page)).toEqual(before);
            await noFocus(page);

            await use(plus(page, 2));
            const popup = await box(add);
            expect(popup.x).toBeGreaterThan(second!.x + second!.width);
            expect(popup.y).toBeGreaterThanOrEqual(second!.y);
            expect(popup.y + popup.height).toBeLessThanOrEqual(second!.y + second!.height);
            expect(popup.y).toBeGreaterThanOrEqual(right.y + right.height);
            await frames.run(2);
            await capture(page, `in-tab-${size}`);
            if (phone) {
                await expect(action(page, "Jump")).toBeVisible();
                await touchTargets(page);
                const name = players(page).nth(2).locator(".member span");
                await name.evaluate((el) => {
                    el.textContent = "A very long player name ".repeat(8);
                });
                expect(await name.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
                expect(await seatBoxes(page)).toEqual(before);
                expect(await box(add)).toEqual(popup);
                expect(await box(page.locator("#room-actions"))).toEqual(right);
                await touchTargets(page);
            }
            await page.keyboard.press("Escape");
            await expect(add).toBeHidden();
            await expect(plus(page, 2)).toHaveAttribute("aria-expanded", "false");
            await use(action(page, "Remove Bot 1"));
            await expect(players(page)).toHaveText(["You", "", "", ""]);
            expect(await seatBoxes(page)).toEqual(before);
            await noFocus(page);
            await page.evaluate(() => {
                window.addEventListener(
                    "keydown",
                    (event) => {
                        if (
                            event.code === "KeyW" &&
                            !event.defaultPrevented &&
                            event.target === document.body
                        )
                            document.body.dataset.gameKey = event.code;
                    },
                    { once: true },
                );
            });
            await page.keyboard.press("KeyW");
            await expect(page.locator("body")).toHaveAttribute("data-game-key", "KeyW");

            await use(action(page, "Drop connection"));
            await expect(page.getByText(/Connection dropped/)).toBeVisible();
            await expect(players(page)).toHaveText(["You", "", "", ""]);
            await expect(plus(page, 1)).toHaveCount(0);
            await expect(plus(page, 2)).toBeDisabled();
            await expect(plus(page, 2)).toHaveCSS("background-color", "rgb(28, 25, 23)");
            await expect(plus(page, 2)).toHaveCSS("color", muted);
            const resume = action(page, "Resume connection");
            await expect(resume).toHaveAttribute("aria-pressed", "true");
            await use(resume);
            await expect
                .poll(async () => {
                    await frames.run(4);
                    return page.locator("#status").textContent();
                })
                .toBe("Rejoined as the same player.");
            await expect(plus(page, 2)).toBeEnabled();
            await noFocus(page);
            await use(leave);
            await expect(action(page, "Quick join")).toBeVisible();
            await noFocus(page);
        });

        test("two tabs join a private room by code; server seats offer copy, not bots", async ({
            page,
            context,
        }) => {
            const host = await server();
            try {
                await context.grantPermissions(["clipboard-read", "clipboard-write"]);
                const twoMembers = "#players li:nth-child(2) .member";
                const frames = await holdFrames(page, twoMembers);
                await page.goto(host.address);
                await use(action(page, "Create private room"));
                await expect(players(page)).toHaveText(["You", "", "", ""]);
                const codeButton = action(page, "Copy room code");
                await expect(codeButton).toHaveText(/^[A-Z0-9]{10}$/);
                await expect(codeButton).toHaveAttribute("title", "Copy room code");
                const code = await codeButton.innerText();
                await expect(codeButton).toHaveAccessibleDescription(code);
                await use(codeButton);
                expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(code);
                await noFocus(page);
                const friend = await context.newPage();
                const friendFrames = await holdFrames(friend, twoMembers);
                await friend.goto(host.address);
                await use(action(friend, "Enter code"));
                await friend.getByRole("textbox", { name: "Room code" }).fill(code);
                await use(action(friend, "Join"));
                await expect(players(friend)).toHaveText(["Player 1", "You", "", ""]);
                await expect(players(page)).toHaveText(["You", "Player 2", "", ""]);
                await noFocus(friend);
                await expect(friend.getByRole("img", { name: "Human player" })).toHaveCount(2);
                await expect(friend.getByRole("img", { name: "Party leader" })).toHaveCount(0);
                await expect(action(friend, "Drop connection")).toBeHidden();
                await use(plus(friend, 3));
                await expect(plus(friend, 3)).toHaveAttribute("aria-expanded", "true");
                const fallback = players(friend).nth(2).getByRole("button", { name: "Copy code" });
                await expect(fallback).toBeVisible();
                await expect(action(friend, "Add bot")).toBeHidden();
                await expect(
                    friend.getByText(code, { exact: true }).filter({ visible: true }),
                ).toHaveCount(1);
                expect(await friend.locator("#session").innerText()).toMatch(
                    new RegExp(`^Player 1\\s+You\\s+Copy code\\s+${code}$`),
                );
                const [seat, actions] = await Promise.all([
                    box(players(friend).first()),
                    box(friend.locator("#room-actions")),
                ]);
                expect(actions.x).toBeGreaterThan(seat.x + seat.width);
                const choice = await box(fallback);
                expect(choice.x + choice.width).toBeLessThanOrEqual(
                    (phone ? 390 : 1280) - (phone ? 16 : 24),
                );
                if (phone) await touchTargets(friend);
                await friendFrames.run(2);
                await capture(friend, `server-${size}`);
                await friend.evaluate(() => navigator.clipboard.writeText(""));
                await use(fallback);
                expect(await friend.evaluate(() => navigator.clipboard.readText())).toBe(code);
                await expect(fallback).toBeHidden();
                await expect(players(friend)).toHaveText(["Player 1", "You", "", ""]);
                await noFocus(friend);
                await use(action(friend, "Leave"));
                await expect
                    .poll(async () => {
                        await frames.run(4);
                        return players(page).allInnerTexts();
                    })
                    .toEqual(["You", "", "", ""]);
            } finally {
                await host.stop();
            }
        });
    });
}

test("a dropped server seat stays occupied for its peer and resumes", async ({ page, context }) => {
    const host = await server();
    try {
        const twoMembers = "#players li:nth-child(2) .member";
        const frames = await holdFrames(page, twoMembers);
        const observer = await context.newPage();
        const observerFrames = await holdFrames(observer, twoMembers);
        let reconnect: (() => void) | undefined;
        let attempts = 0;
        await page.routeWebSocket("**/*", (route) => {
            if (++attempts === 1) route.connectToServer();
            else
                reconnect = () => {
                    route.connectToServer();
                };
        });
        await page.addInitScript(() => {
            const sockets: WebSocket[] = [];
            const NativeSocket = WebSocket;
            window.WebSocket = class extends NativeSocket {
                constructor(url: string | URL, protocols?: string | string[]) {
                    super(url, protocols);
                    sockets.push(this);
                }
            };
            Object.assign(window, {
                dropRoomSocket: () => sockets.at(-1)!.close(4001, "Test cut"),
            });
        });
        await page.goto(host.query);
        await action(page, "Quick join").click();
        await expect(players(page)).toHaveText(["You", "", "", ""]);
        const codeButton = action(page, "Copy room code");
        await expect(codeButton).toHaveText(/^[A-Z0-9]{10}$/);
        const code = await codeButton.innerText();
        await observer.goto(host.address);
        await action(observer, "Quick join").click();
        await expect(players(observer)).toHaveText(["Player 1", "You", "", ""]);
        await expect(players(page)).toHaveText(["You", "Player 2", "", ""]);
        const before = await seatBoxes(observer);
        await page.evaluate(() =>
            (window as Window & { dropRoomSocket?: () => void }).dropRoomSocket!(),
        );
        await expect.poll(() => reconnect !== undefined).toBe(true);
        await observerFrames.run(8);
        await expect(players(observer)).toHaveText(["Player 1", "You", "", ""]);
        await expect(plus(observer, 1)).toHaveCount(0);
        expect(await seatBoxes(observer)).toEqual(before);
        reconnect!();
        await expect
            .poll(async () => {
                await frames.run(4);
                return page.locator("#status").textContent();
            })
            .toBe("Rejoined as the same player.");
        await expect(players(page)).toHaveText(["You", "Player 2", "", ""]);
        await expect(codeButton).toHaveText(code);
    } finally {
        await host.stop();
    }
});
