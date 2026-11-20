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

async function expectSidePopup(page: Page, slot: number) {
    const panes = await seatBoxes(page);
    const seat = panes[slot - 1]!;
    const group = page.getByRole("group", { name: `Seat ${slot} actions` });
    const popup = await box(group);
    const right = await box(page.locator("#room-actions"));
    const { width, height } = page.viewportSize()!;
    const inset = width <= 480 ? 16 : 24;
    expect(popup.x).toBe(seat.x + seat.width + 8);
    expect([popup.width, popup.height]).not.toEqual([seat.width, seat.height]);
    for (const pane of [...panes, right])
        expect(
            popup.x >= pane.x + pane.width ||
                popup.x + popup.width <= pane.x ||
                popup.y >= pane.y + pane.height ||
                popup.y + popup.height <= pane.y,
        ).toBe(true);
    expect(popup.y).toBeGreaterThanOrEqual(inset);
    expect(popup.x + popup.width).toBeLessThanOrEqual(width - inset);
    expect(popup.y + popup.height).toBeLessThanOrEqual(height - inset);
    await expect(group).toHaveCSS("display", "grid");
    await expect(group).toHaveCSS("padding-left", "8px");
    await expect(plus(page, slot)).toHaveCSS("border-top-color", "rgb(212, 149, 96)");
    expect(await box(plus(page, slot))).toEqual(seat);
    return popup;
}
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

async function expectCopyFeedback(page: Page, button: Locator) {
    await expect(button).toHaveAccessibleName("Room code copied");
    await expect(button.locator("svg")).toHaveAttribute("data-icon", "check");
    await expect(page.locator("#status")).toBeHidden();
    const announcement = page.getByText("Room code copied", { exact: true });
    await expect(announcement).toHaveCount(1);
    await expect(announcement).toHaveAttribute("role", "status");
    await expect(announcement).toHaveCSS("clip-path", "inset(50%)");
    await expect(button.locator("svg")).toHaveAttribute("data-icon", "copy");
    await expect(announcement).toHaveCount(0);
}

async function expectProfilerToggle(page: Page, advance: () => Promise<void>) {
    const profiler = page.locator("[data-shallot-profile]");
    await expect(profiler).toBeHidden();
    await page.keyboard.press("F3");
    await expect
        .poll(async () => {
            await advance();
            return profiler.isVisible();
        })
        .toBe(true);
    await page.keyboard.press("F3");
    await expect
        .poll(async () => {
            await advance();
            return profiler.isVisible();
        })
        .toBe(false);
}

async function expectEntryRoute(page: Page) {
    const { width, height } = page.viewportSize()!;
    await expect
        .poll(async () => {
            const png = await page.screenshot({
                clip: { x: width / 2, y: height / 2, width: 16, height: height / 2 },
            });
            return page.evaluate(
                async (bytes) => {
                    const bitmap = await createImageBitmap(
                        new Blob([Uint8Array.from(bytes)], { type: "image/png" }),
                    );
                    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext(
                        "2d",
                    )!;
                    context.drawImage(bitmap, 0, 0);
                    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
                    const colors = new Set<number>();
                    for (let i = 0; i < data.length; i += 4)
                        colors.add((data[i]! << 16) | (data[i + 1]! << 8) | data[i + 2]!);
                    return colors.size;
                },
                [...png],
            );
        })
        .toBeGreaterThan(2);
}

async function expectBotClear(page: Page) {
    const png = await page.screenshot({ scale: "css" });
    const bounds = await page.evaluate(
        async (bytes) => {
            const bitmap = await createImageBitmap(
                new Blob([Uint8Array.from(bytes)], { type: "image/png" }),
            );
            const { width, height } = bitmap;
            const context = new OffscreenCanvas(width, height).getContext("2d")!;
            context.drawImage(bitmap, 0, 0);
            const { data } = context.getImageData(0, 0, width, height);
            let left = width,
                right = -1,
                bottom = -1;
            for (let y = 0; y < height; y++)
                for (let x = 0; x < width; x++) {
                    const i = 4 * (y * width + x);
                    const r = data[i]!,
                        g = data[i + 1]!,
                        b = data[i + 2]!;
                    if (g > r + 20 && b > r + 20) {
                        left = Math.min(left, x);
                        right = Math.max(right, x);
                        bottom = y;
                    }
                }
            return { left, right, bottom, width, height };
        },
        [...png],
    );
    expect(bounds.right - bounds.left).toBeGreaterThan(16);
    expect(bounds.left).toBeGreaterThanOrEqual(16);
    expect(bounds.right).toBeLessThan(bounds.width - 16);
    // The start of the floor hint is below this line at both authored viewports.
    expect(bounds.bottom).toBeLessThan(bounds.height * 0.71);
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

        test("entry offers direct code entry; local party fills fixed seats", async ({ page }) => {
            const frames = await holdFrames(page, "#players .member");
            await page.goto("/");
            const lobby = page.getByRole("region", { name: "Join a room" });
            const input = page.getByRole("textbox", { name: "Room code" });
            const quick = action(page, "Quick join");
            const create = action(page, "Create private room");
            await expect(lobby.getByRole("button")).toHaveText([
                "Quick join",
                "Create private room",
                "Join",
            ]);
            await expect(lobby.locator("[aria-expanded]")).toHaveCount(0);
            await expect(input).toBeVisible();
            await expect(create.locator("svg")).toHaveAttribute("data-icon", "lock");
            await expect(create).toHaveAccessibleDescription("Only people with the code can join");
            for (const side of ["left", "right"])
                expect(
                    await input.evaluate(
                        (el, side) =>
                            parseFloat(getComputedStyle(el).getPropertyValue(`padding-${side}`)),
                        side,
                    ),
                ).toBeGreaterThanOrEqual(8);
            await expect(action(page, "Jump")).toBeHidden();
            await expectEntryRoute(page);
            if (!phone) await expectProfilerToggle(page, frames.run);
            await expect(create).toHaveCSS("background-color", raised);
            if (!phone) {
                const trigger = await box(create);
                await create.hover();
                await expect(create).toHaveAttribute("title", "Only people with the code can join");
                await expect(create).toHaveCSS("border-top-color", muted);
                await page.mouse.down();
                await expect(create).toHaveCSS("background-color", line);
                expect(await box(create)).toEqual(trigger);
                await page.mouse.move(0, 0);
                await page.mouse.up();
            }
            await noFocus(page);
            await page.keyboard.press("Tab");
            await noFocus(page);
            if (phone) await touchTargets(page);
            await capture(page, `entry-${size}`);

            await use(quick);
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
            if (!phone) await expectProfilerToggle(page, frames.run);
            await noFocus(page);
            const before = await seatBoxes(page);
            const [first] = before;
            expect(first!.x).toBe(phone ? 16 : 24);
            for (const [index, rect] of before.entries()) {
                expect([rect.x, rect.width, rect.height]).toEqual([
                    first!.x,
                    first!.width,
                    first!.height,
                ]);
                expect(rect.y).toBe(first!.y + index * (first!.height + 8));
            }
            await expect(players(page).first().locator(".member")).toHaveCSS(
                "background-color",
                raised,
            );
            for (const seat of [2, 3, 4]) {
                await expect(plus(page, seat)).toHaveCSS("background-color", "rgb(28, 25, 23)");
                expect(await box(plus(page, seat))).toEqual(before[seat - 1]);
                await expect(
                    players(page)
                        .nth(seat - 1)
                        .getByRole("button"),
                ).toHaveCount(1);
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

            await use(plus(page, 2));
            await expectSidePopup(page, 2);
            await use(plus(page, 3));
            await expectSidePopup(page, 3);
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

            await use(plus(page, 4));
            const menu = page.getByRole("group", { name: "Seat 4 actions" });
            const popup = await expectSidePopup(page, 4);
            await frames.run(2);
            await expectBotClear(page);
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
                expect(await box(menu)).toEqual(popup);
                expect(await box(page.locator("#room-actions"))).toEqual(right);
                await touchTargets(page);
            }
            await page.keyboard.press("Escape");
            await expect(add).toBeHidden();
            await expect(plus(page, 4)).toHaveAttribute("aria-expanded", "false");
            await use(plus(page, 4));
            await use(players(page).first().locator(".member"));
            await expect(add).toBeHidden();
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
            await expect
                .poll(async () => {
                    await frames.run(3);
                    return action(page, "Quick join").isVisible();
                })
                .toBe(true);
            await frames.run(3);
            await expectEntryRoute(page);
            await noFocus(page);
        });

        test("two server pages share a bot; only the leader can manage it", async ({
            page,
            context,
        }) => {
            const host = await server();
            try {
                await context.grantPermissions(["clipboard-read", "clipboard-write"]);
                const twoMembers = "#players li:nth-child(2) .member";
                const leaderFrames = await holdFrames(page, twoMembers);
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
                await expectCopyFeedback(page, page.locator("#copy"));
                await noFocus(page);
                await use(plus(page, 2));
                await expectSidePopup(page, 2);
                if (phone) await touchTargets(page);
                await use(action(page, "Add bot"));
                await expect(players(page)).toHaveText(["You", "Bot 2", "", ""]);
                await leaderFrames.run(3);
                await use(plus(page, 3));
                await expectSidePopup(page, 3);
                await expect(
                    page.getByRole("group", { name: "Seat 3 actions" }).getByRole("button"),
                ).toHaveText(["Add bot", "Copy code"]);
                await expectBotClear(page);
                await capture(page, `server-${size}`);
                await page.keyboard.press("Escape");
                await use(action(page, "Remove Bot 2"));
                await expect
                    .poll(async () => {
                        await leaderFrames.run(2);
                        return players(page).allInnerTexts();
                    })
                    .toEqual(["You", "", "", ""]);
                const friend = await context.newPage();
                const friendFrames = await holdFrames(friend, twoMembers);
                await friend.goto(host.address);
                await friend.getByRole("textbox", { name: "Room code" }).fill(code);
                await use(action(friend, "Join"));
                await expect(players(friend)).toHaveText(["Player 1", "You", "", ""]);
                await expect
                    .poll(async () => {
                        await leaderFrames.run(2);
                        return players(page).allInnerTexts();
                    })
                    .toEqual(["You", "Player 2", "", ""]);
                await noFocus(friend);
                await expect(friend.getByRole("img", { name: "Human player" })).toHaveCount(2);
                for (const tab of [page, friend]) {
                    const star = tab.getByRole("img", { name: "Party leader" });
                    await expect(star).toHaveCount(1);
                    await expect(
                        players(tab).first().getByRole("img", { name: "Party leader" }),
                    ).toBeVisible();
                }
                await expect(action(friend, "Drop connection")).toBeHidden();
                await use(plus(friend, 4));
                await expect(plus(friend, 4)).toHaveAttribute("aria-expanded", "true");
                const fallback = players(friend).nth(3).getByRole("button", { name: "Copy code" });
                await expect(fallback).toBeVisible();
                await expectSidePopup(friend, 4);
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
                await friend.evaluate(() => navigator.clipboard.writeText(""));
                await use(fallback);
                expect(await friend.evaluate(() => navigator.clipboard.readText())).toBe(code);
                await expectCopyFeedback(
                    friend,
                    players(friend).nth(3).locator(".slot-actions button"),
                );
                await expect(players(friend)).toHaveText(["Player 1", "You", "", "Copy code"]);
                await noFocus(friend);
                await use(plus(page, 3));
                await use(action(page, "Add bot"));
                await expect
                    .poll(async () => {
                        await leaderFrames.run(2);
                        return action(page, "Remove Bot 3").isVisible();
                    })
                    .toBe(true);
                await expect
                    .poll(async () => {
                        await friendFrames.run(2);
                        return friend.getByRole("img", { name: "Bot" }).count();
                    })
                    .toBe(1);
                await expect(action(friend, "Add bot")).toHaveCount(0);
                await expect(action(friend, "Remove Bot 3")).toHaveCount(0);
                await capture(friend, `server-observer-${size}`);
                await use(action(page, "Leave"));
                await friend.keyboard.press("Escape");
                await expect
                    .poll(async () => {
                        await friendFrames.run(4);
                        return players(friend).allInnerTexts();
                    })
                    .toEqual(["", "You", "Bot 3", ""]);
                await expect(friend.getByRole("img", { name: "Party leader" })).toHaveCount(1);
                await expect(
                    players(friend).nth(1).getByRole("img", { name: "Party leader" }),
                ).toBeVisible();
                await use(action(friend, "Remove Bot 3"));
                await expect
                    .poll(async () => {
                        await friendFrames.run(2);
                        return players(friend).allInnerTexts();
                    })
                    .toEqual(["", "You", "", ""]);
                for (const slot of [1, 3, 4]) {
                    await use(plus(friend, slot));
                    const choice = action(friend, "Copy code");
                    await expectSidePopup(friend, slot);
                    expect(
                        await choice.evaluate((button) => button.scrollWidth <= button.clientWidth),
                    ).toBe(true);
                    if (phone) await touchTargets(friend);
                    if (slot === 1) await capture(friend, `server-seat-one-${size}`);
                    await friend.keyboard.press("Escape");
                    await expect(choice).toBeHidden();
                }
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
