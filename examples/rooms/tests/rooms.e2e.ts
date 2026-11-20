import { spawn } from "node:child_process";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

test.use(pipelineWarnings);

const players = (page: Page) => page.locator("#players li");
const action = (page: Page, name: string) => page.getByRole("button", { name, exact: true });

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

test("quick join opens a party with fixed slots and local bot controls", async ({
    page,
    context,
}) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto("/");
    const input = page.getByRole("textbox", { name: "Room code" });
    await expect(input).toBeHidden();
    await expect(page.getByText("Public rooms")).toBeHidden();
    const enter = page.getByText("Enter code");
    await enter.hover();
    await expect(enter).toHaveCSS("border-top-color", "rgb(160, 140, 120)");
    await enter.click();
    await page.keyboard.press("Tab");
    await expect(input.and(page.locator(":focus-visible"))).toBeFocused();
    await input.hover();
    await expect(input).toHaveCSS("border-top-color", "rgb(160, 140, 120)");
    await page.mouse.down();
    await expect(input).toHaveCSS("background-color", "rgb(42, 36, 32)");
    await page.mouse.up();
    await expect(action(page, "Join")).toBeVisible();
    await action(page, "Quick join").click();
    await expect(page.locator("#party-count")).toHaveText("1 / 4");
    await expect(players(page)).toHaveCount(4);
    await expect(players(page).first()).toContainText("You");
    await expect(page.getByLabel("You, party leader")).toBeVisible();
    await action(page, "Copy room code").click();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(/^[A-Z0-9]{10}$/);
    await action(page, "Empty slot 3").click();
    await action(page, "Add bot").click();
    await expect(players(page).nth(2)).toContainText("Bot 1");
    await expect(page.locator("#party-count")).toHaveText("2 / 4");
    await action(page, "Remove Bot 1").click();
    await expect(action(page, "Empty slot 3")).toBeVisible();
    await action(page, "Leave").click();
    await expect(action(page, "Quick join")).toBeVisible();
});

test("two tabs at the same Bun URL join a private party by its copied code", async ({
    page,
    context,
}) => {
    const host = await server();
    try {
        await context.grantPermissions(["clipboard-read", "clipboard-write"]);
        await page.goto(host.address);
        await action(page, "Create private room").click();
        await expect(players(page).first()).toContainText("You");
        await expect(page.locator("#share-code")).toHaveText(/^[A-Z0-9]{10}$/);
        const code = await page.locator("#share-code").innerText();
        await action(page, "Copy room code").click();
        expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(code);
        const friend = await context.newPage();
        await friend.goto(host.address);
        await friend.getByText("Enter code").click();
        await friend.getByRole("textbox", { name: "Room code" }).fill(code);
        await action(friend, "Join").click();
        await expect(players(friend).nth(0)).toContainText("Player 1");
        await expect(players(friend).nth(1)).toContainText("You");
        await expect(players(page).nth(1)).toContainText("Player 2");
        await expect(friend.locator("#party-crown")).toBeHidden();
        await expect(action(friend, "Empty slot 3")).toBeHidden();
        await expect(action(friend, "Drop connection")).toBeHidden();
        await action(friend, "Leave").click();
        await expect(page.locator("#party-count")).toHaveText("1 / 4");
    } finally {
        await host.stop();
    }
});

test("a served page resumes its own avatar when its socket is dropped", async ({ page }) => {
    const host = await server();
    try {
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
        await expect(players(page).first()).toContainText("You");
        await expect(page.locator("#share-code")).toHaveText(/^[A-Z0-9]{10}$/);
        const code = await page.locator("#share-code").innerText();
        await page.evaluate(() =>
            (window as Window & { dropRoomSocket?: () => void }).dropRoomSocket!(),
        );
        await expect(page.getByText("Rejoined as the same player.")).toBeVisible();
        await expect(players(page).first()).toContainText("You");
        await expect(page.locator("#share-code")).toHaveText(code);
    } finally {
        await host.stop();
    }
});

test.describe("phone", () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
    test("the party stays compact on touch and opens for slot and connection actions", async ({
        page,
    }) => {
        await page.goto("/");
        await action(page, "Quick join").tap();
        const panel = page.getByRole("complementary", { name: "Party" });
        const toggle = page.locator("#room-toggle");
        await expect(page.locator("#room-code")).toBeVisible();
        await expect(page.locator("#party-count")).toHaveText("1 / 4");
        await expect(page.locator("#room-panel")).not.toHaveAttribute("open", "");
        await expect(page.locator("#party-crown")).toBeVisible();
        const box = (await panel.boundingBox())!;
        expect(box.height).toBeLessThanOrEqual(64);
        expect(box.x + box.width).toBeLessThanOrEqual(390);
        await expect(action(page, "Jump")).toBeVisible();
        await toggle.tap();
        await action(page, "Empty slot 2").tap();
        await action(page, "Add bot").tap();
        await expect(page.locator("#party-count")).toHaveText("2 / 4");
        for (const button of await panel.getByRole("button").all())
            expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44);
        await action(page, "Drop connection").tap();
        await expect(page.getByText(/Connection dropped/)).toBeVisible();
        await action(page, "Resume connection").tap();
        await expect(page.getByText("Rejoined as the same player.")).toBeVisible();
    });
});
