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
        return { query: `/?server=ws://127.0.0.1:${port}`, stop };
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
    await expect(page.getByRole("textbox", { name: "Room code" })).toBeHidden();
    await expect(page.getByText("Public rooms")).toBeHidden();
    await page.getByText("Have a code?").click();
    await expect(action(page, "Join")).toBeVisible();
    await action(page, "Quick join").click();
    await expect(page.locator("#party-count")).toHaveText("1 / 4");
    await expect(players(page)).toHaveCount(4);
    await expect(players(page).first()).toContainText("You");
    await expect(page.getByLabel("Party leader")).toBeVisible();
    await action(page, "Copy code").click();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(/^[A-Z0-9]{10}$/);
    await action(page, "Empty slot 3").click();
    await action(page, "Add bot").click();
    await expect(players(page).nth(2)).toContainText("Bot 1");
    await expect(page.locator("#party-count")).toHaveText("2 / 4");
    await action(page, "Remove Bot 1").click();
    await expect(action(page, "Empty slot 3")).toBeVisible();
    await action(page, "Leave room").click();
    await expect(action(page, "Quick join")).toBeVisible();
});

test("a private Bun party accepts its code and its copied invite URL", async ({
    page,
    context,
}) => {
    const host = await server();
    try {
        await context.grantPermissions(["clipboard-read", "clipboard-write"]);
        await page.goto(host.query);
        await action(page, "Create private room").click();
        await expect(players(page).first()).toHaveText("You");
        await expect(page.locator("#room-code")).toHaveText(/^[A-Z0-9]{10}$/);
        const code = await page.locator("#room-code").innerText();
        await action(page, "Empty slot 2").click();
        await action(page, "Copy invite").last().click();
        const invite = await page.evaluate(() => navigator.clipboard.readText());
        expect(new URL(invite).searchParams.get("code")).toBe(code);
        const friend = await context.newPage();
        await friend.goto(host.query);
        await friend.getByText("Have a code?").click();
        await friend.getByRole("textbox", { name: "Room code" }).fill(code);
        await action(friend, "Join").click();
        await expect(players(friend).nth(0)).toHaveText("Player 1");
        await expect(players(friend).nth(1)).toHaveText("You");
        await expect(players(page).nth(1)).toHaveText("Player 2");
        await expect(friend.locator("#party-crown")).toBeHidden();
        await expect(action(friend, "Add bot")).toBeHidden();
        await expect(action(friend, "Drop connection")).toBeHidden();
        await action(friend, "Leave room").click();
        await expect(page.locator("#party-count")).toHaveText("1 / 4");
        await friend.goto(invite);
        await expect(friend.locator("#party-count")).toHaveText("2 / 4");
        await expect(players(friend).nth(1)).toHaveText("You");
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
        await expect(players(page).first()).toHaveText("You");
        await expect(page.locator("#room-code")).toHaveText(/^[A-Z0-9]{10}$/);
        const code = await page.locator("#room-code").innerText();
        await page.evaluate(() =>
            (window as Window & { dropRoomSocket?: () => void }).dropRoomSocket!(),
        );
        await expect(page.getByText("Rejoined as the same player.")).toBeVisible();
        await expect(players(page).first()).toHaveText("You");
        await expect(page.locator("#room-code")).toHaveText(code);
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
