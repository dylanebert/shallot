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

test("an in-tab quick join shows its players and lets the player manage bots", async ({
    page,
}, info) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await expect(page.getByText("No public rooms yet.")).toBeVisible();
    await action(page, "Quick join").click();
    await expect(players(page)).toHaveText(["AsterYou"]);
    await action(page, "Add bot").click();
    await expect(players(page)).toHaveText(["AsterYou", "BirchBot"]);
    await action(page, "Add bot").click();
    await expect(players(page)).toHaveText(["AsterYou", "BirchBot", "CedarBot"]);
    if (process.env.SHALLOT_ROOMS_CAPTURES)
        await page.screenshot({ path: info.outputPath("desktop.png") });
    await action(page, "Remove bot").click();
    await expect(players(page)).toHaveText(["AsterYou", "BirchBot"]);
    await action(page, "Leave room").click();
    await expect(action(page, "Quick join")).toBeVisible();
});

test("two pages share a private Bun room by code without listing it publicly", async ({
    page,
    context,
}) => {
    const host = await server();
    try {
        await page.goto(host.query);
        await action(page, "Create private room").click();
        await expect(players(page)).toHaveText(["AsterYou"]);
        const code = (await page.locator("#room-code").innerText()).replace("Room code: ", "");
        const friend = await context.newPage();
        await friend.goto(host.query);
        await expect(friend.getByText("No public rooms yet.")).toBeVisible();
        await friend.getByRole("textbox", { name: "Room code" }).fill(code);
        await action(friend, "Join by code").click();
        await expect(players(friend)).toHaveText(["AsterPlayer", "BirchYou"]);
        await expect(players(page)).toHaveText(["AsterYou", "BirchPlayer"]);
        await expect(action(friend, "Add bot")).toBeHidden();
        await expect(action(friend, "Drop connection")).toBeHidden();
        await action(friend, "Leave room").click();
        await expect(players(page)).toHaveText(["AsterYou"]);
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
        await expect(players(page)).toHaveText(["AsterYou"]);
        await page.evaluate(() =>
            (window as Window & { dropRoomSocket?: () => void }).dropRoomSocket!(),
        );
        await expect(page.getByText("Rejoined as the same player.")).toBeVisible();
        await expect(players(page)).toHaveText(["AsterYou"]);
    } finally {
        await host.stop();
    }
});

test.describe("phone", () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
    test("the room panel fits a touch viewport and accepts taps", async ({ page }, info) => {
        await page.goto("/");
        await action(page, "Quick join").tap();
        await expect(players(page)).toHaveText(["AsterYou"]);
        for (const name of ["BirchBot", "CedarBot"]) {
            await action(page, "Add bot").tap();
            await expect(players(page).last()).toHaveText(name);
        }
        const panel = page.getByRole("complementary", { name: "Rooms" });
        const box = (await panel.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(390);
        expect(box.y).toBeGreaterThanOrEqual(0);
        expect(box.y + box.height).toBeLessThan(844 / 2);
        for (const button of await panel.getByRole("button").all())
            expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44);
        if (process.env.SHALLOT_ROOMS_CAPTURES)
            await page.screenshot({ path: info.outputPath("phone.png") });
        await action(page, "Drop connection").tap();
        await expect(page.getByText(/Connection dropped/)).toBeVisible();
        await action(page, "Resume connection").tap();
        await expect(page.getByText("Rejoined as the same player.")).toBeVisible();
    });
});
