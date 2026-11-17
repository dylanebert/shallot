import { expect, setDefaultTimeout, test } from "bun:test";
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { publishTestChromium } from "../../scripts/publish-browser";
import { observePublishedServer } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const example = resolve(import.meta.dir, "../../examples/multiplayer");

test("publish observes confirmed network state at the build's endpoint and query overrides it", async () => {
    const server = spawn("bun", ["server.ts", "--host", "127.0.0.1", "--port", "0"], {
        cwd: example,
        stdio: ["ignore", "pipe", "inherit"],
    });
    let vite: ReturnType<typeof spawn> | undefined;
    let browser: Awaited<ReturnType<typeof publishTestChromium>> | undefined;
    const address = (child: ReturnType<typeof spawn>, pattern: RegExp) =>
        new Promise<string>((accept, reject) => {
            let output = "";
            child.stdout!.on("data", (chunk) => {
                output += chunk;
                const match = pattern.exec(output);
                if (match) accept(match[1]!);
            });
            child.once("exit", (code) => reject(new Error(`Child exited ${code}: ${output}`)));
            child.once("error", reject);
        });
    try {
        const port = await address(server, /ws:\/\/localhost:(\d+)/);
        const endpoint = `ws://127.0.0.1:${port}`;
        vite = spawn("bun", ["x", "vite", "--host", "127.0.0.1", "--port", "0"], {
            cwd: example,
            env: { ...process.env, VITE_SHALLOT_SERVER: endpoint },
            stdio: ["ignore", "pipe", "inherit"],
        });
        const url = await address(vite, /(http:\/\/127\.0\.0\.1:\d+\/)/);
        browser = await publishTestChromium();
        const published = await browser.newPage();
        const verify = observePublishedServer(published, endpoint);
        await published.goto(url);
        await published.locator("[data-measured-rtt]").filter({ hasText: /ms/ }).waitFor();
        await verify();
        await published.close();
        const wrong = await browser.newPage();
        const verifyWrong = observePublishedServer(wrong, "ws://wrong-endpoint.invalid");
        await wrong.goto(url);
        await wrong.locator("[data-measured-rtt]").filter({ hasText: /ms/ }).waitFor();
        await expect(verifyWrong()).rejects.toThrow(
            "Published page did not connect to ws://wrong-endpoint.invalid",
        );
        await wrong.close();
        // An explicit empty query selects in-tab play, rather than the build's socket.
        const page = await browser.newPage();
        await page.goto(`${url}?server=`);
        await page.getByText("In-tab server · thin client", { exact: true }).waitFor();
        await page.close();
        expect(browser.isConnected()).toBe(true);
    } finally {
        await browser?.close();
        const stop = async (child: ReturnType<typeof spawn> | undefined) => {
            if (!child || child.exitCode !== null) return;
            const exited = new Promise<void>((accept) => child.once("exit", () => accept()));
            child.kill();
            await exited;
        };
        await stop(vite);
        await stop(server);
    }
});
