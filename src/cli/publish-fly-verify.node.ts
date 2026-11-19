import { expect, setDefaultTimeout, test } from "bun:test";
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { stripVTControlCharacters } from "node:util";
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
                // Under CI, Vite colors its output even when piped, splitting the URL with escapes.
                const match = pattern.exec(stripVTControlCharacters(output));
                if (match) accept(match[1]!);
            });
            child.once("exit", (code) => reject(new Error(`Child exited ${code}: ${output}`)));
            child.once("error", reject);
        });
    try {
        const port = await address(server, /ws:\/\/localhost:(\d+)/);
        const endpoint = `ws://127.0.0.1:${port}`;
        // Own the Vite process itself so teardown also closes its inherited stderr on Windows.
        const viteBin = resolve(import.meta.dir, "../../node_modules/vite/bin/vite.js");
        vite = spawn("node", [viteBin, "--host", "127.0.0.1", "--port", "0"], {
            cwd: example,
            env: { ...process.env, VITE_SHALLOT_SERVER: endpoint },
            stdio: ["ignore", "pipe", "inherit"],
        });
        const url = await address(vite, /(http:\/\/127\.0\.0\.1:\d+\/)/);
        browser = await publishTestChromium();
        const published = await browser.newPage();
        const verify = await observePublishedServer(published, endpoint, 5000);
        await published.addInitScript(() => {
            const style = document.createElement("style");
            style.textContent = "[data-shallot-profile] { display: none !important }";
            document.addEventListener("DOMContentLoaded", () => document.head.append(style));
        });
        await published.goto(url);
        await published.locator("[data-measured-rtt]").filter({ hasText: /ms/ }).waitFor();
        await verify();
        await published.close();
        const wrong = await browser.newPage();
        const verifyWrong = await observePublishedServer(
            wrong,
            "ws://wrong-endpoint.invalid",
            5000,
        );
        await wrong.goto(url);
        await wrong.locator("[data-measured-rtt]").filter({ hasText: /ms/ }).waitFor();
        await expect(verifyWrong()).rejects.toThrow(
            "Published page received no frames from ws://wrong-endpoint.invalid",
        );
        await wrong.close();
        // An explicit empty query selects in-tab play, rather than the build's socket.
        const page = await browser.newPage();
        const verifyInTab = await observePublishedServer(page, endpoint, 5000);
        await page.addInitScript(() => {
            const state: { first?: number; tick?: number; endpoint?: string } = {};
            Object.defineProperty(window, "__inTabState", { value: state });
            window.addEventListener("shallot:startup", (event) => {
                const detail = (event as CustomEvent).detail;
                if (detail.kind !== "server-state") return;
                state.first ??= detail.tick;
                state.tick = detail.tick;
                state.endpoint = detail.endpoint;
            });
        });
        const received = new Promise<void>((accept) =>
            page.on("websocket", (socket) => {
                if (socket.url() === `${endpoint}/`) socket.once("framereceived", () => accept());
            }),
        );
        await page.goto(`${url}?server=`);
        await page.getByText("In-tab server · thin client", { exact: true }).waitFor();
        await page.evaluate(
            (endpoint) =>
                Object.defineProperty(window, "unusedSocket", { value: new WebSocket(endpoint) }),
            endpoint,
        );
        await received;
        await page.waitForFunction(() => {
            const state = (window as unknown as { __inTabState: { first?: number; tick?: number } })
                .__inTabState;
            return (
                state.first !== undefined &&
                state.tick !== undefined &&
                state.tick >= state.first + 60
            );
        });
        expect(
            await page.evaluate(
                () =>
                    (window as unknown as { __inTabState: { endpoint?: string } }).__inTabState
                        .endpoint,
            ),
        ).toBeUndefined();
        await expect(verifyInTab()).rejects.toThrow(
            `Published page did not advance confirmed server state by 60 ticks from ${endpoint}`,
        );
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
