import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import type { APIResponse, Browser, BrowserType, Page } from "playwright";
import {
    LEAVE_CLOSE_CODE,
    ROOM_AUTH_REFUSED_CLOSE_CODE,
    ROOMS_LIST_HEADER,
    ROOMS_LIST_VERSION,
} from "../extras/multiplayer";
import { CROSS_ORIGIN_ISOLATION, contentType } from "../project";
import { installDev } from "./publish-dependencies";
import { publishProgress } from "./publish-output";

export interface PublishAsset {
    path: string;
    size: number;
}
export interface PublishVerification {
    hardware: boolean;
    adapter: string;
    threads?: number;
    artifact?: "shared" | "single";
    physicsStepped?: boolean;
}

const PUBLISH_LAUNCH_OPTIONS = {
    channel: "chromium",
    chromiumSandbox: true,
    args: ["--enable-unsafe-webgpu"],
};

export async function launchPublishChromium(
    chromium: Pick<BrowserType, "launch">,
    verifyServer = false,
    linuxCI = process.platform === "linux" && !!process.env.CI,
): Promise<Browser> {
    const options = {
        ...PUBLISH_LAUNCH_OPTIONS,
        args: [
            ...PUBLISH_LAUNCH_OPTIONS.args,
            ...(verifyServer && linuxCI
                ? [
                      "--use-angle=swiftshader",
                      "--enable-features=Vulkan",
                      "--use-vulkan=swiftshader",
                  ]
                : []),
        ],
    };
    try {
        return await chromium.launch(options);
    } catch (error) {
        if (!(error instanceof Error) || !error.message.includes("No usable sandbox!")) throw error;
        const browser = await chromium.launch({
            ...options,
            chromiumSandbox: false,
        });
        publishProgress(
            "Verification ran without Chromium's sandbox: Chromium reported No usable sandbox!",
        );
        return browser;
    }
}

export function resolveProjectPlaywright(projectDir: string): string {
    const require = createRequire(join(projectDir, "package.json"));
    try {
        return require.resolve("playwright");
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "MODULE_NOT_FOUND") throw error;
        const range = JSON.parse(
            readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
        ).peerDependencies.playwright;
        installDev(projectDir, `playwright@${range}`);
        return require.resolve("playwright");
    }
}

export async function projectChromium(projectDir: string, verifyServer = false): Promise<Browser> {
    const require = createRequire(join(projectDir, "package.json"));
    const moduleURL = pathToFileURL(resolveProjectPlaywright(projectDir));
    const loaded = await import(moduleURL.href);
    const { chromium } = (loaded.default ?? loaded) as typeof import("playwright");
    try {
        return await launchPublishChromium(chromium, verifyServer);
    } catch (error) {
        if (!(error instanceof Error) || !error.message.includes("Executable doesn't exist"))
            throw error;
        publishProgress("Installing the project's Playwright Chromium.");
        const { spawnSync } = await import("node:child_process");
        const manifestPath = require.resolve("playwright/package.json");
        const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
        const cli = join(dirname(manifestPath), manifest.bin.playwright);
        const result = spawnSync("node", [cli, "install", "chromium"], {
            cwd: projectDir,
            stdio: ["ignore", 2, 2],
        });
        if (result.status !== 0) throw new Error("Playwright Chromium installation failed.");
        return launchPublishChromium(chromium, verifyServer);
    }
}

/** How a host serves a build's files. */
export interface PublishHost {
    /** The headers a file answers with. */
    headers(path: string): Record<string, string>;
    /** The bytes the host serves for a built file, when it changes them. */
    bytes?(path: string, built: Buffer): Buffer;
    /** Whether a console error at `url` is the host's own, not the game's. */
    foreign?(url: string): boolean;
}

/**
 * Load anonymously, checking build bytes and actual startup. `url` is the build's root or its `.html`
 * document, beside which its files are served, and its query is sent with each file; `host` says how the host serves each file. startupTimeout bounds observation
 * waiting after navigation, in milliseconds. availabilityTimeout bounds retries of asset 404/5xx
 * answers and stale bytes across all assets from the first check, in milliseconds (default 120 seconds);
 * each asset's first request is bounded as a page load is.
 * serverEndpoint requires this same page to connect there and announce advancing applied server state.
 */
export async function verifyPublishedGame(
    browser: Browser,
    url: string,
    directory: string,
    assets: PublishAsset[],
    startupTimeout: number = TIMEOUT.startup,
    host: PublishHost = { headers: () => CROSS_ORIGIN_ISOLATION },
    availabilityTimeout = 120_000,
    serverEndpoint?: string,
): Promise<PublishVerification> {
    const { search, pathname } = new URL(url);
    const base = pathname.endsWith(".html")
        ? new URL(".", url).href
        : `${url.replace(/[?#].*$/, "").replace(/\/$/, "")}/`;
    // A query on the document is kept on every request, so none reads an answer cached before this build.
    const at = (path: string) => `${new URL(path, base).href}${search}`;
    const context = await browser.newContext();
    const firstCheck = performance.now();
    const deadline = firstCheck + availabilityTimeout;
    let retried = false;
    try {
        for (const asset of assets) {
            if (asset.path === "_headers" || asset.path === "_redirects") continue;
            let delay = 250;
            let response: APIResponse;
            let lastAnswer: string | undefined;
            for (;;) {
                try {
                    response = await context.request.get(
                        at(asset.path.split("/").map(encodeURIComponent).join("/")),
                        // Only retries spend the availability wait; a first request gets a load's bound.
                        {
                            timeout: lastAnswer
                                ? Math.max(1, deadline - performance.now())
                                : TIMEOUT.startup,
                        },
                    );
                } catch (error) {
                    if (lastAnswer && performance.now() >= deadline)
                        throw new Error(
                            `Availability wait of ${availabilityTimeout} ms exhausted; ${lastAnswer}`,
                            { cause: error },
                        );
                    throw error;
                }
                const status = response.status();
                let answer = `${asset.path}: expected 200, got ${status}`;
                if (status === 200) {
                    const type = response.headers()["content-type"]?.split(";")[0];
                    const expected =
                        (
                            {
                                html: "text/html",
                                js: "application/javascript",
                                css: "text/css",
                            } as Record<string, string>
                        )[asset.path.split(".").pop()!] ?? contentType(asset.path);
                    if (
                        !type ||
                        (expected &&
                            type !== expected &&
                            !(expected === "application/javascript" && type === "text/javascript"))
                    )
                        throw new Error(
                            `${asset.path}: expected ${expected ?? "a content type"}, got ${type}`,
                        );
                    for (const [name, value] of Object.entries(host.headers(asset.path))) {
                        if (response.headers()[name.toLowerCase()] !== value)
                            throw new Error(`${asset.path}: missing ${name}: ${value}`);
                    }
                    const built = readFileSync(join(directory, asset.path));
                    const matches = (await response.body()).equals(
                        host.bytes?.(asset.path, built) ?? built,
                    );
                    await response.dispose();
                    if (matches) break;
                    answer = `${asset.path}: the deployed bytes still differ from this build`;
                } else {
                    await response.body();
                    await response.dispose();
                    if (status !== 404 && (status < 500 || status > 599)) throw new Error(answer);
                }
                lastAnswer = answer;
                const remaining = deadline - performance.now();
                if (remaining <= 0)
                    throw new Error(
                        `Availability wait of ${availabilityTimeout} ms exhausted; ${answer}`,
                    );
                retried = true;
                await new Promise((resolve) => setTimeout(resolve, Math.min(delay, remaining)));
                if (performance.now() >= deadline)
                    throw new Error(
                        `Availability wait of ${availabilityTimeout} ms exhausted; ${answer}`,
                    );
                delay = Math.min(delay * 2, 5000);
            }
        }
        if (retried)
            publishProgress(
                `Assets available after ${Math.round(performance.now() - firstCheck)} ms.`,
            );
        const missing = await context.request.get(at(`__shallot_missing_${crypto.randomUUID()}`), {
            maxRedirects: 0,
        });
        const missingStatus = missing.status();
        await missing.dispose();
        if (missingStatus !== 404)
            throw new Error(`A missing path answered ${missingStatus}, not 404.`);
        const page = await context.newPage();
        const verifyServer = serverEndpoint
            ? await observePublishedServer(page, serverEndpoint)
            : undefined;
        const errors: string[] = [];
        const failure = new Promise<never>((_, reject) => {
            const fail = (cause: string) => {
                errors.push(cause);
                reject(new Error(`Published page verification failed:\n${errors.join("\n")}`));
            };
            page.on("pageerror", (error) => fail(error.message));
            page.on("console", (message) => {
                const { url, lineNumber, columnNumber } = message.location();
                if (message.type() === "error" && !host.foreign?.(url))
                    fail(
                        url
                            ? `${url}:${lineNumber + 1}:${columnNumber + 1}: ${message.text()}`
                            : message.text(),
                    );
            });
            page.on("crash", () => fail("The published page crashed."));
            page.on("close", () => fail("The published page closed during verification."));
            browser.on("disconnected", disconnected);
            function disconnected() {
                fail("Chromium disconnected during verification.");
            }
            context.on("close", () => browser.off("disconnected", disconnected));
        });
        return await Promise.race([
            failure,
            (async () => {
                await observeStartup(page);
                await page.goto(url, { waitUntil: "load", timeout: TIMEOUT.startup });
                if (!(await page.evaluate(() => crossOriginIsolated)))
                    throw new Error("The published page is not cross-origin isolated.");
                await page
                    .waitForFunction(
                        () => {
                            const status = (
                                window as unknown as { __shallotPublish: BrowserStartup }
                            ).__shallotPublish;
                            return (
                                status.artifact === "single" ||
                                (status.started && status.adapter !== undefined)
                            );
                        },
                        undefined,
                        { timeout: startupTimeout },
                    )
                    .catch(async (error: unknown) => {
                        if (error instanceof Error && error.name === "TimeoutError") {
                            const started = await page.evaluate(
                                () =>
                                    (window as unknown as { __shallotPublish: BrowserStartup })
                                        .__shallotPublish.started,
                            );
                            if (!started)
                                throw new Error(
                                    `The published page sent no shallot:startup app event within ${startupTimeout} ms. Publish verification needs the page to start runApp without interaction.`,
                                    { cause: error },
                                );
                        }
                        throw error;
                    });
                const status = await page.evaluate(
                    () =>
                        (window as unknown as { __shallotPublish: BrowserStartup })
                            .__shallotPublish,
                );
                if (status.threads !== undefined && status.artifact !== "shared")
                    throw new Error(
                        `Physics initialized the ${status.artifact ?? "unknown"} artifact, not shared.`,
                    );
                if (errors.length) throw new Error(`Published page errors:\n${errors.join("\n")}`);
                await verifyServer?.();
                return {
                    hardware: !!status.hardware,
                    adapter: status.adapter!,
                    threads: status.threads,
                    artifact: status.artifact,
                    physicsStepped: status.physicsStepped,
                };
            })(),
        ]);
    } finally {
        await context.close();
    }
}

async function fetchRoomListing(url: URL, signal: AbortSignal): Promise<Response> {
    // Observed Fly cold starts took up to 10.3 seconds. Bound classification separately.
    const startup = AbortSignal.any([signal, AbortSignal.timeout(20_000)]);
    let failure: unknown;
    try {
        for (let delay = 250; ; delay = Math.min(delay * 2, 2000)) {
            let response: Response | undefined;
            try {
                response = await fetch(url, { signal: startup, redirect: "manual" });
            } catch (error) {
                failure = error;
            }
            if (response) {
                if (response.status < 500 || response.status > 599) return response;
                failure = new Error(`GET /rooms returned ${response.status}.`);
                await response.body?.cancel();
            }
            await sleep(delay, undefined, { signal: startup });
        }
    } catch (cause) {
        throw new Error("GET /rooms did not become available within the startup budget.", {
            cause: failure ?? cause,
        });
    }
}

async function reservePublishedRoom(url: URL, signal: AbortSignal, deadline: number) {
    for (;;) {
        const response = await fetch(url, { method: "POST", signal, redirect: "error" });
        if (response.status === 201) return response.json();
        await response.body?.cancel();
        if (response.status !== 503)
            throw new Error(`POST /rooms returned ${response.status}, expected 201.`);
        const retryAfter = response.headers.get("Retry-After") ?? "";
        const wait = /^\d+$/.test(retryAfter)
            ? Number(retryAfter) * 1000
            : Date.parse(retryAfter) - Date.now();
        const delay = Number.isNaN(wait) ? 1000 : Math.max(250, wait);
        if (delay >= deadline - performance.now())
            throw new Error(
                "POST /rooms returned 503; Retry-After exceeds the verification budget.",
            );
        await sleep(delay, undefined, { signal });
    }
}

/** Verify a rooms host without credentials within timeout milliseconds. False leaves single-room verification to the page. */
export async function verifyPublishedRooms(
    endpoint: string,
    timeout: number = TIMEOUT.startup,
): Promise<boolean> {
    const socketUrl = new URL(endpoint);
    const roomsUrl = new URL("/rooms", endpoint);
    roomsUrl.protocol = socketUrl.protocol === "wss:" ? "https:" : "http:";
    const deadline = performance.now() + timeout;
    const signal = AbortSignal.timeout(timeout);
    try {
        const listing = await fetchRoomListing(roomsUrl, signal);
        if (listing.headers.get(ROOMS_LIST_HEADER) !== ROOMS_LIST_VERSION) {
            await listing.body?.cancel();
            return false;
        }
        if (listing.status !== 200) {
            await listing.body?.cancel();
            throw new Error(`GET /rooms returned ${listing.status}, expected 200.`);
        }
        const rooms: unknown = await listing.json();
        if (
            !Array.isArray(rooms) ||
            !rooms.every(
                (room) =>
                    typeof room?.code === "string" &&
                    Number.isInteger(room.clients) &&
                    Number.isInteger(room.maxClients),
            )
        )
            throw new Error("GET /rooms did not return a room listing.");

        // Claim an isolated room, rather than spend a seat in an existing public room.
        const { code, creator } = await reservePublishedRoom(roomsUrl, signal, deadline);
        if (typeof code !== "string" || !code || typeof creator !== "string" || !creator)
            throw new Error("POST /rooms did not return a room reservation.");
        socketUrl.search = new URLSearchParams({
            intent: "join",
            code,
            creator,
            announceRoom: "1",
        }).toString();
        signal.throwIfAborted();
        const socket = new WebSocket(socketUrl);
        try {
            await new Promise<void>((resolve, reject) => {
                signal.addEventListener("abort", () => reject(signal.reason), { once: true });
                socket.addEventListener("message", ({ data }) => {
                    if (data === `room:${code}`) resolve();
                });
                socket.addEventListener("close", ({ code }) => {
                    if (code === ROOM_AUTH_REFUSED_CLOSE_CODE) resolve();
                    else
                        reject(new Error(`Room WebSocket closed before admission (code ${code}).`));
                });
                socket.addEventListener("error", () => reject(new Error("Room WebSocket failed.")));
            });
        } finally {
            socket.close(LEAVE_CLOSE_CODE);
        }
        return true;
    } catch (cause) {
        throw new Error(
            `Server verification failed at ${roomsUrl.href}: ${cause instanceof Error ? cause.message : String(cause)}`,
            { cause },
        );
    }
}

/** Observe before navigation; verify applied server-state events and endpoint frames after startup. */
export async function observePublishedServer(
    page: Page,
    endpoint: string,
    timeout: number = TIMEOUT.startup,
): Promise<() => Promise<void>> {
    const expected = new URL(endpoint).href;
    await page.addInitScript((endpoint) => {
        const state: { first?: number; tick?: number } = {};
        Object.defineProperty(window, "__shallotServerState", { value: state });
        window.addEventListener("shallot:startup", (event) => {
            const detail = (event as CustomEvent).detail;
            if (
                detail.kind === "server-state" &&
                detail.endpoint === endpoint &&
                Number.isInteger(detail.tick) &&
                detail.tick >= 0
            ) {
                state.first ??= detail.tick;
                state.tick = detail.tick;
            }
        });
    }, expected);
    let receivedFrame = false;
    page.on("websocket", (socket) => {
        if (new URL(socket.url()).href === expected)
            socket.on("framereceived", () => {
                receivedFrame = true;
            });
    });
    return async () => {
        try {
            await page.waitForFunction(
                () => {
                    const state = (
                        window as unknown as {
                            __shallotServerState: { first?: number; tick?: number };
                        }
                    ).__shallotServerState;
                    return (
                        state.first !== undefined &&
                        state.tick !== undefined &&
                        state.tick >= state.first + 60
                    );
                },
                undefined,
                { timeout },
            );
        } catch (error) {
            throw new Error(
                receivedFrame
                    ? `Published page did not advance confirmed server state by 60 ticks from ${endpoint}`
                    : `Published page received no frames from ${endpoint}`,
                { cause: error },
            );
        }
        if (!receivedFrame) throw new Error(`Published page received no frames from ${endpoint}`);
    };
}

interface BrowserStartup {
    started: boolean;
    physicsStepped?: boolean;
    threads?: number;
    artifact?: "shared" | "single";
    adapter?: string;
    hardware?: boolean;
}

async function observeStartup(page: Page): Promise<void> {
    await page.addInitScript(() => {
        const status: BrowserStartup = { started: false };
        Object.defineProperty(window, "__shallotPublish", { value: status });
        window.addEventListener("shallot:startup", (event) => {
            const detail = (event as CustomEvent).detail;
            if (detail.kind === "app") {
                status.started = true;
                status.adapter = detail.adapter?.identity ?? "No GPU required";
                status.hardware = detail.adapter?.class === "real";
            }
            if (detail.kind === "physics") {
                status.threads = detail.threads;
                status.artifact = detail.artifact;
            }
            if (detail.kind === "physics-step") {
                status.physicsStepped = true;
                status.threads = detail.threads;
            }
        });
    });
}
