import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import type { APIResponse, Browser, BrowserType, Page } from "playwright";
import { CROSS_ORIGIN_ISOLATION, contentType } from "../project";
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
): Promise<Browser> {
    try {
        return await chromium.launch(PUBLISH_LAUNCH_OPTIONS);
    } catch (error) {
        if (!(error instanceof Error) || !error.message.includes("No usable sandbox!")) throw error;
        const browser = await chromium.launch({
            ...PUBLISH_LAUNCH_OPTIONS,
            chromiumSandbox: false,
        });
        publishProgress(
            "Verification ran without Chromium's sandbox: Chromium reported No usable sandbox!",
        );
        return browser;
    }
}

export async function projectChromium(projectDir: string): Promise<Browser> {
    const require = createRequire(join(projectDir, "package.json"));
    let modulePath: string;
    try {
        modulePath = require.resolve("playwright");
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "MODULE_NOT_FOUND") throw error;
        const range = JSON.parse(
            readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
        ).peerDependencies.playwright;
        throw new Error(
            `Verification needs this project's Playwright. Run bun add --dev playwright@${range}, then rerun publish.`,
        );
    }
    const { chromium } = (await import(modulePath)) as typeof import("playwright");
    try {
        return await launchPublishChromium(chromium);
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
        return launchPublishChromium(chromium);
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
 * answers and stale bytes across all assets from the first check, in milliseconds (default 120 seconds).
 */
export async function verifyPublishedGame(
    browser: Browser,
    url: string,
    directory: string,
    assets: PublishAsset[],
    startupTimeout: number = TIMEOUT.startup,
    host: PublishHost = { headers: () => CROSS_ORIGIN_ISOLATION },
    availabilityTimeout = 120_000,
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
                        { timeout: Math.max(1, deadline - performance.now()) },
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
                await page.waitForFunction(
                    () => {
                        const status = (window as unknown as { __shallotPublish: BrowserStartup })
                            .__shallotPublish;
                        return (
                            status.artifact === "single" ||
                            (status.started && status.adapter !== undefined)
                        );
                    },
                    undefined,
                    { timeout: startupTimeout },
                );
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
