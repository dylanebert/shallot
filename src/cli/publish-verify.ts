import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type { Browser, Page } from "playwright";
import { contentType } from "../project/assets.js";
import { CROSS_ORIGIN_ISOLATION } from "../project/isolation.js";
import { BROWSER_CONFIG, TIMEOUT } from "../project/testing.js";

export interface PublishAsset {
    path: string;
    size: number;
}
export interface PublishVerification {
    hardware: boolean;
    adapter: string;
    threads?: number;
    physicsStepped?: boolean;
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
        return await chromium.launch(BROWSER_CONFIG.use.launchOptions);
    } catch (error) {
        if (!(error instanceof Error) || !error.message.includes("Executable doesn't exist"))
            throw error;
        console.error("Installing the project's Playwright Chromium.");
        const { spawnSync } = await import("node:child_process");
        const manifestPath = require.resolve("playwright/package.json");
        const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
        const cli = join(dirname(manifestPath), manifest.bin.playwright);
        const result = spawnSync("node", [cli, "install", "chromium"], {
            cwd: projectDir,
            stdio: ["ignore", 2, 2],
        });
        if (result.status !== 0) throw new Error("Playwright Chromium installation failed.");
        return chromium.launch(BROWSER_CONFIG.use.launchOptions);
    }
}

/** Load anonymously, checking the build's bytes as well as the page's actual startup observations. */
export async function verifyPublishedGame(
    browser: Browser,
    url: string,
    directory: string,
    assets: PublishAsset[],
): Promise<PublishVerification> {
    const context = await browser.newContext();
    try {
        for (const asset of assets) {
            if (asset.path === "_headers" || asset.path === "_redirects") continue;
            const response = await context.request.get(
                new URL(
                    asset.path.split("/").map(encodeURIComponent).join("/"),
                    `${url.replace(/\/$/, "")}/`,
                ).href,
            );
            if (response.status() !== 200)
                throw new Error(`${asset.path}: expected 200, got ${response.status()}`);
            const type = response.headers()["content-type"]?.split(";")[0];
            const expected =
                (
                    { html: "text/html", js: "application/javascript", css: "text/css" } as Record<
                        string,
                        string
                    >
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
            if (!(await response.body()).equals(readFileSync(join(directory, asset.path))))
                throw new Error(`${asset.path}: the deployed bytes differ from this build`);
            for (const [name, value] of Object.entries(CROSS_ORIGIN_ISOLATION)) {
                if (response.headers()[name.toLowerCase()] !== value)
                    throw new Error(`${asset.path}: missing ${name}: ${value}`);
            }
            await response.dispose();
        }
        const missing = await context.request.get(
            new URL(`__shallot_missing_${crypto.randomUUID()}`, `${url.replace(/\/$/, "")}/`).href,
            { maxRedirects: 0 },
        );
        const missingStatus = missing.status();
        await missing.dispose();
        if (missingStatus !== 404)
            throw new Error(`A missing path answered ${missingStatus}, not 404.`);
        const page = await context.newPage();
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "error") errors.push(message.text());
        });
        await observeStartup(page);
        await page.goto(url, { waitUntil: "load", timeout: TIMEOUT.startup });
        if (!(await page.evaluate(() => crossOriginIsolated)))
            throw new Error("The published page is not cross-origin isolated.");
        await page.waitForFunction(
            () => {
                const status = (window as unknown as { __shallotPublish: BrowserStartup })
                    .__shallotPublish;
                return status.started && status.adapter !== undefined;
            },
            undefined,
            { timeout: TIMEOUT.startup },
        );
        const status = await page.evaluate(
            () => (window as unknown as { __shallotPublish: BrowserStartup }).__shallotPublish,
        );
        if (status.threads !== undefined && !(status.threads > 1))
            throw new Error(`Physics started with ${status.threads} thread, not a worker pool.`);
        if (errors.length) throw new Error(`Published page errors:\n${errors.join("\n")}`);
        return {
            hardware: !!status.hardware,
            adapter: status.adapter!,
            threads: status.threads,
            physicsStepped: status.physicsStepped,
        };
    } finally {
        await context.close();
    }
}

interface BrowserStartup {
    started: boolean;
    physicsStepped?: boolean;
    threads?: number;
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
            if (detail.kind === "physics") status.threads = detail.threads;
            if (detail.kind === "physics-step") {
                status.physicsStepped = true;
                status.threads = detail.threads;
            }
        });
    });
}
