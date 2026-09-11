import { resolve } from "node:path";
import { REAL_GPU_LAUNCH } from "./index";
import type { Verdict } from "./runtime";

interface BrowserVerdict extends Verdict {
    runtime: string;
    hardware: string;
}

/** an error that retains the browser host labels for the verdict line. */
export class BrowserDriverError extends Error {
    runtime: string;
    hardware: string;

    constructor(message: string, runtime: string, hardware = "none") {
        super(message);
        this.name = "BrowserDriverError";
        this.runtime = runtime;
        this.hardware = hardware;
    }
}

async function waitForServer(url: string, process: ReturnType<typeof Bun.spawn>): Promise<void> {
    const deadline = performance.now() + 15_000;
    while (performance.now() < deadline) {
        if (process.exitCode !== null)
            throw new Error(`shallot dev exited with ${process.exitCode}`);
        try {
            const response = await fetch(url, { signal: AbortSignal.timeout(500) });
            if (response.ok) return;
        } catch {
            // Vite is still starting or its first response is not ready.
        }
        await Bun.sleep(50);
    }
    throw new Error(`timed out waiting for shallot dev at ${url}`);
}

async function adapterLabel(page: import("playwright").Page): Promise<string> {
    return page.evaluate(async () => {
        const gpu = navigator.gpu;
        if (!gpu) return "none";
        const adapter = await gpu.requestAdapter();
        if (!adapter) return "none";
        const info = (adapter as GPUAdapter & { info?: Record<string, string> }).info;
        const label = [info?.description, info?.device, info?.vendor, info?.architecture]
            .filter((part) => typeof part === "string" && part.length > 0)
            .join(" ");
        return label || "gpu";
    });
}

/**
 * Boot a manifest recipe through the normal dev host and return the page's in-engine Verdict.
 * The page owns the stepped-clock assertion; this process only waits for readiness and transports
 * the resulting JSON across the browser boundary.
 */
export async function runBrowserCheck(projectDir: string): Promise<BrowserVerdict> {
    const { chromium } = await import("playwright");
    const browser = await chromium.launch({ headless: true, ...REAL_GPU_LAUNCH });
    const runtime = `bun ${Bun.version} + chromium ${browser.version()}`;
    const port = 4000 + Math.floor(Math.random() * 1000);
    const url = `http://localhost:${port}/`;
    const server = Bun.spawn(
        [
            process.execPath,
            resolve(import.meta.dir, "../../bin/shallot.ts"),
            "dev",
            resolve(projectDir),
            "--port",
            String(port),
            "--strict-port",
            "--no-open",
        ],
        { cwd: resolve(import.meta.dir, "../.."), stdout: "ignore", stderr: "ignore" },
    );
    const page = await browser.newPage();
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "error") pageErrors.push(message.text());
    });
    try {
        await waitForServer(url, server);
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 10_000 });
        await page.waitForFunction(() => window.__harness?.ready === true, undefined, {
            timeout: 10_000,
        });
        const hardware = await adapterLabel(page);
        const verdict = (await page.evaluate(async () => {
            const harness = window.__harness;
            if (!harness?.run) throw new Error("page did not install window.__harness.run");
            return harness.run({ tier: "browser" });
        })) as Verdict;
        return { ...verdict, runtime, hardware };
    } catch (error) {
        const detail = pageErrors.length > 0 ? ` (${pageErrors.join(" | ")})` : "";
        throw new BrowserDriverError(
            `${error instanceof Error ? error.message : String(error)}${detail}`,
            runtime,
            await adapterLabel(page).catch(() => "none"),
        );
    } finally {
        await page.close();
        await browser.close();
        server.kill();
        await server.exited;
    }
}
