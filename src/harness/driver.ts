import { version as engineVersion } from "../../package.json" with { type: "json" };
import { CAPTURE_CONTRACT, type CaptureIdentity, captureIdentityLabel } from "./capture";
import { launchOptions, launchPlan } from "./launch";
import type { Verdict } from "./runtime";
import { type AdapterFacts, classifyAdapter, resolveSeat } from "./seat";
import type { Reproduction, VerdictDiagnostics } from "./verdict";

/** the verdict a browser row returns: the page's own verdict plus the reproduction record behind it. */
export interface BrowserVerdict extends Verdict {
    runtime: string;
    hardware: string;
    reproduction: Reproduction;
}

/** how much of each diagnostic survives into a verdict. Bounded, because a verdict line is read. */
const MAX_PAGE_ERRORS = 20;
const MAX_ERROR_CHARS = 500;
const MAX_SERVER_LOG_CHARS = 4_000;
/** the cap on a retained failure artifact; a capture at the declared contract is far below it. */
const MAX_ARTIFACT_BYTES = 4 * 1024 * 1024;
const PNG_DATA_URL = "data:image/png;base64,";

/** an error that retains the reproduction record and bounded diagnostics for the verdict line. */
export class BrowserDriverError extends Error {
    runtime: string;
    hardware: string;
    reproduction: Reproduction;
    diagnostics: VerdictDiagnostics;

    constructor(message: string, reproduction: Reproduction, diagnostics: VerdictDiagnostics = {}) {
        super(message);
        this.name = "BrowserDriverError";
        this.runtime = reproduction.runtime;
        this.hardware = reproduction.adapter;
        this.reproduction = reproduction;
        this.diagnostics = diagnostics;
    }
}

function bounded(values: readonly string[]): string[] {
    return values.slice(0, MAX_PAGE_ERRORS).map((value) => value.slice(0, MAX_ERROR_CHARS));
}

function tail(value: string, limit = MAX_SERVER_LOG_CHARS): string {
    return value.length <= limit ? value : `…${value.slice(value.length - limit)}`;
}

async function readStream(stream: unknown): Promise<string> {
    if (stream === null || stream === undefined || typeof stream === "number") return "";
    try {
        return (await new Response(stream as ReadableStream<Uint8Array>).text()).trim();
    } catch {
        return "";
    }
}

async function waitForServer(url: string, process: ReturnType<typeof Bun.spawn>): Promise<void> {
    const deadline = performance.now() + 15_000;
    while (performance.now() < deadline) {
        if (process.exitCode !== null)
            throw new Error(`serve command exited with ${process.exitCode}`);
        try {
            const response = await fetch(url, { signal: AbortSignal.timeout(500) });
            if (response.ok) return;
        } catch {
            // The serve command is still starting or its first response is not ready.
        }
        await Bun.sleep(50);
    }
    throw new Error(`timed out waiting for serve command at ${url}`);
}

/**
 * Read the adapter facts the seat policy classifies. Each `info` field is read by name because the info
 * object's fields are prototype accessors and never survive a spread — reading them as a bag is how a
 * masked identity used to arrive as an empty object and get labelled a device anyway.
 */
async function adapterFacts(page: import("playwright").Page): Promise<AdapterFacts> {
    return page.evaluate(async () => {
        const gpu = navigator.gpu;
        if (!gpu) return { present: false };
        const adapter = await gpu.requestAdapter();
        if (!adapter) return { present: false };
        const info = adapter.info as (GPUAdapterInfo & { isFallbackAdapter?: boolean }) | undefined;
        return {
            present: true,
            info: {
                vendor: info?.vendor,
                architecture: info?.architecture,
                device: info?.device,
                description: info?.description,
                isFallbackAdapter: info?.isFallbackAdapter,
            },
        };
    });
}

/** The capture surface's own geometry, or null when the page presents no canvas. */
async function captureGeometry(
    page: import("playwright").Page,
): Promise<{ width: number; height: number; deviceScale: number } | null> {
    return page.evaluate(() => {
        const canvas = document.querySelector("canvas");
        if (!canvas) return null;
        return { width: canvas.width, height: canvas.height, deviceScale: window.devicePixelRatio };
    });
}

export type BrowserServeCommand = (port: number) => string[];

/**
 * Boot a project through its serve command and return the page's in-engine {@link Verdict} with its
 * reproduction record. The driver supplies an unused port; the command owns its host-specific arguments.
 *
 * The page owns the stepped-clock assertion; this process fixes the seat and the capture geometry, waits
 * for readiness, and transports the resulting JSON across the browser boundary.
 *
 * The launch is always headless: the mode is policy, not a caller's choice. A host whose headless Chromium
 * reaches only a fallback adapter refuses, because a software adapter is not the `chromium` seat, and
 * opening a window to pass would report one seat's result as another's.
 *
 * @example const verdict = await runBrowserCheck((port) => ["bun", "serve.ts", "--port", String(port)]);
 */
export async function runBrowserCheck(
    serveCommand: BrowserServeCommand,
    opts: { contract?: CaptureIdentity } = {},
): Promise<BrowserVerdict> {
    if (Object.hasOwn(opts, "headless")) {
        throw new Error(
            "runBrowserCheck refused: `headless` is not a caller option; the harness launches headless and a host that cannot reach a real adapter that way refuses",
        );
    }
    const contract = opts.contract ?? CAPTURE_CONTRACT;
    const host = process.platform;
    const plan = launchPlan(host);
    const reproduction: Reproduction = {
        host: `${host}-${process.arch}`,
        launch: "headless",
        runtime: `bun ${Bun.version}`,
        chromium: "none",
        adapter: "none",
        adapterClass: "absent",
        viewport: `${contract.width}x${contract.height}@${contract.deviceScale}`,
        capture: captureIdentityLabel(contract),
        engine: engineVersion,
    };
    if ("refused" in plan) throw new BrowserDriverError(plan.refused, reproduction);

    const { chromium } = await import("playwright");
    const browser = await chromium.launch(launchOptions(plan));
    reproduction.chromium = browser.version();
    reproduction.runtime = `bun ${Bun.version} + chromium ${browser.version()}`;
    reproduction.adapterEvidence = plan.adapterEvidence;
    const port = 4000 + Math.floor(Math.random() * 1000);
    const url = `http://localhost:${port}/`;
    // The serve command's own output is evidence: a server that fails to boot, or warns about the route
    // under test, used to be discarded here and read as an unexplained page timeout.
    const server = Bun.spawn(serveCommand(port), { stdout: "pipe", stderr: "pipe" });
    const context = await browser.newContext({
        viewport: { width: contract.width, height: contract.height },
        deviceScaleFactor: contract.deviceScale,
    });
    const page = await context.newPage();
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
        const facts = await adapterFacts(page);
        const adapter = classifyAdapter(facts);
        reproduction.adapter = adapter.identity;
        reproduction.adapterClass = adapter.class;
        const geometry = await captureGeometry(page);
        const observed: CaptureIdentity | null =
            geometry === null ? null : { ...contract, ...geometry };
        if (observed !== null) reproduction.capture = captureIdentityLabel(observed);
        const seat = resolveSeat(
            "chromium",
            {
                browser: {
                    launch: plan,
                    adapter: facts,
                    ...(observed === null ? {} : { capture: { identity: observed } }),
                },
            },
            observed === null ? undefined : contract,
        );
        if (!seat.ok) throw new Error(seat.reason);
        const verdict = (await page.evaluate(async () => {
            const harness = window.__harness;
            if (!harness?.run) throw new Error("page did not install window.__harness.run");
            return harness.run({ size: "integration", requires: ["chromium"] });
        })) as Verdict;
        if (typeof verdict.tick === "number") reproduction.tick = verdict.tick;
        if (verdict.ok === false) {
            throw new BrowserDriverError(
                "the page returned a failing verdict",
                reproduction,
                await diagnose(page, contract, pageErrors, server, verdict),
            );
        }
        return {
            ...verdict,
            runtime: reproduction.runtime,
            hardware: adapter.identity,
            reproduction,
        };
    } catch (error) {
        if (error instanceof BrowserDriverError) throw error;
        throw new BrowserDriverError(
            error instanceof Error ? error.message : String(error),
            reproduction,
            await diagnose(page, contract, pageErrors, server),
        );
    } finally {
        await page.close();
        await context.close();
        await browser.close();
        server.kill();
        await server.exited;
    }
}

/**
 * Collect the bounded failure diagnostics: page errors, the serve command's output, the page's own GPU
 * errors, the failing sub-checks with their structured data, and one retained actual capture.
 */
async function diagnose(
    page: import("playwright").Page,
    contract: CaptureIdentity,
    pageErrors: readonly string[],
    server: ReturnType<typeof Bun.spawn>,
    verdict?: Verdict,
): Promise<VerdictDiagnostics> {
    const diagnostics: VerdictDiagnostics = {};
    if (pageErrors.length > 0) diagnostics.pageErrors = bounded(pageErrors);
    const gpu = await page
        .evaluate(() => (window as { __gpuErrors?: unknown }).__gpuErrors)
        .catch(() => undefined);
    if (Array.isArray(gpu) && gpu.length > 0) diagnostics.gpuErrors = bounded(gpu.map(String));
    const failed = (verdict?.checks ?? []).filter((entry) => !entry.ok);
    if (failed.length > 0) {
        diagnostics.checks = failed.map((entry) => ({
            name: entry.name,
            ...(entry.detail === undefined
                ? {}
                : { detail: entry.detail.slice(0, MAX_ERROR_CHARS) }),
            ...(entry.data === undefined ? {} : { data: entry.data }),
        }));
    }
    const artifact = await retainArtifact(page, contract).catch(() => undefined);
    if (artifact !== undefined) diagnostics.artifacts = [artifact];
    // The serve command's streams read only once the process has ended, so this comes last.
    server.kill();
    await server.exited.catch(() => undefined);
    const serverLog = [await readStream(server.stdout), await readStream(server.stderr)]
        .filter((part) => part !== "")
        .join("\n");
    if (serverLog !== "") diagnostics.serverLog = tail(serverLog);
    return diagnostics;
}

/** Write the actual frame through the same capture surface, bounded, and return its path. */
async function retainArtifact(
    page: import("playwright").Page,
    contract: CaptureIdentity,
): Promise<string | undefined> {
    const dataUrl = await page.evaluate(() => {
        const canvas = document.querySelector("canvas");
        return canvas === null ? null : canvas.toDataURL("image/png");
    });
    if (typeof dataUrl !== "string" || !dataUrl.startsWith(PNG_DATA_URL)) return undefined;
    const bytes = Buffer.from(dataUrl.slice(PNG_DATA_URL.length), "base64");
    if (bytes.length === 0 || bytes.length > MAX_ARTIFACT_BYTES) return undefined;
    // Imported here, not at module scope: `./index` reaches the page bundle too, and the driver's own
    // node dependencies must not follow it there.
    const { mkdirSync, writeFileSync } = await import("node:fs");
    const { resolve } = await import("node:path");
    const root = resolve(process.env.SHALLOT_PROJECT_ROOT ?? process.cwd(), ".artifacts");
    mkdirSync(root, { recursive: true });
    const path = resolve(
        root,
        `actual-${contract.width}x${contract.height}-${Date.now().toString(36)}.png`,
    );
    writeFileSync(path, bytes);
    return path;
}
