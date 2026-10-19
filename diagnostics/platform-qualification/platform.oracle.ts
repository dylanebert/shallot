import { mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { firefox, webkit } from "playwright";
import { bunChild } from "../../scripts/bun-child";

const [browserName, directory, control] = process.argv.slice(2);
if (
    (browserName !== "webkit" && browserName !== "firefox") ||
    !directory ||
    (control && control !== "--control")
)
    throw new Error(
        "Usage: bun diagnostics/platform-qualification/platform.oracle.ts <webkit|firefox> <output-directory> [--control]. Firefox is a display run: only the person starts it, with SHALLOT_DISPLAY_SEAT set.",
    );
if (browserName === "firefox" && !process.env.SHALLOT_DISPLAY_SEAT)
    throw new Error("Firefox requires SHALLOT_DISPLAY_SEAT identifying the physical monitor.");
const output = resolve(directory);
const root = resolve(import.meta.dir, "../..");
if (output === root || output.startsWith(`${root}/`))
    throw new Error("Frame and report output must be outside the checkout.");
mkdirSync(output, { recursive: true });
const examples = control
    ? ["control", "control-dom"]
    : readdirSync(resolve(root, "examples"), { withFileTypes: true })
          .filter((entry) => entry.isDirectory())
          .map((entry) => entry.name)
          .sort();
const browser = await (browserName === "webkit" ? webkit : firefox).launch({
    headless: browserName !== "firefox",
});
let failed = false;
try {
    for (const example of examples) {
        const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
        page.setDefaultTimeout(TIMEOUT.browser);
        const errors: string[] = [];
        const warnings: string[] = [];
        const notices: string[] = [];
        const report = {
            example,
            verdict: "FAIL",
            adapter: [] as unknown[],
            errors,
            warnings,
            notices,
            frame: resolve(output, `${example}.png`),
        };
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "error") errors.push(message.text());
            else if (message.type() === "warning") warnings.push(message.text());
            else notices.push(message.text());
        });
        await page.addInitScript(() => {
            const observations: unknown[] = [];
            Object.assign(window, { platformAdapters: observations });
            if (!navigator.gpu) return;
            const request = navigator.gpu.requestAdapter.bind(navigator.gpu);
            navigator.gpu.requestAdapter = async (options) => {
                const adapter = await request(options);
                observations.push(
                    adapter
                        ? {
                              vendor: adapter.info.vendor,
                              architecture: adapter.info.architecture,
                              device: adapter.info.device,
                              description: adapter.info.description,
                              isFallbackAdapter: adapter.info.isFallbackAdapter,
                          }
                        : null,
                );
                return adapter;
            };
        });
        let preview: ReturnType<typeof Bun.spawn> | undefined;
        let captured = false;
        try {
            if (example.startsWith("control")) {
                const overlay =
                    example === "control-dom"
                        ? "<div style='position:absolute;inset:0;color:white;font:48px sans-serif'>Loading scene…</div>"
                        : "";
                await page.goto(
                    `data:text/html,${encodeURIComponent(`<!doctype html><html><head><meta charset="utf-8"></head><body style='margin:0;background:black'><canvas width='1280' height='720'></canvas>${overlay}</body></html>`)}`,
                );
            } else {
                const cwd = resolve(root, "examples", example);
                const vite = resolve(root, "node_modules/vite/bin/vite.js");
                const build = Bun.spawn({
                    ...bunChild([vite, "build"]),
                    cwd,
                    stdout: "inherit",
                    stderr: "inherit",
                    timeout: TIMEOUT.startup,
                });
                if ((await build.exited) !== 0) throw new Error(`${example}: vite build failed`);
                const reservation = createServer();
                await new Promise<void>((resolve, reject) => {
                    reservation.once("error", reject);
                    reservation.listen(0, "127.0.0.1", resolve);
                });
                const address = reservation.address();
                if (!address || typeof address === "string") throw new Error("No preview port");
                const port = address.port;
                await new Promise<void>((resolve, reject) =>
                    reservation.close((error) => (error ? reject(error) : resolve())),
                );
                const url = `http://127.0.0.1:${port}`;
                preview = Bun.spawn({
                    ...bunChild([
                        vite,
                        "preview",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        String(port),
                        "--strictPort",
                    ]),
                    cwd,
                    stdout: "inherit",
                    stderr: "inherit",
                });
                const started = performance.now();
                while (true) {
                    if (preview.exitCode !== null)
                        throw new Error(`${example}: vite preview exited ${preview.exitCode}`);
                    if (
                        await fetch(url)
                            .then((response) => response.ok)
                            .catch(() => false)
                    )
                        break;
                    if (performance.now() - started >= TIMEOUT.startup)
                        throw new Error(`${example}: vite preview startup timed out`);
                    await Bun.sleep(100);
                }
                await page.goto(url);
            }
            await page.locator("body > [data-shallot-loading]").waitFor({ state: "detached" });
            // This example owns Loading.complete's intro/dwell hold and its card removal;
            // startup.e2e.ts waits for the same card before observing the scene.
            if (example === "loading-screen")
                await page.locator("[data-title-card]").waitFor({ state: "detached" });
            await page.waitForFunction(() => {
                const canvas = document.querySelector("canvas");
                if (
                    !canvas ||
                    !canvas.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })
                )
                    return false;
                for (let element: Element | null = canvas; element; element = element.parentElement)
                    if (getComputedStyle(element).opacity !== "1") return false;
                return true;
            });
            // Element screenshots include overlapping DOM. Hide that paint, including
            // CSS decoration, and composite transparent pixels onto uniform black
            // without replacing or reading back the presenting canvas.
            const screenshot = await page.locator("canvas").screenshot({
                path: report.frame,
                style: `* { visibility: hidden !important; background: none !important;
                    border-color: transparent !important; box-shadow: none !important;
                    outline: none !important; }
                    *::before, *::after { visibility: hidden !important; }
                    html, canvas { background: black !important; }
                    canvas { visibility: visible !important; }`,
            });
            captured = true;
            const nonuniform = await page.evaluate(async (base64) => {
                const image = new Image();
                image.src = `data:image/png;base64,${base64}`;
                await image.decode();
                const canvas = document.createElement("canvas");
                canvas.width = image.width;
                canvas.height = image.height;
                const context = canvas.getContext("2d")!;
                context.drawImage(image, 0, 0);
                const { data } = context.getImageData(0, 0, image.width, image.height);
                return data.some((value, i) => value !== data[i % 4]);
            }, screenshot.toString("base64"));
            if (!nonuniform) throw new Error("canvas screenshot is uniform: no drawn frame");
            if (errors.length) throw new Error(`page and console errors: ${errors[0]}`);
            report.verdict = "PASS";
        } catch (error) {
            failed = true;
            errors.push(error instanceof Error ? error.message : String(error));
            if (!captured)
                await page
                    .locator("canvas")
                    .screenshot({ path: report.frame })
                    .catch(() => {});
        } finally {
            report.adapter = await page
                .evaluate(
                    () =>
                        (window as unknown as { platformAdapters?: unknown[] }).platformAdapters ??
                        [],
                )
                .catch(() => []);
            writeFileSync(
                resolve(output, `${example}.json`),
                `${JSON.stringify(report, null, 2)}\n`,
            );
            console.log(JSON.stringify(report));
            await page.close();
            if (preview) {
                preview.kill();
                await preview.exited;
            }
        }
    }
} finally {
    await browser.close();
}
process.exitCode = failed ? 1 : 0;
