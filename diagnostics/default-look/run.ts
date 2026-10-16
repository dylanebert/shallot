/*
# Default-look scene captures

`scenes.ts` is the fixed scene set. `palette.ts` is its shared material-colour table; adding a palette renders the same scene list under a `test-results/default-look/<palette>/` directory.

Run the capture tool from the repository root:

```sh
bun diagnostics/default-look/run.ts
```

The tool serves the scene page, drives headless Chromium at a 1280×720 viewport and device scale 1, and writes one PNG per scene and palette under the ignored `test-results/` directory. `test-results/default-look/index.html` compares the captures and lists colour values with range statuses. Candidate hard-bound violations fail the report; theme role ranges are reported as a reference, and the unchanged baseline's violations are shown. It requires a hardware WebGPU adapter and refuses software adapters. To verify repeatability on the same machine, save the image and page hashes, rerun the tool, and compare:

```sh
shasum -a 256 test-results/default-look/*/*.png test-results/default-look/index.html > /tmp/default-look.sha256
bun diagnostics/default-look/run.ts
shasum -a 256 test-results/default-look/*/*.png test-results/default-look/index.html | diff -u /tmp/default-look.sha256 -
```
*/

import { mkdir, writeFile } from "node:fs/promises";
import { createServer as createHttpServer } from "node:http";
import { resolve } from "node:path";
import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import { shallot } from "@dylanebert/shallot/vite";
import { chromium, type Page } from "playwright";
import { createServer as createViteServer } from "vite";
import { PALETTES } from "./palette";
import { reportHtml } from "./report";
import { SCENES } from "./scenes";

const root = resolve(import.meta.dir, "../..");
const outputRoot = resolve(root, "test-results/default-look");
const pageHtml = `<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shallot — default look</title>
    <style>
        * { box-sizing: border-box; }
        html, body, canvas { width: 100%; height: 100%; margin: 0; }
        body { overflow: hidden; background: #1c1917; }
        canvas { display: block; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script type="module" src="/diagnostics/default-look/main.ts"></script>
</body>
</html>`;

async function visiblePixelCount(page: Page): Promise<number> {
    const screenshot = await page.locator("#canvas").screenshot();
    return page.evaluate(async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
        const image = await createImageBitmap(new Blob([bytes]));
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Could not inspect the canvas screenshot");
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        image.close();

        let count = 0;
        for (let offset = 0; offset < pixels.length; offset += 4) {
            if (
                Math.abs(pixels[offset]! - 28) > 8 ||
                Math.abs(pixels[offset + 1]! - 25) > 8 ||
                Math.abs(pixels[offset + 2]! - 23) > 8
            )
                count++;
        }
        return count;
    }, screenshot.toString("base64"));
}

async function waitForScene(page: Page, ready: string): Promise<void> {
    await page.waitForFunction((expected) => document.body.dataset.ready === expected, ready, {
        timeout: 30_000,
    });
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
        if ((await visiblePixelCount(page)) > 800) return;
        await new Promise((resolve_) => setTimeout(resolve_, 100));
    }
    throw new Error(`${ready} did not paint geometry beyond its clear color`);
}

const vite = await createViteServer({
    configFile: false,
    root,
    plugins: [shallot()],
    server: { middlewareMode: true, hmr: false },
    appType: "custom",
});
const server = createHttpServer((request, response) => {
    if (new URL(request.url ?? "/", "http://localhost").pathname === "/") {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(pageHtml);
        return;
    }
    vite.middlewares(request, response, (error?: Error | null) => {
        if (error) {
            response.statusCode = 500;
            response.end(error.message);
        } else {
            response.statusCode = 404;
            response.end("Not found");
        }
    });
});

let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
try {
    await new Promise<void>((resolve_, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve_);
    });
    const address = server.address();
    if (!address || typeof address === "string")
        throw new Error("Could not bind diagnostic server");

    browser = await chromium.launch({
        headless: true,
        channel: BROWSER_CONFIG.use.channel,
        ...BROWSER_CONFIG.use.launchOptions,
    });
    const page = await browser.newPage({
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${address.port}/`);
    const adapter = await page.evaluate(async () => {
        if (!navigator.gpu) throw new Error("WebGPU is unavailable in Chromium");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU did not provide an adapter");
        return {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
            description: adapter.info.description,
        };
    });
    const identity = Object.values(adapter).join(" ").trim();
    if (!identity || /swiftshader|llvmpipe|lavapipe|software rasterizer/i.test(identity))
        throw new Error(
            `Captures require a hardware WebGPU adapter; received ${JSON.stringify(adapter)}`,
        );
    console.log(`Default-look WebGPU adapter: ${JSON.stringify(adapter)}`);

    for (const [paletteName] of Object.entries(PALETTES)) {
        for (const scene of SCENES) {
            const ready = `${paletteName}/${scene.id}`;
            await page.goto(
                `http://127.0.0.1:${address.port}/?scene=${encodeURIComponent(scene.id)}&palette=${encodeURIComponent(paletteName)}`,
            );
            await waitForScene(page, ready);
            if (errors.length)
                throw new Error(`${ready} raised a page error: ${errors.join("; ")}`);

            const directory = resolve(outputRoot, paletteName);
            await mkdir(directory, { recursive: true });
            const path = resolve(directory, `${scene.id}.png`);
            await page.screenshot({ path, animations: "disabled" });
            console.log(`Captured ${ready}: ${path}`);
        }
    }
    await writeFile(resolve(outputRoot, "index.html"), reportHtml());
    console.log(`Comparison page: ${resolve(outputRoot, "index.html")}`);
} finally {
    await browser?.close();
    await new Promise<void>((resolve_, reject) => {
        server.close((error) => (error ? reject(error) : resolve_()));
    });
    await vite.close();
}
