import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";

const output = resolve(import.meta.dir, "../../.artifacts/portable-gpu-presentation");
mkdirSync(output, { recursive: true });
const server = Bun.serve({
    port: 4179,
    hostname: "127.0.0.1",
    fetch: () =>
        new Response(Bun.file(new URL("./index.html", import.meta.url)), {
            headers: { "content-type": "text/html" },
        }),
});

const threeJsFlags = [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--no-sandbox",
];
const variants = [
    {
        name: "threejs-swiftshader-webgpu-vulkan",
        args: [...threeJsFlags, "--enable-unsafe-webgpu", "--enable-features=Vulkan"],
    },
    {
        name: "threejs-swiftshader-webgpu-vulkan-no-unsafe-swiftshader",
        args: [
            "--use-angle=swiftshader",
            "--no-sandbox",
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
        ],
    },
    {
        name: "chrome-web-ai-vulkan",
        args: [
            "--enable-unsafe-webgpu",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--disable-vulkan-surface",
        ],
    },
];

async function pixelEvidence(page, red, green) {
    return page.evaluate(
        async ({ red, green }) => {
            const pixels = async (base64) => {
                const bitmap = await createImageBitmap(
                    await (await fetch(`data:image/png;base64,${base64}`)).blob(),
                );
                const canvas = document.createElement("canvas");
                canvas.width = bitmap.width;
                canvas.height = bitmap.height;
                const context = canvas.getContext("2d");
                context.drawImage(bitmap, 0, 0);
                bitmap.close();
                return {
                    width: canvas.width,
                    height: canvas.height,
                    bytes: context.getImageData(0, 0, canvas.width, canvas.height).data,
                };
            };
            const first = await pixels(red);
            const second = await pixels(green);
            if (first.width !== second.width || first.height !== second.height)
                return { screenshotsDiffer: true, changedPixels: null, samples: [] };
            let changedPixels = 0;
            for (let offset = 0; offset < first.bytes.length; offset += 4) {
                if (
                    first.bytes[offset] !== second.bytes[offset] ||
                    first.bytes[offset + 1] !== second.bytes[offset + 1] ||
                    first.bytes[offset + 2] !== second.bytes[offset + 2] ||
                    first.bytes[offset + 3] !== second.bytes[offset + 3]
                )
                    changedPixels++;
            }
            const center = (first.height >> 1) * first.width * 4 + (first.width >> 1) * 4;
            return {
                screenshotsDiffer: changedPixels > 0,
                changedPixels,
                samples: [
                    Array.from(first.bytes.subarray(center, center + 4)),
                    Array.from(second.bytes.subarray(center, center + 4)),
                ],
            };
        },
        { red: red.toString("base64"), green: green.toString("base64") },
    );
}

try {
    for (const variant of variants) {
        const browser = await chromium.launch({
            headless: true,
            channel: "chromium",
            args: variant.args,
        });
        try {
            const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
            const pageErrors = [];
            page.on("pageerror", (error) => pageErrors.push(error.message));
            await page.goto("http://127.0.0.1:4179");
            await page.waitForFunction(
                () =>
                    window.webgpuRepro?.frames >= 60 ||
                    window.webgpuRepro?.deviceLoss ||
                    window.webgpuRepro?.error,
            );
            await page.waitForFunction(
                () =>
                    window.webgpuRepro?.readbacks.red !== undefined ||
                    window.webgpuRepro?.readbackErrors.length > 0 ||
                    window.webgpuRepro?.deviceLoss ||
                    window.webgpuRepro?.error,
            );
            const red = await page.screenshot();
            await page.waitForFunction(
                () =>
                    window.webgpuRepro?.frames >= 180 ||
                    window.webgpuRepro?.deviceLoss ||
                    window.webgpuRepro?.error,
            );
            if (await page.evaluate(() => window.webgpuRepro?.frames >= 180)) {
                await page.waitForFunction(
                    () =>
                        window.webgpuRepro?.readbacks.green !== undefined ||
                        window.webgpuRepro?.readbackErrors.some((error) =>
                            error.startsWith("green:"),
                        ) ||
                        window.webgpuRepro?.deviceLoss ||
                        window.webgpuRepro?.error,
                );
            }
            const green = await page.screenshot();
            writeFileSync(resolve(output, `${variant.name}-red.png`), red);
            writeFileSync(resolve(output, `${variant.name}-green.png`), green);
            const result = await page.evaluate(() => ({ ...window.webgpuRepro }));
            const screenshot = await pixelEvidence(page, red, green);
            const record = {
                variant: variant.name,
                browser: browser.version(),
                channel: "chromium",
                flags: variant.args,
                adapter: result.adapter,
                format: result.format,
                frames: result.frames,
                deviceLoss: result.deviceLoss,
                gpuErrors: result.errors,
                pageErrors,
                error: result.error,
                readbacks: result.readbacks,
                readbackErrors: result.readbackErrors,
                screenshotsDiffer: screenshot.screenshotsDiffer,
                changedPixels: screenshot.changedPixels,
                screenshotSamples: screenshot.samples,
                redReached: result.frames >= 60,
                greenReached: result.frames >= 180,
            };
            writeFileSync(
                resolve(output, `${variant.name}.json`),
                `${JSON.stringify(record, null, 2)}\n`,
            );
            console.log(JSON.stringify(record));
        } finally {
            await browser.close();
        }
    }
} finally {
    server.stop();
}
