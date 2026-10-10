import { chromium } from "playwright";

const windowsBaseArgs = [
    "--enable-unsafe-webgpu",
    "--enable-features=WebGPUDeveloperFeatures",
    "--enable-webgpu-developer-features",
    "--enable-gpu",
];

const variants = [
    { name: "a", args: ["--use-webgpu-adapter=swiftshader"] },
    {
        name: "b",
        args: ["--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"],
    },
    {
        name: "c",
        args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"],
    },
    {
        name: "d",
        args: ["--use-angle=d3d11-warp", "--use-webgpu-adapter=d3d11"],
    },
    { name: "e", args: ["--ignore-gpu-blocklist"] },
] as const;

if (process.platform !== "win32") throw new Error("Windows adapter probe requires windows-latest");

function withTimeout<T>(promise: Promise<T>, label: string, milliseconds: number): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`${label} timed out after ${milliseconds} ms`)),
            milliseconds,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}

for (const variant of variants) {
    let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
    try {
        browser = await chromium.launch({
            channel: "chromium",
            args: [...windowsBaseArgs, ...variant.args],
            ignoreDefaultArgs: ["--enable-unsafe-swiftshader"],
        });
        const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
        const phases: string[] = [];
        page.on("console", (message) => {
            const text = message.text();
            if (text.startsWith("WINDOWS_ADAPTER_PHASE ")) phases.push(text);
        });
        await page.route("http://localhost/**", (route) =>
            route.fulfill({
                status: 200,
                contentType: "text/html",
                body: '<canvas id="frame" width="1280" height="720"></canvas>',
            }),
        );
        await page.goto("http://localhost/windows-adapter-diagnostic");
        const probe = await withTimeout(
            page.evaluate(async (variantName) => {
                const mark = (phase: string): void =>
                    console.log(`WINDOWS_ADAPTER_PHASE ${variantName} ${phase}`);
                const bounded = <T>(promise: Promise<T>, label: string): Promise<T> =>
                    new Promise((resolve, reject) => {
                        const timer = window.setTimeout(
                            () => reject(new Error(`${label} timed out after 15000 ms`)),
                            15000,
                        );
                        promise.then(
                            (value) => {
                                window.clearTimeout(timer);
                                resolve(value);
                            },
                            (error) => {
                                window.clearTimeout(timer);
                                reject(error);
                            },
                        );
                    });
                const result: {
                    secureContext: boolean;
                    adapter?: {
                        vendor: string;
                        architecture: string;
                        device: string;
                        description: string;
                        isFallbackAdapter: boolean;
                    };
                    deviceCreated?: boolean;
                    submitted?: boolean;
                    frame?: {
                        format: string;
                        rgba: number[];
                        expected: number[];
                        matches: boolean;
                    };
                    stoppedAt: string;
                    error?: string;
                } = {
                    secureContext: isSecureContext,
                    stoppedAt: "navigator.gpu",
                };
                mark(`secure context=${isSecureContext}; navigator.gpu=${Boolean(navigator.gpu)}`);
                let device: GPUDevice | undefined;
                let readback: GPUBuffer | undefined;
                try {
                    if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");
                    result.stoppedAt = "requestAdapter";
                    mark("before requestAdapter");
                    const adapter = await bounded(navigator.gpu.requestAdapter(), "requestAdapter");
                    mark(`after requestAdapter; adapter=${Boolean(adapter)}`);
                    if (!adapter) throw new Error("requestAdapter returned null");
                    const info = adapter.info;
                    const fallback = adapter as GPUAdapter & { isFallbackAdapter?: boolean };
                    result.adapter = {
                        vendor: info.vendor,
                        architecture: info.architecture,
                        device: info.device,
                        description: info.description,
                        isFallbackAdapter: Boolean(fallback.isFallbackAdapter),
                    };

                    result.stoppedAt = "requestDevice";
                    mark("before requestDevice");
                    device = await bounded(adapter.requestDevice(), "requestDevice");
                    result.deviceCreated = true;
                    mark("after requestDevice");

                    const canvas = document.querySelector<HTMLCanvasElement>("#frame")!;
                    const context = canvas.getContext("webgpu");
                    if (!context) throw new Error("canvas.getContext('webgpu') returned null");
                    const format = navigator.gpu.getPreferredCanvasFormat();
                    context.configure({
                        device,
                        format,
                        alphaMode: "opaque",
                        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
                    });
                    const bytesPerRow = Math.ceil((canvas.width * 4) / 256) * 256;
                    readback = device.createBuffer({
                        size: bytesPerRow * canvas.height,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    });

                    result.stoppedAt = "first frame submit and readback";
                    mark("waiting for first requestAnimationFrame");
                    const expected = [32, 128, 191, 255];
                    const rgba = await new Promise<number[]>((resolve, reject) => {
                        const timer = window.setTimeout(
                            () =>
                                reject(new Error("requestAnimationFrame timed out after 15000 ms")),
                            15000,
                        );
                        requestAnimationFrame(async () => {
                            try {
                                mark("inside first requestAnimationFrame");
                                const texture = context.getCurrentTexture();
                                const encoder = device!.createCommandEncoder();
                                const pass = encoder.beginRenderPass({
                                    colorAttachments: [
                                        {
                                            view: texture.createView(),
                                            clearValue: { r: 0.125, g: 0.5, b: 0.75, a: 1 },
                                            loadOp: "clear",
                                            storeOp: "store",
                                        },
                                    ],
                                });
                                pass.end();
                                encoder.copyTextureToBuffer(
                                    { texture },
                                    { buffer: readback!, bytesPerRow, rowsPerImage: canvas.height },
                                    { width: canvas.width, height: canvas.height },
                                );
                                mark("before first queue.submit");
                                device!.queue.submit([encoder.finish()]);
                                result.submitted = true;
                                mark("after first queue.submit");
                                await bounded(
                                    device!.queue.onSubmittedWorkDone(),
                                    "queue completion",
                                );
                                await bounded(
                                    readback!.mapAsync(GPUMapMode.READ),
                                    "frame readback map",
                                );
                                const pixel = [
                                    ...new Uint8Array(readback!.getMappedRange()).slice(0, 4),
                                ];
                                const sample = format.startsWith("bgra")
                                    ? [pixel[2]!, pixel[1]!, pixel[0]!, pixel[3]!]
                                    : pixel;
                                mark("frame readback mapped");
                                resolve(sample);
                            } catch (error) {
                                reject(error);
                            } finally {
                                clearTimeout(timer);
                            }
                        });
                    });
                    result.frame = {
                        format,
                        rgba,
                        expected,
                        matches: rgba.every(
                            (channel, index) => Math.abs(channel - expected[index]!) <= 1,
                        ),
                    };
                    result.stoppedAt = "complete";
                } catch (error) {
                    result.error =
                        error instanceof Error ? `${error.name}: ${error.message}` : String(error);
                    mark(`failed at ${result.stoppedAt}: ${result.error}`);
                } finally {
                    readback?.destroy();
                    device?.destroy();
                }
                return result;
            }, variant.name),
            `variant ${variant.name} page probe`,
            45000,
        ).catch((error) => ({
            secureContext: false,
            stoppedAt: phases.at(-1) ?? "page evaluation",
            error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        }));
        console.log(
            `WINDOWS_ADAPTER_RESULT ${JSON.stringify({
                variant: variant.name,
                commonArgs: windowsBaseArgs,
                args: variant.args,
                phases,
                ...probe,
            })}`,
        );
    } catch (error) {
        console.log(
            `WINDOWS_ADAPTER_RESULT ${JSON.stringify({
                variant: variant.name,
                commonArgs: windowsBaseArgs,
                args: variant.args,
                stoppedAt: "browser launch",
                error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
            })}`,
        );
    } finally {
        await browser?.close().catch(() => {});
    }
}
