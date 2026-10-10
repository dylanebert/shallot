import { expect, test } from "playwright/test";

test("bare Chromium WebGPU submits 120 clear passes on animation frames", async ({ page }) => {
    await page.goto("/");
    const report = await page.evaluate(async () => {
        type Lost = { reason: string; message: string };
        type Report = {
            adapter: {
                vendor: string;
                architecture: string;
                device: string;
                description: string;
            } | null;
            frames: number;
            deviceLost: Lost | null;
            error: string | null;
        };
        const gpu = navigator.gpu;
        const adapter = await gpu?.requestAdapter();
        if (!adapter)
            return {
                adapter: null,
                frames: 0,
                deviceLost: null,
                error: "navigator.gpu.requestAdapter() returned null",
            } satisfies Report;
        const info = adapter.info;
        const identity = {
            vendor: info.vendor,
            architecture: info.architecture,
            device: info.device,
            description: info.description,
        };
        try {
            const device = await adapter.requestDevice();
            const canvas = document.querySelector<HTMLCanvasElement>("#surface");
            const context = canvas?.getContext("webgpu");
            if (!context)
                return {
                    adapter: identity,
                    frames: 0,
                    deviceLost: null,
                    error: "canvas.getContext('webgpu') returned null",
                } satisfies Report;
            context.configure({
                device,
                format: gpu.getPreferredCanvasFormat(),
                alphaMode: "opaque",
            });
            return await new Promise<Report>((resolve) => {
                let frames = 0;
                let deviceLost: Lost | null = null;
                let error: string | null = null;
                let finished = false;
                const finish = () => {
                    if (finished) return;
                    finished = true;
                    clearTimeout(watchdog);
                    resolve({ adapter: identity, frames, deviceLost, error });
                };
                const watchdog = setTimeout(() => {
                    error = "no device loss or 120 animation frames within 20 seconds";
                    finish();
                }, 20_000);
                void device.lost.then((info) => {
                    deviceLost = { reason: info.reason, message: info.message };
                    finish();
                });
                const clear = () => {
                    if (finished) return;
                    try {
                        const encoder = device.createCommandEncoder({ label: "bare-smoke-clear" });
                        const pass = encoder.beginRenderPass({
                            colorAttachments: [
                                {
                                    view: context.getCurrentTexture().createView(),
                                    clearValue: { r: 0.12, g: 0.24, b: 0.48, a: 1 },
                                    loadOp: "clear",
                                    storeOp: "store",
                                },
                            ],
                        });
                        pass.end();
                        device.queue.submit([encoder.finish()]);
                        frames++;
                    } catch (cause) {
                        error = cause instanceof Error ? cause.message : String(cause);
                        finish();
                        return;
                    }
                    if (frames >= 120) {
                        void device.queue.onSubmittedWorkDone().then(finish, (cause: unknown) => {
                            error = cause instanceof Error ? cause.message : String(cause);
                            finish();
                        });
                    } else requestAnimationFrame(clear);
                };
                requestAnimationFrame(clear);
            });
        } catch (cause) {
            return {
                adapter: identity,
                frames: 0,
                deviceLost: null,
                error: cause instanceof Error ? cause.message : String(cause),
            } satisfies Report;
        }
    });
    console.info("bare Chromium WebGPU reproduction:", JSON.stringify(report));
    expect(report.error, JSON.stringify(report)).toBeNull();
    expect(report.frames, JSON.stringify(report)).toBeGreaterThanOrEqual(120);
    expect(report.deviceLost, JSON.stringify(report)).toBeNull();
});
