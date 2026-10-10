import { appendFileSync } from "node:fs";
import { chromium } from "playwright";
import { expect, test } from "playwright/test";
import { CHROMIUM_USE } from "../chromium";
import { CHROMIUM_VARIANTS } from "./variants";

type DeviceLost = { reason: string; message: string };
type BareReport = {
    adapter: { vendor: string; architecture: string; device: string; description: string } | null;
    frames: number;
    deviceLost: DeviceLost | null;
    error: string | null;
};
type VariantReport = BareReport & {
    id: string;
    name: string;
    args: readonly string[];
};

async function probe(id: string, name: string, args: readonly string[]): Promise<VariantReport> {
    let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
    try {
        browser = await chromium.launch({ channel: CHROMIUM_USE.channel, args: [...args] });
        const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
        await page.goto("http://127.0.0.1:4176/");
        const report = await page.evaluate(async () => {
            type Lost = { reason: string; message: string };
            type Result = {
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
                } satisfies Result;
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
                    } satisfies Result;
                context.configure({
                    device,
                    format: gpu.getPreferredCanvasFormat(),
                    alphaMode: "opaque",
                });
                return await new Promise<Result>((resolve) => {
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
                            const encoder = device.createCommandEncoder({
                                label: "bare-smoke-clear",
                            });
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
                            void device.queue
                                .onSubmittedWorkDone()
                                .then(finish, (cause: unknown) => {
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
                } satisfies Result;
            }
        });
        return { id, name, args, ...report };
    } catch (cause) {
        return {
            id,
            name,
            args,
            adapter: null,
            frames: 0,
            deviceLost: null,
            error: cause instanceof Error ? cause.message : String(cause),
        };
    } finally {
        await browser?.close();
    }
}

test("compare Chromium WebGPU launch variants outside Shallot", async () => {
    const reports: VariantReport[] = [];
    for (const variant of CHROMIUM_VARIANTS)
        reports.push(await probe(variant.id, variant.name, variant.args));
    console.info("bare Chromium WebGPU variant table:", JSON.stringify(reports));
    const selected = reports.find(
        (report) => report.error === null && report.deviceLost === null && report.frames >= 120,
    );
    if (selected && process.env.GITHUB_OUTPUT)
        appendFileSync(process.env.GITHUB_OUTPUT, `selected=${selected.id}\n`);
    expect(selected, JSON.stringify(reports)).toBeDefined();
});
