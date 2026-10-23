import { expect, test } from "playwright/test";

test("only pre-handoff render pipelines bind once in compatible throwaway bundles", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
        if (
            (message.type() === "warning" || message.type() === "error") &&
            !message.text().includes('Pipeline "late-') &&
            !message.text().includes("WebGPU reports a fallback adapter")
        )
            errors.push(message.text());
    });
    await page.addInitScript(() => {
        const primes: unknown[] = [];
        Reflect.set(window, "primes", primes);
        const create = GPUDevice.prototype.createRenderBundleEncoder;
        GPUDevice.prototype.createRenderBundleEncoder = function (descriptor) {
            const encoder = create.call(this, descriptor);
            const set = encoder.setPipeline.bind(encoder);
            encoder.setPipeline = (pipeline) => {
                primes.push({ label: pipeline.label, ...descriptor });
                set(pipeline);
            };
            for (const name of [
                "draw",
                "drawIndexed",
                "drawIndirect",
                "drawIndexedIndirect",
            ] as const) {
                encoder[name] = () => {
                    throw new Error("bundle preparation drew work");
                };
            }
            return encoder;
        };
        GPUQueue.prototype.submit = () => {
            throw new Error("bundle preparation submitted work");
        };
    });
    await page.goto("/bundles.html");
    await page.waitForFunction(() => Reflect.get(window, "ready"));
    expect(await page.evaluate(() => Reflect.get(window, "primes"))).toEqual([
        {
            label: "early-generator-sync",
            colorFormats: ["rgba8unorm"],
            depthStencilFormat: undefined,
            sampleCount: 1,
        },
        {
            label: "early-generator-async",
            colorFormats: ["rgba8unorm"],
            depthStencilFormat: undefined,
            sampleCount: 1,
        },
        {
            label: "early-sync",
            colorFormats: ["rgba8unorm"],
            depthStencilFormat: "depth24plus-stencil8",
            sampleCount: 4,
        },
        {
            label: "early-async",
            colorFormats: ["rgba8unorm"],
            depthStencilFormat: "depth24plus-stencil8",
            sampleCount: 4,
        },
        {
            label: "early-depth",
            colorFormats: [],
            depthStencilFormat: "depth24plus-stencil8",
            sampleCount: 1,
        },
        {
            label: "early-mutated",
            colorFormats: ["rgba8unorm"],
            depthStencilFormat: "depth24plus",
            sampleCount: 4,
        },
        {
            label: "early-null-slot",
            colorFormats: ["rgba8unorm", null, "bgra8unorm"],
            depthStencilFormat: undefined,
            sampleCount: 1,
        },
    ]);
    expect(errors).toEqual([]);
});
