import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { attachCanvas, Camera, captureFrame, RenderingPlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [RenderingPlugin] },
    (world) => {
        const canvas = {
            width: 4,
            height: 4,
            style: {},
            getBoundingClientRect: () => ({ width: 4, height: 4 }),
        } as unknown as HTMLCanvasElement;
        const context = new CanvasContext(canvas, 4, 4);
        canvas.getContext = (() => context) as unknown as typeof canvas.getContext;
        context.configure({
            device: world.gpu.device,
            format: "bgra8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        world.onDispose(() => context.unconfigure());
        const camera = world.create();
        world.add(camera, Camera);
        attachCanvas(world, camera, canvas);
        return canvas;
    },
);

test("a queued canvas capture rejects the first device error without a frame", async () => {
    const { world, scene: canvas } = subject();
    const start = performance.now();
    const capture = captureFrame(canvas);
    expectError(/GpuValidationError\]: .*usage/i);
    world.gpu.device.createBuffer({ size: 4, usage: 0 });
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        await expect(
            Promise.race([
                capture,
                new Promise<never>((_, reject) => {
                    timer = setTimeout(() => reject(new Error("captureFrame did not settle")), 800);
                }),
            ]),
        ).rejects.toThrow(/canvas frame capture: .*usage/i);
    } finally {
        clearTimeout(timer);
        console.log("queued capture settlement ms:", performance.now() - start);
    }
});
