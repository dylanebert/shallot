import { createApp, Camera, StandardRenderer, Transform, AmbientLight, PointLight, MeshInstance } from "../../src/index";
import { Render } from "../../src/core/rendering";
import { attachCanvas } from "../../src/core/rendering/view";
import { CanvasContext } from "../../src/engine/app/canvas.fixture";


export let controlSink: object;
export function control() { controlSink = { frame: 0 }; }

export default async function create(_input = "", device?: GPUDevice) {
    const resizeObserver = Object.getOwnPropertyDescriptor(globalThis, "ResizeObserver");
    globalThis.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} } as unknown as typeof ResizeObserver;
    const app = await createApp({ plugins: [], device });
    const world = app.world;
    let validation: GPUError | undefined;
    let failValidation!: (error: GPUError) => void;
    const failed = new Promise<never>((_, reject) => { failValidation = reject; });
    void failed.catch(() => {});
    const onError = (event: GPUUncapturedErrorEvent) => { validation ??= event.error; failValidation(validation); };
    world.gpu.device.addEventListener("uncapturederror", onError);
    let context: CanvasContext;
    const canvas = {
        width: 32, height: 24, style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = world.create();
    world.add(camera, Transform); world.add(camera, Camera); world.add(camera, StandardRenderer);
    world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, world);
    const ambient = world.create(); world.add(ambient, AmbientLight);
    const light = world.create(); world.add(light, Transform); world.add(light, PointLight);
    const part = world.create(); world.add(part, Transform); world.add(part, MeshInstance);
    world.step(1 / 60);
    if (world.resource(Render).shadeCount === 0) throw new Error("allocation subject did not render a shaded view");
    return {
        world,
        step: () => { if (validation) throw validation; world.step(1 / 60); },
        wait: async () => {
            let timer: ReturnType<typeof setTimeout> | undefined;
            try {
                await Promise.race([world.gpu.device.queue.onSubmittedWorkDone(), failed, new Promise<never>((_, reject) => {
                    timer = setTimeout(() => reject(new Error("render allocation frame submissions exceeded 750 ms")), 750);
                })]);
                if (validation) throw validation;
            } finally { clearTimeout(timer); }
        },
        dispose: () => {
            world.gpu.device.removeEventListener("uncapturederror", onError);
            context.unconfigure(); app.dispose();
            if (resizeObserver) Object.defineProperty(globalThis, "ResizeObserver", resizeObserver);
            else Reflect.deleteProperty(globalThis, "ResizeObserver");
        },
    };
}
