import { build, Camera, Sear, Transform, AmbientLight, PointLight, Part, Color } from "../../src/index";
import { Render } from "../../src/core/rendering";
import { attachCanvas } from "../../src/core/rendering/view";
import { CanvasContext } from "../../src/engine/app/canvas.fixture";


export let controlSink: object;
export function control() { controlSink = { frame: 0 }; }

export default async function create(_input = "", device?: GPUDevice) {
    const resizeObserver = Object.getOwnPropertyDescriptor(globalThis, "ResizeObserver");
    globalThis.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} } as unknown as typeof ResizeObserver;
    const app = await build({ plugins: [], device });
    const state = app.state;
    let validation: GPUError | undefined;
    let failValidation!: (error: GPUError) => void;
    const failed = new Promise<never>((_, reject) => { failValidation = reject; });
    void failed.catch(() => {});
    const onError = (event: GPUUncapturedErrorEvent) => { validation ??= event.error; failValidation(validation); };
    state.gpu.device.addEventListener("uncapturederror", onError);
    let context: CanvasContext;
    const canvas = {
        width: 32, height: 24, style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = state.create();
    state.add(camera, Transform); state.add(camera, Camera); state.add(camera, Sear);
    state.of(Transform).pos.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, state);
    const ambient = state.create(); state.add(ambient, AmbientLight);
    const light = state.create(); state.add(light, Transform); state.add(light, PointLight);
    const part = state.create(); state.add(part, Transform); state.add(part, Part); state.add(part, Color);
    state.step(1 / 60);
    if (state.resource(Render).shadeCount === 0) throw new Error("allocation subject did not render a shaded view");
    return {
        state,
        step: () => { if (validation) throw validation; state.step(1 / 60); },
        wait: async () => {
            let timer: ReturnType<typeof setTimeout> | undefined;
            try {
                await Promise.race([state.gpu.device.queue.onSubmittedWorkDone(), failed, new Promise<never>((_, reject) => {
                    timer = setTimeout(() => reject(new Error("render allocation frame submissions exceeded 750 ms")), 750);
                })]);
                if (validation) throw validation;
            } finally { clearTimeout(timer); }
        },
        dispose: () => {
            state.gpu.device.removeEventListener("uncapturederror", onError);
            context.unconfigure(); app.dispose();
            if (resizeObserver) Object.defineProperty(globalThis, "ResizeObserver", resizeObserver);
            else Reflect.deleteProperty(globalThis, "ResizeObserver");
        },
    };
}
