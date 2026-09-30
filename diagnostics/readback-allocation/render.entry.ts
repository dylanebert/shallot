import { build, Camera, Sear, Transform, AmbientLight, PointLight, Part, Color } from "../../src/index";
import { Render } from "../../src/core/rendering";
import { attachCanvas } from "../../src/core/rendering/view";
import { CanvasContext } from "../../src/engine/app/canvas.fixture";

export let controlSink: object;
export function control() { controlSink = { frame: 0 }; }

export default async function create(_input = "", device?: GPUDevice) {
    globalThis.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} } as unknown as typeof ResizeObserver;
    const app = await build({ plugins: [], device });
    const state = app.state;
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
    if (Render.shadeCount === 0) throw new Error("allocation subject did not render a shaded view");
    return {
        step: () => state.step(1 / 60),
        wait: () => state.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => { context.unconfigure(); app.dispose(); },
    };
}
