// The capture fixture's scene: a WebGPU canvas whose composited frame is deterministic content at the
// declared capture geometry, stepped by an explicit tick rather than by wall time. It is deliberately
// engine-free, because the subject here is the capture and seat mechanism — a fixture that booted the
// engine would make an engine defect read as a capture defect.

import { CAPTURE_CONTRACT } from "@dylanebert/shallot/harness/capture";

/** the clear colour and the drawn tag colour, as the sRGB bytes a classifier sees. */
export const SCENE_CLEAR = [16, 20, 28] as const;
export const SCENE_TAG = [224, 32, 32] as const;
/** the tag quad covers NDC x in [-0.9,-0.1] and y in [-0.5,0.5]: 40% of the width by half the height. */
export const SCENE_TAG_PIXELS = (CAPTURE_CONTRACT.width * 0.4 * CAPTURE_CONTRACT.height) / 2;

const SHADER = /* wgsl */ `
struct Out { @builtin(position) pos: vec4f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    var xs = array<f32, 6>(-0.9, -0.1, -0.9, -0.1, -0.1, -0.9);
    var ys = array<f32, 6>(-0.5, -0.5, 0.5, -0.5, 0.5, 0.5);
    var out: Out;
    out.pos = vec4f(xs[i], ys[i], 0.0, 1.0);
    return out;
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(${(SCENE_TAG[0] / 255).toFixed(6)}, ${(SCENE_TAG[1] / 255).toFixed(6)}, ${(SCENE_TAG[2] / 255).toFixed(6)}, 1.0);
}
`;

/** the fixture's handle: its canvas, its stepped tick, and the GPU errors the device reported. */
export interface SceneHandle {
    canvas: HTMLCanvasElement;
    /** advance one tick, draw it, and resolve after the browser has composited it. */
    step(): Promise<void>;
    /** the number of steps drawn so far; the fixture's stepped clock. */
    tick(): number;
}

/** Build the fixture canvas and its pipeline, or throw the real WebGPU failure. */
export async function buildScene(): Promise<SceneHandle> {
    const gpuErrors: string[] = [];
    (window as { __gpuErrors?: string[] }).__gpuErrors = gpuErrors;
    const gpu = navigator.gpu;
    if (!gpu) throw new Error("navigator.gpu is unavailable in this browser");
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("navigator.gpu.requestAdapter() returned no adapter");
    const device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", (event) => {
        gpuErrors.push(String((event as GPUUncapturedErrorEvent).error.message));
    });
    const canvas = document.createElement("canvas");
    canvas.width = CAPTURE_CONTRACT.width;
    canvas.height = CAPTURE_CONTRACT.height;
    // The CSS box equals the backing store, so the driver's fixed viewport and this canvas are one
    // declared geometry rather than two that happen to agree.
    canvas.style.width = `${CAPTURE_CONTRACT.width}px`;
    canvas.style.height = `${CAPTURE_CONTRACT.height}px`;
    document.body.style.margin = "0";
    document.body.appendChild(canvas);
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("canvas.getContext('webgpu') returned no context");
    const format = gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
    const module = device.createShaderModule({ code: SHADER });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
    });
    const clearValue = {
        r: SCENE_CLEAR[0] / 255,
        g: SCENE_CLEAR[1] / 255,
        b: SCENE_CLEAR[2] / 255,
        a: 1,
    };
    let ticks = 0;
    return {
        canvas,
        tick: () => ticks,
        step(): Promise<void> {
            ticks += 1;
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        clearValue,
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            pass.setPipeline(pipeline);
            pass.draw(6);
            pass.end();
            device.queue.submit([encoder.finish()]);
            return new Promise<void>((done) => requestAnimationFrame(() => done()));
        },
    };
}
