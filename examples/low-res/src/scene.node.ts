import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    CameraMode,
    Materials,
    MeshInstance,
    MeshMaterial,
    Resolution,
    StandardMaterial,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import {
    attachTexture,
    captureTexture,
    type EffectPass,
    EffectPasses,
    fullscreenVertex,
    ImageBindings,
} from "@dylanebert/shallot/rendering";
import {
    beginRenderPass,
    draw,
    endRenderPass,
    precompile,
    setRenderPipeline,
} from "@dylanebert/shallot/runtime";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Posterize } from "./posterize";
import { Scene, ScenePlugin } from "./scene";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof openGpuApp>>;
let world: World;
let camera: number;
let gradient: EffectPass;
const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
beforeAll(async () => {
    app = await openGpuApp({
        plugins: [
            {
                name: "CaptureLowRes",
                dependencies: [ScenePlugin],
                initialize(w) {
                    const camera = w.resource(Scene).camera;
                    const height = w.storage(Resolution).height.get(camera);
                    // Texture targets keep their owner's size; mirror the camera's canvas resolution.
                    attachTexture(w, camera, { width: Math.round((height * 16) / 9), height });
                },
                warm(w) {
                    const fragment = tgpu.fragmentFn({
                        in: { position: d.builtin.position },
                        out: d.vec4f,
                    })((input) => {
                        "use gpu";
                        return d.vec4f(d.vec3f(0.2 + (input.position.x / 640) * 0.6), 1);
                    });
                    const pipeline = w.gpu.root.unwrap(
                        w.gpu.root.createRenderPipeline({
                            vertex: fullscreenVertex,
                            fragment,
                            targets: { format: "rgba8unorm" },
                        }),
                    );
                    precompile(w, "test-gradient", () => [pipeline]);
                    gradient = (w, _camera, _view, _input, output) => {
                        const pass = beginRenderPass(w.frameEncoder()!, {
                            colorAttachments: [{ view: output, loadOp: "clear", storeOp: "store" }],
                        });
                        setRenderPipeline(pass, pipeline);
                        draw(pass, 3);
                        endRenderPass(pass);
                    };
                },
            },
        ],
    });
    world = app.world;
    camera = world.resource(Scene).camera;
    console.info("low-res adapter", world.gpu.adapter);
});
afterAll(() => app?.dispose());
async function frame() {
    world.step(0);
    return captureTexture(world, camera);
}
test("Resolution renders 360 lines and every display channel lies on the level grid", async () => {
    const image = await frame();
    expect(image.height).toBe(360);
    let invalid = 0;
    for (let i = 0; i < image.rgba.length; i++)
        if (i % 4 !== 3) {
            const c = image.rgba[i];
            if (Math.abs(c - Math.round((Math.round((c / 255) * 7) / 7) * 255)) > 1) invalid++;
        }
    expect(invalid).toBe(0);
});
test("a smooth gradient holds the render-pixel 4 by 4 Bayer mix", async () => {
    await frame();
    const stack = world.resource(EffectPasses).get(camera)!.after;
    stack.unshift(gradient);
    try {
        const { rgba } = await frame();
        let wrong = 0;
        let low = 0;
        let high = 0;
        for (let y = 100; y < 116; y++)
            for (let x = 300; x < 316; x++) {
                const source = Math.round((0.2 + ((x + 0.5) / 640) * 0.6) * 255) / 255;
                const rank = bayer[(y % 4) * 4 + (x % 4)];
                const expected = Math.round((Math.floor(source * 7 + (rank + 0.5) / 16) / 7) * 255);
                if (Math.abs(rgba[(y * 640 + x) * 4] - expected) > 1) wrong++;
                if (rgba[(y * 640 + x) * 4] < 128) low++;
                else high++;
            }
        expect(wrong).toBe(0);
        expect(low).toBeGreaterThan(0);
        expect(high).toBeGreaterThan(0);
    } finally {
        stack.splice(stack.indexOf(gradient), 1);
    }
});
test("the crate's registered sampler preserves hard texel steps across its face", async () => {
    world.remove(camera, Orbit);
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, 2);
    world.storage(Transform).translation.z.set(camera, 5);
    for (const axis of ["x", "y"] as const)
        world.storage(Transform).translation[axis].set(camera, 0);
    for (const axis of ["x", "y", "z"] as const)
        world.storage(Transform).rotation[axis].set(camera, 0);
    world.storage(Transform).rotation.w.set(camera, 1);
    const blocks = [...world.query([MeshInstance, MeshMaterial])];
    const crate = blocks[2];
    for (const eid of blocks) world.storage(Transform).translation.x.set(eid, 100);
    for (const axis of ["x", "y", "z"] as const) {
        world.storage(Transform).translation[axis].set(crate, 0);
        world.storage(Transform).scale[axis].set(crate, 2);
    }
    world.add(
        crate,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ unlit: true, baseColorTexture: 0 })),
    );
    const texture = world.gpu.device.createTexture({
        size: [16, 16, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(texture);
    const pixels = new Uint8Array(16 * 16 * 4);
    for (let y = 0; y < 16; y++)
        for (let x = 0; x < 16; x++)
            pixels.set(x % 2 ? [255, 255, 255, 255] : [0, 0, 0, 255], (y * 16 + x) * 4);
    world.gpu.device.queue.writeTexture(
        { texture },
        pixels,
        { bytesPerRow: 64, rowsPerImage: 16 },
        [16, 16, 1],
    );
    world.resource(ImageBindings).images = texture;
    world.storage(Posterize).dither.set(camera, 0);
    const { rgba } = await frame();
    const values = new Set<number>();
    for (let x = 240; x < 400; x++) values.add(rgba[(180 * 640 + x) * 4]);
    expect([...values].sort((a, b) => a - b)).toEqual([0, 182]);
});
