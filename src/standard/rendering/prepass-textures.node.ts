import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as s from "typegpu/std";
import {
    createTestGpuApp,
    disposeGpuApps,
    expectAdapterNotice,
    gpuScene,
} from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachCanvas,
    attachTexture,
    Camera,
    captureFrame,
    DepthPrepass,
    NormalPrepass,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import {
    AlphaMode,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    prepassDepth,
    prepassNormal,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);

// Undo presentation's sRGB encoding so captured bytes encode the measured quantity, not lighting.
const linearChannel = tgpu.fn(
    [d.f32],
    d.f32,
)((value) => {
    "use gpu";
    if (value <= 0.04045) return value / 12.92;
    return s.pow((value + 0.055) / 1.055, 2.4);
});
const Parameters = d.struct({ mode: d.u32 });
const Reader = materialType({
    name: "PrepassTextureReader",
    parameters: Parameters,
    readsPrepass: { depth: true, normal: true },
    defaults: { mode: 0 },
    fragment: (ctx) => {
        "use gpu";
        const depth = prepassDepth(ctx.fragCoord);
        const normal = prepassNormal(ctx.fragCoord);
        let value = d.vec3f(depth);
        if (ctx.parameters.mode === 1) value = s.add(s.mul(normal, 0.5), 0.5);
        if (ctx.parameters.mode === 2) {
            // Perspective reverse-Z is near / distance; this scene uses near = 1.
            value = d.vec3f(s.saturate((1 / depth - 1 / ctx.fragCoord.z) / 2));
        }
        let alpha = d.f32(1);
        if (ctx.parameters.mode === 3 && ctx.alphaMode === 0) {
            // Mask's depth fragment must see stubs, not sample its own attachments.
            alpha = s.select(d.f32(0), d.f32(1), depth === 0 && s.all(s.eq(normal, d.vec3f(0))));
        }
        return d.vec4f(
            linearChannel(value.x),
            linearChannel(value.y),
            linearChannel(value.z),
            alpha,
        );
    },
});
const Undeclared = materialType({
    name: "UndeclaredPrepassReader",
    parameters: Parameters,
    defaults: { mode: 0 },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(prepassDepth(ctx.fragCoord), 0, 0, 1);
    },
});

const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(Reader)],
    },
    (world) => {
        const mesh = world.resource(Meshes).add({
            name: "prepass-quad",
            vertices: new Float32Array([
                -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0,
                0, 0, 0, 1, 1,
            ]),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        });
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera, { near: 1, antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        let width = 33;
        let height = 33;
        let context: CanvasContext;
        const canvas = {
            width,
            height,
            clientWidth: width,
            clientHeight: height,
            style: { imageRendering: "auto" },
            getContext: () => context,
            getBoundingClientRect: () => ({ width, height }),
        } as unknown as HTMLCanvasElement;
        context = new CanvasContext(canvas, width, height);
        attachCanvas(world, camera, canvas);
        const acquire = context.getCurrentTexture.bind(context);
        context.getCurrentTexture = () => {
            context.configure({
                device: world.gpu.device,
                format: navigator.gpu.getPreferredCanvasFormat(),
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            return acquire();
        };
        const resize = (w: number, h: number) => {
            width = w;
            height = h;
            Object.assign(canvas, { clientWidth: w, clientHeight: h });
            context.setSize(w, h);
        };
        // Prepare both sample counts and Mask's fragment at warm, before live toggles.
        const multisampled = world.create();
        world.add(multisampled, Transform, { translation: [10, 0, 5, 0] });
        world.add(multisampled, Camera);
        world.add(multisampled, StandardRenderer);
        world.add(multisampled, NormalPrepass);
        attachTexture(world, multisampled, { width: 3, height: 3 });
        world.resource(Reader).add({ mode: 3 }, { alphaMode: AlphaMode.Mask(0.5) });
        const back = world.create();
        world.add(back, Transform);
        world.add(back, MeshInstance, { mesh });
        world.add(
            back,
            MeshMaterial,
            world.resource(Materials).add(
                StandardMaterial({
                    baseColor: [0, 0, 0, 1],
                    unlit: true,
                }),
            ),
        );
        const material = world.resource(Reader).add({ mode: 0 }, { alphaMode: AlphaMode.Blend });
        const front = world.create();
        world.add(front, Transform, { translation: [0, 0, 0.5, 0] });
        world.add(front, MeshInstance, { mesh });
        world.add(front, MeshMaterial, material);
        return { camera, front, material, canvas, resize };
    },
);

async function pixel(mode: number) {
    const {
        world,
        scene: { material, canvas },
    } = subject();
    world.resource(Reader).update(material, { mode });
    world.step(0);
    const pending = captureFrame(canvas);
    world.step(0);
    const frame = await pending;
    const at = (Math.floor(frame.height / 2) * frame.width + Math.floor(frame.width / 2)) * 4;
    return Array.from(frame.rgba.subarray(at, at + 3));
}
function nearBytes(actual: number[], expected: number[]) {
    for (let i = 0; i < 3; i++) expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThanOrEqual(1);
}

for (const antialias of [0, 1]) {
    test(`a declaring Blend material requests this frame's depth and world normal without camera prepass components at ${antialias ? 4 : 1}x`, async () => {
        const {
            world,
            scene: { camera },
        } = subject();
        expect(world.has(camera, DepthPrepass)).toBe(false);
        expect(world.has(camera, NormalPrepass)).toBe(false);
        world.storage(Camera).antialias.set(camera, antialias);
        nearBytes(await pixel(0), [51, 51, 51]); // reverse-Z: near / distance = 1 / 5
        nearBytes(await pixel(1), [128, 128, 255]);
    });
}

test("soft-intersection fade grows with the distance from the opaque prepass surface", async () => {
    const {
        world,
        scene: { front },
    } = subject();
    world.storage(Transform).translation.z.set(front, 0.25);
    const close = await pixel(2);
    nearBytes(close, [32, 32, 32]);
    world.storage(Transform).translation.z.set(front, 1.5);
    const far = await pixel(2);
    nearBytes(far, [191, 191, 191]);
    expect(far[0]!).toBeGreaterThan(close[0]!);
    world.storage(Transform).translation.z.set(front, 0.5);
});

test("resize and antialias toggles rebind the current view's prepass textures", async () => {
    const {
        world,
        scene: { camera, resize },
    } = subject();
    for (const [width, height, antialias] of [
        [47, 35, 1],
        [29, 41, 0],
        [33, 33, 1],
    ]) {
        resize(width, height);
        world.storage(Camera).antialias.set(camera, antialias);
        nearBytes(await pixel(0), [51, 51, 51]);
        nearBytes(await pixel(1), [128, 128, 255]);
    }
});

test("prepass requests disappear with the declaring type's last draw and return with its next draw", async () => {
    const {
        world,
        scene: { camera, front, material },
    } = subject();
    expect(world.has(camera, DepthPrepass)).toBe(false);
    expect(world.has(camera, NormalPrepass)).toBe(false);
    await pixel(0);
    expect(world.resource(Views).get(camera)!.depth).not.toBeNull();
    expect(world.resource(Views).get(camera)!.normal).not.toBeNull();
    world.remove(front, MeshMaterial);
    world.step(0);
    world.step(0);
    expect(world.resource(Views).get(camera)!.depth).toBeNull();
    expect(world.resource(Views).get(camera)!.normal).toBeNull();
    world.add(front, MeshMaterial, material);
    nearBytes(await pixel(0), [51, 51, 51]);
    expect(world.resource(Views).get(camera)!.depth).not.toBeNull();
    expect(world.resource(Views).get(camera)!.normal).not.toBeNull();
});

test("a DepthPrepass camera's clipped depth path reads zero depth and zero normal stubs", async () => {
    const {
        world,
        scene: { camera, material },
    } = subject();
    world.add(camera, DepthPrepass);
    world.add(camera, NormalPrepass);
    world.resource(Reader).update(material, { mode: 3 }, { alphaMode: AlphaMode.Mask(0.5) });
    // Mode 3 relies on the clip depth fragment's ctx.alphaMode being 0.
    // If the depth-pass fragment sees the real back surface instead of the stubs it discards;
    // the main-pass read then reports 1/5 rather than the front quad's 1/4.5.
    nearBytes(await pixel(3), [57, 57, 57]);
    expect(world.resource(Views).get(camera)!.depth).not.toBeNull();
    world.resource(Reader).update(material, {}, { alphaMode: AlphaMode.Blend });
    world.remove(camera, DepthPrepass);
    world.remove(camera, NormalPrepass);
});

test("calling prepassDepth without readsPrepass is refused with the material name and read", async () => {
    const { world } = subject();
    let app: Awaited<ReturnType<typeof createTestGpuApp>> | undefined;
    expectAdapterNotice(world.gpu.adapter);
    try {
        await expect(
            (async () => {
                app = await createTestGpuApp({
                    defaults: false,
                    device: world.gpu.device,
                    plugins: [
                        StandardRenderingPlugin,
                        MeshRenderPlugin,
                        MaterialPlugin(Undeclared),
                    ],
                });
            })(),
        ).rejects.toThrow(/UndeclaredPrepassReader[\s\S]*readsPrepass[\s\S]*depth/i);
    } finally {
        if (app) await disposeGpuApps([app]);
    }
});
