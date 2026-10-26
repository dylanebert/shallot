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
    Xform,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { precompile } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { probeBuffer } from "../../engine/runtime";
import {
    AlphaMode,
    AlphaPipelineKey,
    DrawIndexedIndirect,
    Draws,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    materialTypeId,
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
const RegisteredReader = materialType({
    name: "RegisteredPrepassReader",
    parameters: Parameters,
    defaults: { mode: 0 },
    readsPrepass: { depth: true, normal: true },
    depthPass: { prepass: false, shadows: false },
    fragment: (ctx) => {
        "use gpu";
        let value = d.vec3f(prepassDepth(ctx.fragCoord));
        if (ctx.parameters.mode === 1) value = s.add(s.mul(prepassNormal(ctx.fragCoord), 0.5), 0.5);
        return d.vec4f(linearChannel(value.x), linearChannel(value.y), linearChannel(value.z), 1);
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

const sampleLayout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepthMultisampled2d() },
    output: { storage: d.arrayOf(d.vec4f), access: "mutable" },
});
const sampleKernel = tgpu.computeFn({
    workgroupSize: [1, 1],
    in: { id: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const size = s.textureDimensions(sampleLayout.$.depth);
    const px = d.vec2i(input.id.xy);
    sampleLayout.$.output[input.id.y * size.x + input.id.x] = d.vec4f(
        s.textureLoad(sampleLayout.$.depth, px, 0),
        s.textureLoad(sampleLayout.$.depth, px, 1),
        s.textureLoad(sampleLayout.$.depth, px, 2),
        s.textureLoad(sampleLayout.$.depth, px, 3),
    );
});
const probeState = {
    create: () => ({
        pipeline: null as ReturnType<import("typegpu").TgpuRoot["createComputePipeline"]> | null,
    }),
};
const samplePlugin = {
    name: "PrepassSampleProbe",
    warm(world: import("../../engine").World) {
        const state = world.resource(probeState);
        state.pipeline = world.gpu.root.createComputePipeline({ compute: sampleKernel });
        precompile(world, "prepass-sample-probe", () => state.pipeline!);
    },
};
const UndeclaredNormal = materialType({
    name: "UndeclaredNormalReader",
    parameters: Parameters,
    defaults: { mode: 0 },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(prepassNormal(ctx.fragCoord), 1);
    },
});
const DepthOnlyNormal = materialType({
    name: "DepthOnlyNormalReader",
    parameters: Parameters,
    defaults: { mode: 0 },
    readsPrepass: { depth: true },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(prepassNormal(ctx.fragCoord), 1);
    },
});

const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(Reader),
            MaterialPlugin(RegisteredReader),
            samplePlugin,
        ],
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
        const registeredBindings = {
            eids: world.gpu.root
                .createBuffer(d.arrayOf(d.vec4u, 1), [d.vec4u(0xffffffff, 0, 0, 0)])
                .$usage("storage"),
            globalTransforms: world.gpu.root
                .createBuffer(d.arrayOf(Xform, 1), [
                    { pos: d.vec3f(0, 0, 0.5), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) },
                ])
                .$usage("storage"),
        };
        const registeredArgs = world.gpu.root.createBuffer(DrawIndexedIndirect).$usage("indirect");
        return {
            camera,
            front,
            back,
            mesh,
            material,
            canvas,
            resize,
            registeredBindings,
            registeredArgs,
        };
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

test("resize and antialias transitions read the moved surface's new depth and rotated world normal", async () => {
    const {
        world,
        scene: { camera, back, resize },
    } = subject();
    try {
        for (const [width, height, antialias, z, angle] of [
            [47, 35, 1, -1, 0.25],
            [29, 41, 0, -2, 0.5],
            [33, 33, 1, -3, 0.75],
        ]) {
            resize(width, height);
            world.storage(Camera).antialias.set(camera, antialias);
            world.storage(Transform).translation.z.set(back, z);
            world.storage(Transform).rotation.y.set(back, Math.sin(angle / 2));
            world.storage(Transform).rotation.w.set(back, Math.cos(angle / 2));
            nearBytes(await pixel(0), Array(3).fill(Math.round(255 / (5 - z))));
            nearBytes(await pixel(1), [
                Math.round((Math.sin(angle) * 0.5 + 0.5) * 255),
                128,
                Math.round((Math.cos(angle) * 0.5 + 0.5) * 255),
            ]);
        }
    } finally {
        world.storage(Transform).translation.z.set(back, 0);
        world.storage(Transform).rotation.y.set(back, 0);
        world.storage(Transform).rotation.w.set(back, 1);
        resize(33, 33);
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

for (const antialias of [0, 1]) {
    test(`a public registered Draw requests prepass textures without declaring mesh rows at ${antialias ? 4 : 1}x`, async () => {
        const {
            world,
            scene: { camera, front, mesh: sourceMesh, registeredBindings, registeredArgs },
        } = subject();
        world.remove(front, MeshInstance);
        world.storage(Camera).antialias.set(camera, antialias);
        world.step(0);
        const mesh = world.resource(Meshes).get(sourceMesh)!;
        const registeredMesh = world
            .resource(Meshes)
            .add({ ...mesh, name: "registered-prepass-quad", bindings: registeredBindings });
        registeredArgs.write({
            indexCount: mesh.indexCount,
            instanceCount: 1,
            firstIndex: mesh.indexBase,
            baseVertex: 0,
            firstInstance: 0,
        });
        world.resource(Draws).register({
            name: "public-prepass-reader",
            materialType: materialTypeId(world, RegisteredReader),
            alphaPipelineKey: AlphaPipelineKey.Opaque,
            mesh: registeredMesh,
            args: { indirect: registeredArgs },
        });
        try {
            world.resource(RegisteredReader).update(0, { mode: 0 });
            nearBytes(await pixel(0), [51, 51, 51]);
            expect(world.resource(Views).get(camera)!.depth).not.toBeNull();
            world.resource(RegisteredReader).update(0, { mode: 1 });
            nearBytes(await pixel(1), [128, 128, 255]);
        } finally {
            world.resource(Draws).delete("public-prepass-reader");
            world.resource(RegisteredReader).update(0, { mode: 0 });
            world.add(front, MeshInstance, { mesh: sourceMesh });
        }
    });
}

test("a width-one view reads a real world normal rather than the absent-normal stub", async () => {
    const {
        world,
        scene: { camera, resize },
    } = subject();
    resize(1, 33);
    world.storage(Camera).antialias.set(camera, 0);
    try {
        nearBytes(await pixel(1), [128, 128, 255]);
    } finally {
        resize(33, 33);
    }
});

test("4x reads sample zero at an edge whose sample zero and sample one depths differ", async () => {
    const {
        world,
        scene: { camera, back, material, canvas },
    } = subject();
    world.storage(Camera).antialias.set(camera, 1);
    world.storage(Transform).scale.x.set(back, 0.37);
    world.storage(Transform).translation.x.set(back, 0.013);
    try {
        world.resource(Reader).update(material, { mode: 0 });
        world.step(0);
        const pending = captureFrame(canvas);
        world.step(0);
        const frame = await pending;
        const view = world.resource(Views).get(camera)!;
        const output = world.gpu.root
            .createBuffer(d.arrayOf(d.vec4f, view.width * view.height))
            .$usage("storage");
        try {
            const group = world.gpu.root.createBindGroup(sampleLayout, {
                depth: view.depth!,
                output,
            });
            world
                .resource(probeState)
                .pipeline!.with(group)
                .dispatchWorkgroups(view.width, view.height);
            const probe = await probeBuffer(world, world.gpu.root.unwrap(output));
            const samples = new Float32Array(probe.bytes);
            let different = 0;
            for (let y = 8; y < 25; y++)
                for (let x = 10; x < 23; x++) {
                    const at = (y * view.width + x) * 4;
                    if (Math.abs(samples[at]! - samples[at + 1]!) < 0.1) continue;
                    different++;
                    expect(
                        Math.abs(frame.rgba[at]! - Math.round(samples[at]! * 255)),
                    ).toBeLessThanOrEqual(1);
                }
            expect(different).toBeGreaterThan(0);
        } finally {
            output.destroy();
        }
    } finally {
        world.storage(Transform).scale.x.set(back, 1);
        world.storage(Transform).translation.x.set(back, 0);
    }
});

for (const type of [UndeclaredNormal, DepthOnlyNormal]) {
    test(`${type.name} refuses a normal read without readsPrepass.normal`, async () => {
        const { world } = subject();
        expectAdapterNotice(world.gpu.adapter);
        let app: Awaited<ReturnType<typeof createTestGpuApp>> | undefined;
        try {
            await expect(
                (async () => {
                    app = await createTestGpuApp({
                        defaults: false,
                        device: world.gpu.device,
                        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(type)],
                    });
                })(),
            ).rejects.toThrow(new RegExp(`${type.name}[\\s\\S]*readsPrepass\\.normal`));
        } finally {
            if (app) await disposeGpuApps([app]);
        }
    });
}

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
