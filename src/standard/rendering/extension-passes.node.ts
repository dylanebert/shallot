import { expect, setDefaultTimeout, test } from "bun:test";
import { lookAtRotation, probeTexture, type World } from "@dylanebert/shallot";
import { MeshInstance } from "@dylanebert/shallot/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import * as rendering from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuScene } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const {
    extendedMaterial,
    Materials,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} = rendering;
const ShiftParameters = d.struct({ offset: d.vec3f });
const Shifted = extendedMaterial(Materials, {
    name: "ExtensionVertexShift",
    parameters: ShiftParameters,
    defaults: { offset: d.vec3f(0.5, 0, 0) },
    vertex: (input, extension) => {
        "use gpu";
        return {
            world: std.add(input.world, d.vec4f(extension.offset, 0)),
            worldNormal: input.worldNormal,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return rendering.applyPbrLighting(rendering.pbrInputFromStandardMaterial(ctx));
    },
});
function observe(device: GPUDevice) {
    const textures = new Map<string, GPUTexture>();
    const shaders: GPUShaderModuleDescriptor[] = [];
    const layouts: GPUBindGroupLayoutDescriptor[] = [];
    const texture = device.createTexture.bind(device);
    const shader = device.createShaderModule.bind(device);
    const layout = device.createBindGroupLayout.bind(device);
    device.createTexture = (descriptor) => {
        const observed =
            descriptor.label?.startsWith("standard-depth-") ||
            descriptor.label === "standard-cascade-shadow-atlas";
        const result = texture(
            observed
                ? { ...descriptor, usage: descriptor.usage | GPUTextureUsage.COPY_SRC }
                : descriptor,
        );
        if (observed) textures.set(descriptor.label!, result);
        return result;
    };
    device.createShaderModule = (descriptor) => {
        shaders.push(descriptor);
        return shader(descriptor);
    };
    device.createBindGroupLayout = (descriptor) => {
        const captured = { ...descriptor, entries: Array.from(descriptor.entries) };
        layouts.push(captured);
        return layout(captured);
    };
    return { textures, shaders, layouts };
}
function scene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, DepthPrepass);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 64 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1, numCascades: 1 });
    world.resource(rendering.DirectionalLightShadowMap).size = 256;
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    return { camera, eid };
}
const Observations = {
    create: () => ({ value: undefined as ReturnType<typeof observe> | undefined }),
};
const Observer = {
    name: "ExtensionPassObservations",
    gpu: {},
    prepareGpu(world: World, gpu: World["gpu"]) {
        world.resource(Observations).value = observe(gpu.device);
    },
};
const shifted = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [Observer, StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(Shifted)],
    },
    (world) => {
        const observations = world.resource(Observations).value!;
        return {
            ...scene(world),
            ...observations,
            base: world.resource(Materials).add(StandardMaterial()),
            shifted: world
                .resource(Shifted)
                .add({ ...StandardMaterial(), offset: d.vec3f(0.5, 0, 0) }),
        };
    },
);
for (const pass of ["color", "depth", "shadow"] as const)
    test(`extension vertex moves ${pass} exactly like world translation`, async () => {
        const { world, scene: s } = shifted();
        const capture = async (extension: boolean, x: number) => {
            world.remove(s.eid, MeshMaterial);
            world.add(s.eid, MeshMaterial, extension ? s.shifted : s.base);
            world.storage(Transform).translation.set(s.eid, x, 0, 0, 0);
            world.step(1 / 60);
            world.step(1 / 60);
            return {
                color: (await captureTexture(world, s.camera)).rgba,
                depth: (await probeTexture(world, s.textures.get(`standard-depth-${s.camera}`)!))
                    .bytes,
                shadow: (
                    await probeTexture(world, s.textures.get("standard-cascade-shadow-atlas")!)
                ).bytes,
            };
        };
        const original = await capture(false, 0);
        const translated = await capture(false, 0.5);
        const extension = await capture(true, 0);
        expect(translated[pass], pass).not.toEqual(original[pass]);
        expect(extension[pass], pass).toEqual(translated[pass]);
    });

const Tint = d.struct({ color: d.vec4f });
// Defer the declaration to fixture preparation so the base revision's missing API is a test failure.
const binding = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            {
                name: "BindingExtensionFixture",
                dependencies: [Observer, StandardRenderingPlugin, MeshRenderPlugin],
                initialize(world) {
                    world.resource(BindingState);
                },
            },
        ],
    },
    (world) => world.resource(BindingState),
);
const BindingState = {
    create(world: World) {
        const observations = world.resource(Observations).value!;
        const layout = rendering.extensionLayout(Materials, d.struct({ gain: d.f32 }), {
            tint: { type: "uniform", struct: Tint },
        });
        const buffer = world.gpu.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        world.gpu.device.queue.writeBuffer(buffer, 0, new Float32Array([0.25, 0.5, 0.75, 1]));
        const resources = { tint: buffer };
        const type = extendedMaterial(Materials, {
            name: "Binding100Extension",
            layout,
            defaults: { gain: 1 },
            bindings: () => resources,
            fragment: (ctx, extension) => {
                "use gpu";
                const input = rendering.pbrInputFromStandardMaterial(ctx);
                return std.mul(layout.$.tint.color, extension.gain * input.occlusion);
            },
        });
        // Register before StandardRendering's warm step through the same public material resource seam.
        const handle = world
            .resource(type)
            .add({ ...StandardMaterial(), gain: 1 }, { alphaMode: rendering.AlphaMode.Mask() });
        const s = scene(world);
        world.add(s.eid, MeshMaterial, handle);
        return { ...observations, ...s, type };
    },
};
test("extension binding 100 is present in resolved colour and clipped depth shaders and the GPU layout", async () => {
    const { world, scene: s } = binding();
    world.step(0);
    world.step(0);
    const shot = await captureTexture(world, s.camera);
    const pixel = Array.from(shot.rgba.slice((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 4));
    expect(pixel).toEqual([137, 188, 225, 255]);
    const uses = s.shaders.filter((shader) => shader.code.includes("@binding(100)"));
    expect(uses.some((shader) => shader.code.includes("@fragment"))).toBe(true);
    expect(uses.length).toBeGreaterThanOrEqual(3);
    expect(
        s.layouts.some((layout) =>
            Array.from(layout.entries).some(
                (entry) => entry.binding === 100 && entry.buffer?.type === "uniform",
            ),
        ),
    ).toBe(true);
});
