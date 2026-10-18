import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Meshes, createApp, Camera, StandardRenderer, Transform, AmbientLight, PointLight, MeshInstance, DirectionalLight, SpotLight } from "../../src/index";
import { LinesPlugin, drawLine, drawArrow, drawWireBox, GizmoAsset, Gizmo, registerGizmo } from "../../src/extras/lines";
import { ProfilePlugin } from "../../src/extras/profile";
import { BackgroundContext, backgroundLayout, CameraBackground, registerProducerTable, Materials, StandardMaterial, registerBackground } from "../../src/standard/rendering";
import { RenderContext } from "../../src/core/rendering";
import { attachCanvas } from "../../src/core/rendering/view";
import { CanvasContext } from "../../src/engine/app/canvas.fixture";
import {
    dispatchWorkgroups,
    dispatchWorkgroups2,
    dispatchWorkgroupsIndirect,
    draw,
    drawIndexedIndirect,
    endComputePass,
    endRenderPass,
    executeRenderBundles,
    gpuAllocationExpressions,
    setComputeBindGroup,
    setComputePipeline,
    setIndexBuffer,
    setRenderBindGroup,
    setRenderPipeline,
    setViewport,
} from "../../src/engine/runtime/allocation-site";


export let controlSink: object;
export function control() { controlSink = { frame: 0 }; }

export default async function create(input = "", device?: GPUDevice) {
    if (input === "profile") {
        const app = await createApp({ defaults: false, plugins: [ProfilePlugin], device });
        return {
            world: app.world,
            warmSamplingSessions: true,
            step: () => app.world.step(1 / 60),
            wait: async () => {},
            dispose: () => app.dispose(),
        };
    }
    const resizeObserver = Object.getOwnPropertyDescriptor(globalThis, "ResizeObserver");
    globalThis.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} } as unknown as typeof ResizeObserver;
    const app = await createApp({ plugins: [LinesPlugin], device });
    const world = app.world;
    const lost = world.gpu.device.lost.then((info) => {
        throw new Error(`render allocation device lost: ${info.message}`);
    });
    void lost.catch(() => {});
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
    world.storage(Transform).translation.set(camera, 0, 0, 5);
    attachCanvas(world, camera, canvas);
    if (input === "background") {
        // A backdrop with a bound uniform takes forward's background path every frame, as the sky does.
        const color = d.struct({ value: d.vec3f });
        const layout = backgroundLayout({ color: { type: "uniform", struct: color } });
        const bindings = { color: world.gpu.root.createBuffer(color, { value: d.vec3f(0.2, 0.4, 0.8) }).$usage("uniform") };
        const background = registerBackground(world, {
            name: "allocation-background",
            layout,
            fs: tgpu.fn([BackgroundContext], d.vec3f)(() => {
                "use gpu";
                return layout.$.color.value;
            }),
            bindings: () => bindings,
        });
        world.add(camera, CameraBackground, { background });
    }
    const ambient = world.create(); world.add(ambient, AmbientLight);
    const light = world.create(); world.add(light, Transform); world.add(light, PointLight);
    const sun = world.create(); world.add(sun, Transform); world.add(sun, DirectionalLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const point = world.create(); world.add(point, Transform); world.add(point, PointLight);
    // Non-integral placement: optimized code boxes a number argument of an out-of-line call unless it is integral.
    world.storage(Transform).translation.set(point, 1.3, 1.1, 0.7);
    world.storage(PointLight).shadowMapsEnabled.set(point, 1);
    const spot = world.create(); world.add(spot, Transform); world.add(spot, SpotLight);
    world.storage(Transform).translation.set(spot, -1.2, 1.3, 2.1);
    world.storage(Transform).rotation.set(spot, 0.1, 0.2, 0.3, Math.sqrt(0.86));
    world.storage(SpotLight).shadowMapsEnabled.set(spot, 1);
    const meshes = world.resource(Meshes);
    const count = world.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    world.own(count);
    const geometry = meshes.values().next().value!;
    world.gpu.device.queue.writeBuffer(count, 0, new Uint32Array([geometry.indexCount]));
    const mesh = meshes.add({ ...geometry, name: "allocation-gpu-count", indirectIndexCount: { buffer: count } });
    const part = world.create(); world.add(part, Transform); world.add(part, MeshInstance, { mesh });
    const materials = world.resource(Materials);
    const material = materials.add(StandardMaterial());
    // An animated material parameter, written every frame as the materials example's wave is.
    const materialValues = { perceptualRoughness: 0.5 };
    const producer = registerProducerTable(world, { name: "allocation-producer", capacity: 1, meshes: [mesh], materials: [material] });
    producer.placements.write([{ pos: d.vec3f(1, 0, 0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) }]);
    producer.meshInstances.write([{ mesh, materialType: material.type, material: material.material, alphaMode: 0, alphaCutoff: 0.5, flags: 0, cullBounds: d.vec4f(0, 0, 0, -1) }]);
    producer.count.write(1);
    const a = new Float32Array([-1, 0, 0]);
    const b = new Float32Array([1, 0, 0]);
    const min = new Float32Array([-0.3, -0.4, -0.5]);
    const max = new Float32Array([1, 1, 1]);
    const asset = new GizmoAsset();
    drawLine(asset, a, b, 0xffaa55);
    drawArrow(asset, a, b, 0x55aaff);
    drawWireBox(asset, min, max, 0xffffff);
    const gizmo = world.create();
    world.add(gizmo, Transform);
    world.add(gizmo, Gizmo, { handle: registerGizmo(world, asset) });
    let frame = 0;
    const compose = () => {
        materialValues.perceptualRoughness = 0.25 + (frame % 60) / 120;
        materials.update(material, materialValues);
        drawLine(world, a, b, 0xffaa55);
        drawArrow(world, a, b, 0x55aaff);
        drawWireBox(world, min, max, 0xffffff);
        world.storage(Transform).translation.set(gizmo, (frame++ % 60) / 60, 0.2, 0.3);
    };
    compose();
    world.step(1 / 60);
    if (world.resource(RenderContext).shadeCount === 0) throw new Error("allocation subject did not render a shaded view");
    return {
        world,
        warmSamplingSessions: true,
        allocationExpressions: gpuAllocationExpressions,
        allocationFunctions: [
            world.gpu.sync,
            endComputePass,
            endRenderPass,
            setComputePipeline,
            setComputeBindGroup,
            executeRenderBundles,
            dispatchWorkgroups,
            dispatchWorkgroups2,
            dispatchWorkgroupsIndirect,
            setIndexBuffer,
            setViewport,
            drawIndexedIndirect,
            setRenderPipeline,
            setRenderBindGroup,
            draw,
        ],
        step: () => { if (validation) throw validation; compose(); world.step(1 / 60); },
        wait: async () => {
            await Promise.race([world.gpu.device.queue.onSubmittedWorkDone(), failed, lost]);
            if (validation) throw validation;
        },
        dispose: () => {
            world.gpu.device.removeEventListener("uncapturederror", onError);
            context.unconfigure(); app.dispose();
            if (resizeObserver) Object.defineProperty(globalThis, "ResizeObserver", resizeObserver);
            else Reflect.deleteProperty(globalThis, "ResizeObserver");
        },
    };
}
