import {
    Compute,
    type Mirror,
    MirrorPlugin,
    mesh,
    mirror,
    type Plugin,
    RenderPlugin,
    run,
    type System,
} from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import {
    BeginFrameSystem,
    DrawIndexedIndirect,
    Draws,
    Meshes,
} from "@dylanebert/shallot/render/core";
import {
    fsCtxSchema,
    PrepassSystem,
    registerSurface,
    surfaceLayout,
} from "@dylanebert/shallot/sear/core";
import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";

// The extension escape is intentionally end-to-end, not a collection of isolated API calls: an adopted
// raw device creates a raw buffer, TypeGPU wraps it with the draw schema, raw WGSL resolved through
// `tgpu.resolve` writes that buffer, Draws consumes the typed handle, and Mirror reads the same allocation
// back. The positive pixel gate proves the record reached a rendered product.

type TypedArgs = TgpuBuffer<typeof DrawIndexedIndirect> & StorageFlag & { usableAsIndirect: true };

let rawArgs: GPUBuffer;
let typedArgs: TypedArgs;
let readback: Mirror<TypedArgs>;
let resolvedWgsl = "";

const vertices = new Float32Array([
    -0.8, -0.7, 0, 0, 0, 0, 1, 0, 0.8, -0.7, 0, 1, 0, 0, 1, 0, 0, 0.8, 0, 0.5, 0, 0, 1, 1,
]);
const indices = new Uint32Array([0, 1, 2]);

const noWallsLayout = surfaceLayout({});
const noWallsFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)(() => {
    "use gpu";
    return d.vec4f(0.95, 0.2, 0.7, 1);
});

const RawProducerSystem: System = {
    name: "no-walls-raw-producer",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    async setup() {
        const { device, root } = Compute;
        const triangle = Meshes.get("no-walls-triangle");
        if (!triangle) throw new Error("no-walls: triangle mesh was not flushed");

        rawArgs = device.createBuffer({
            label: "no-walls-raw-draw-args",
            size: d.sizeOf(DrawIndexedIndirect),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
        });
        typedArgs = root.createBuffer(DrawIndexedIndirect, rawArgs).$usage("storage", "indirect");
        readback = mirror(typedArgs);

        resolvedWgsl = tgpu.resolve({
            names: "strict",
            template: /* wgsl */ `
                @group(0) @binding(0) var<storage, read_write> args: DrawIndexedIndirect;

                @compute @workgroup_size(1)
                fn main() {
                    args.indexCount = ${triangle.indexCount}u;
                    args.instanceCount = 1u;
                    args.firstIndex = ${triangle.indexBase}u;
                    args.baseVertex = 0i;
                    args.firstInstance = 0u;
                }
            `,
            externals: { DrawIndexedIndirect },
        });
        const module = device.createShaderModule({
            label: "no-walls-resolved-raw-wgsl",
            code: resolvedWgsl,
        });
        const pipeline = await device.createComputePipelineAsync({
            label: "no-walls-raw-pipeline",
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
        const group = device.createBindGroup({
            label: "no-walls-raw-group",
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: rawArgs } }],
        });

        const encoder = device.createCommandEncoder({ label: "no-walls-raw-encode" });
        const pass = encoder.beginComputePass({ label: "no-walls-raw-dispatch" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);

        Draws.register({
            name: "no-walls",
            surface: "no-walls",
            mesh: "no-walls-triangle",
            args: { indirect: typedArgs },
        });
    },
};

const NoWallsPlugin: Plugin = {
    name: "NoWalls",
    dependencies: [RenderPlugin, MirrorPlugin],
    systems: [RawProducerSystem],
    initialize(state) {
        mesh({ name: "no-walls-triangle", vertices, indices });
        registerSurface(state, {
            name: "no-walls",
            layout: noWallsLayout,
            fs: noWallsFs,
        });
    },
};

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("no-walls: no WebGPU adapter");
const requiredFeatures = [
    "indirect-first-instance",
    "bgra8unorm-storage",
    "rg11b10ufloat-renderable",
] as const satisfies readonly GPUFeatureName[];
for (const feature of requiredFeatures) {
    if (!adapter.features.has(feature)) throw new Error(`no-walls: missing ${feature}`);
}
const adoptedDevice = await adapter.requestDevice({
    requiredFeatures: [...requiredFeatures],
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
});

const scene = `<scene>
    <a camera sear transform="pos: 0 0 4" />
</scene>`;
const { state } = await run({
    plugins: [MirrorPlugin, NoWallsPlugin],
    scene,
    device: adoptedDevice,
});
const harness = installHarness(state);
harness.run = async () => {
    for (let i = 0; i < 60 && !readback.snapshot; i++) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
    const snapshot = readback.snapshot;
    const values = snapshot ? new Uint32Array(snapshot.bytes) : null;
    const adopted = Compute.device === adoptedDevice;
    const wrapped = Compute.root.unwrap(typedArgs) === rawArgs;
    const schema = typedArgs.dataType === DrawIndexedIndirect;
    const resolved = resolvedWgsl.includes("struct DrawIndexedIndirect");
    const dispatched = values?.[0] === 3 && values?.[1] === 1;
    const registered = Draws.get("no-walls")?.args.indirect === typedArgs;
    const checks = [
        { name: "external GPUDevice adopted", ok: adopted },
        { name: "raw GPUBuffer wrapped with schema", ok: wrapped && schema },
        { name: "raw WGSL resolved through TypeGPU", ok: resolved },
        {
            name: "resolved raw dispatch wrote the typed draw record",
            ok: dispatched,
            detail: values ? `indexCount=${values[0]} instanceCount=${values[1]}` : "no snapshot",
        },
        { name: "Draws retained the typed indirect handle", ok: registered },
    ];
    return { ok: checks.every((check) => check.ok), checks };
};
