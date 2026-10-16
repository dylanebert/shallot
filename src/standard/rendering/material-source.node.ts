import { expect, setDefaultTimeout, test } from "bun:test";
import {
    ClearChangeMarksSystem,
    component,
    type Plugin,
    probeBuffer,
    vec2,
    vec4,
    type World,
} from "@dylanebert/shallot";
import { MeshInstance } from "@dylanebert/shallot/mesh";
import { attachTexture, Camera, captureTexture } from "@dylanebert/shallot/rendering";
import {
    AlphaModeCode,
    AlphaPipelineKey,
    MaterialAssets,
    MaterialPlugin,
    MeshInstanceInput,
    MeshMaterial,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    materialType,
    materialTypeId,
    meshInstanceTable,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import tgpu, { readFromArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const Surface = component(
    "MaterialSourceProof",
    { color: vec4, alphaMode: vec2 },
    { defaults: () => ({ color: [1, 0, 0, 0.3], alphaMode: [AlphaModeCode.Mask, 0.5] }) },
);
const Parameters = d.struct({ color: d.vec4f, alphaMode: d.vec2f });
const layout = tgpu.bindGroupLayout({
    parameters: { storage: d.arrayOf(Parameters), access: "readonly" },
    rows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    instances: { storage: d.arrayOf(MeshInstanceInput), access: "mutable" },
    params: { uniform: d.vec2u },
});
const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    (args) => {
        "use gpu";
        if (args.gid.x >= layout.$.params.x) return;
        const row = layout.$.rows[args.gid.x].y;
        if (layout.$.instances[row].materialType !== layout.$.params.y) return;
        const alpha = layout.$.parameters[layout.$.instances[row].material].alphaMode;
        layout.$.instances[row].alphaMode = d.u32(alpha.x);
        layout.$.instances[row].alphaCutoff = alpha.y;
    },
);
const Source = {
    create(world: World) {
        const table = world.table("material-source-proof", Parameters);
        table.bindComponent(Surface, { color: "color", alphaMode: "alphaMode" });
        const pipeline = world.gpu.root.createComputePipeline({ compute: kernel });
        const params = world.gpu.root.createBuffer(d.vec2u).$usage("uniform");
        world.own(world.gpu.root.unwrap(params));
        let view: DataView | undefined;
        return {
            table,
            warm() {
                world.gpu.root.unwrap(pipeline);
            },
            alphaKeys: (1 << AlphaPipelineKey.Mask) | (1 << AlphaPipelineKey.Blend),
            alphaMode(row: number) {
                if (view?.buffer !== table.bytes.buffer) view = new DataView(table.bytes.buffer);
                return view.getFloat32(row * table.rowBytes + 16, true);
            },
            publish() {
                const instances = meshInstanceTable(world);
                if (!instances.count) return;
                params.write(d.vec2u(instances.count, materialTypeId(world, SurfaceMaterial)));
                const group = world.gpu.root.createBindGroup(layout, {
                    parameters: table.typed,
                    rows: instances.activeRowsTyped!,
                    instances: instances.typed,
                    params,
                } as never);
                const pass = world.frameEncoder()!.beginComputePass();
                pass.setPipeline(world.gpu.root.unwrap(pipeline));
                pass.setBindGroup(0, world.gpu.root.unwrap(group));
                pass.dispatchWorkgroups(Math.ceil(instances.count / 64));
                pass.end();
            },
        };
    },
};
const SurfaceMaterial = materialType({
    name: "MaterialSourceProof",
    parameters: Parameters,
    source: Source,
    defaults: { color: d.vec4f(1), alphaMode: d.vec2f(1, 0.5) },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
});
const Proof: Plugin = {
    name: "MaterialSourceProof",
    components: [Surface],
    dependencies: [MeshRenderPlugin, StandardRenderingPlugin, MaterialPlugin(SurfaceMaterial)],
    warm(world) {
        world.resource(Source).warm();
    },
    systems: [
        {
            name: "material-source-proof-publish",
            group: "draw",
            after: [ClearChangeMarksSystem],
            before: [MeshPreprocessSystem],
            update(world) {
                world.resource(Source).publish();
            },
        },
    ],
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [Proof] }]);

test("a material borrows externally owned rows and alpha inputs without taking over their allocation", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    const source = world.resource(Source);
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Surface);
    const row = source.table.rowIndex(eid);
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, { type: materialTypeId(world, SurfaceMaterial), material: row });
    const shared = world.create();
    world.add(shared, Transform, { translation: [1.4, 0, 0] });
    world.add(shared, MeshInstance);
    world.add(shared, MeshMaterial, {
        type: materialTypeId(world, SurfaceMaterial),
        material: row,
    });
    world.step(0);
    const masked = await captureTexture(world, camera);
    world.storage(Surface).alphaMode.y.set(eid, 0.1);
    world.step(0);
    const visible = await captureTexture(world, camera);
    expect(visible.rgba).not.toEqual(masked.rgba);
    world.storage(Surface).alphaMode.x.set(eid, AlphaModeCode.Blend);
    world.step(0);
    const blended = await captureTexture(world, camera);
    expect(blended.rgba).not.toEqual(visible.rgba);
    expect(blended.rgba).not.toEqual(masked.rgba);
    const sharedPixel = (16 * 32 + 25) * 4;
    expect(blended.rgba[sharedPixel]).toBeGreaterThan(blended.rgba[sharedPixel + 1]!);
    const snapshot = await probeBuffer(world, source.table.buffer, {
        offset: row * d.sizeOf(Parameters),
        size: d.sizeOf(Parameters),
    });
    expect(readFromArrayBuffer(snapshot.bytes, Parameters).alphaMode.x).toBe(AlphaModeCode.Blend);
    expect(world.storage(MeshMaterial).alphaMode.get(eid)).toBe(0);
    const refusal =
        'MaterialAssets "MaterialSourceProof" cannot allocate assets: the material type uses an external parameter source';
    expect(() => world.resource(SurfaceMaterial)).toThrow(refusal);
    expect(() => new MaterialAssets(world, SurfaceMaterial)).toThrow(refusal);
    world.step(0);
    expect((await captureTexture(world, camera)).rgba).toEqual(blended.rgba);
    world.destroy(shared);
    world.destroy(eid);
    const replacement = world.create();
    world.add(replacement, Surface);
    expect(source.table.rowIndex(replacement)).toBe(row);
    world.remove(replacement, Surface);
    expect(source.table.rowIndex(replacement)).toBe(-1);
});
