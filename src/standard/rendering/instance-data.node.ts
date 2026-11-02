import { expect, setDefaultTimeout, test } from "bun:test";
import { component, probeBuffer, vec4 } from "@dylanebert/shallot";
import { Meshes, MeshInstance } from "@dylanebert/shallot/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import {
    applyPbrLighting,
    extendedMaterial,
    extensionLayout,
    instanceData,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    pbrInputFromStandardMaterial,
    registerProducerTable,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import * as d from "typegpu/data";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshDrawBuffers } from "./preprocess";

setDefaultTimeout(TIMEOUT.node);
const Tint = component("InstanceTint", { color: vec4 });
const Dim = component("InstanceDim", {});
const ColorFields = d.struct({ color: d.vec4f });
const Colors = instanceData(Tint, ColorFields);
const Dimming = instanceData(Dim);
const Varied = extendedMaterial(Materials, {
    name: "ComponentInstanceMaterial",
    instanceData: [Colors, Dimming],
    fragment: (ctx) => {
        "use gpu";
        const input = pbrInputFromStandardMaterial(ctx);
        input.baseColor = Colors.read(ctx, ColorFields({ color: input.baseColor })).color;
        if (Dimming.present(ctx))
            input.baseColor = d.vec4f(
                input.baseColor.x * 0.5,
                input.baseColor.y * 0.5,
                input.baseColor.z * 0.5,
                input.baseColor.w * 0.5,
            );
        return applyPbrLighting(input);
    },
});
const Own = materialLayout(
    ColorFields,
    { factor: { type: "uniform", struct: ColorFields } },
    { instanceData: [Colors] },
);
const ownFactor = {
    create: (world: import("@dylanebert/shallot").World) =>
        world.gpu.root
            .createBuffer(ColorFields, { color: d.vec4f(0.5, 1, 1, 1) })
            .$usage("uniform"),
};
const Custom = materialType({
    name: "CustomComponentInstanceMaterial",
    layout: Own,
    defaults: { color: d.vec4f(1) },
    bindings: (world) => ({ factor: world.resource(ownFactor) }),
    fragment: (ctx) => {
        "use gpu";
        const color = Colors.read(ctx, ctx.parameters).color;
        return d.vec4f(color.x * Own.$.factor.color.x, color.y, color.z, color.w);
    },
});
const fixture = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(Varied),
            MaterialPlugin(Custom),
            { name: "InstanceComponents", components: [Tint, Dim] },
        ],
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 8, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, DepthPrepass);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 128, height: 64 });
        const sun = world.create();
        world.add(sun, Transform);
        world.add(sun, DirectionalLight, { shadowMapsEnabled: 0 });
        const base = StandardMaterial({
            baseColor: d.vec4f(0.7, 0.3, 0.2, 1),
            perceptualRoughness: 1,
        });
        const handle = world.resource(Varied).add(base);
        const entities = Array.from({ length: 4 }, (_, i) => {
            const eid = world.create();
            world.add(eid, Transform, { translation: [(i - 1.5) * 1.6, 0, 0, 0] });
            world.add(eid, MeshInstance);
            world.add(eid, MeshMaterial, handle);
            return eid;
        });
        const tint = d.vec4f(0.1, 0.8, 0.4, 1);
        world.add(entities[1]!, Tint, { color: [tint.x, tint.y, tint.z, tint.w] });
        world.add(entities[2]!, Dim);
        world.add(entities[3]!, Tint, { color: [tint.x, tint.y, tint.z, tint.w] });
        world.add(entities[3]!, Dim);
        const references = [
            base.baseColor,
            tint,
            d.vec4f(0.35, 0.15, 0.1, 0.5),
            d.vec4f(0.05, 0.4, 0.2, 0.5),
        ].map((baseColor) => world.resource(Materials).add({ ...base, baseColor }));
        const mesh = [...world.resource(Meshes).entries()][0]![0];
        const producer = registerProducerTable(world, {
            name: "instance-absence",
            capacity: 1,
            meshes: [mesh],
            materials: [handle, references[0]!],
        });
        producer.placements.write([
            { pos: d.vec3f(0, 1.8, 0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(0.7) },
        ]);
        const producerRow = {
            mesh,
            materialType: handle.type,
            material: handle.material,
            alphaMode: 0,
            alphaCutoff: 0.5,
            flags: 0,
            cullMode: 0,
            cullBounds: d.vec4f(0, 0, 0, -1),
        };
        producer.meshInstances.write([producerRow]);
        producer.count.write(0);
        return { camera, entities, handle, references, producer, producerRow };
    },
);

test("two component declarations share a lit draw, use each entity's values, and removal falls back next frame", async () => {
    const { world, scene } = fixture();
    world.step(0);
    const varied = (await captureTexture(world, scene.camera)).rgba;
    const buffers = world.resource(MeshDrawBuffers);
    const args = new Uint32Array(
        (await probeBuffer(world, world.gpu.root.unwrap(buffers.drawArgs!))).bytes,
    );
    const populated = [];
    for (let i = 0; i < args.length; i += 5) if (args[i + 1]) populated.push(args[i + 1]);
    expect(populated).toEqual([4]);
    for (let i = 0; i < scene.entities.length; i++) {
        world.remove(scene.entities[i]!, MeshMaterial);
        world.add(scene.entities[i]!, MeshMaterial, scene.references[i]!);
    }
    world.step(0);
    const reference = (await captureTexture(world, scene.camera)).rgba;
    expect(varied).toEqual(reference);
    expect(new Set(varied).size).toBeGreaterThan(4);
    for (const eid of scene.entities) {
        world.remove(eid, MeshMaterial);
        world.add(eid, MeshMaterial, scene.handle);
    }
    world.remove(scene.entities[1]!, Tint);
    world.step(0);
    const removed = (await captureTexture(world, scene.camera)).rgba;
    expect(removed).not.toEqual(varied);
    world.add(scene.entities[1]!, Tint, { color: [0.7, 0.3, 0.2, 1] });
    world.step(0);
    expect((await captureTexture(world, scene.camera)).rgba).toEqual(removed);
    // Producer row zero must not read entity row zero's fields or presence.
    world.add(scene.entities[0]!, Tint, { color: [0, 0, 1, 1] });
    world.add(scene.entities[0]!, Dim);
    scene.producer.count.write(1);
    world.step(0);
    const produced = (await captureTexture(world, scene.camera)).rgba;
    scene.producer.count.write(0);
    world.step(0);
    expect((await captureTexture(world, scene.camera)).rgba).not.toEqual(produced);
    scene.producer.count.write(1);
    scene.producer.meshInstances.write([
        {
            ...scene.producerRow,
            materialType: scene.references[0]!.type,
            material: scene.references[0]!.material,
        },
    ]);
    world.step(0);
    expect((await captureTexture(world, scene.camera)).rgba).toEqual(produced);
});

test("an explicit materialLayout carries instance data beside the resources its shader closes over", async () => {
    const { world, scene } = fixture();
    const custom = world.resource(Custom).add({ color: d.vec4f(1) });
    for (const eid of scene.entities) {
        world.remove(eid, MeshMaterial);
        world.add(eid, MeshMaterial, custom);
    }
    world.step(0);
    const pixels = (await captureTexture(world, scene.camera)).rgba;
    for (const eid of scene.entities) {
        const color = world.has(eid, Tint) ? world.get(eid, Tint)!.color : [1, 1, 1, 1];
        const reference = world.resource(Materials).add(
            StandardMaterial({
                unlit: true,
                baseColor: d.vec4f(color[0]! * 0.5, color[1]!, color[2]!, color[3]!),
            }),
        );
        world.remove(eid, MeshMaterial);
        world.add(eid, MeshMaterial, reference);
    }
    world.step(0);
    expect((await captureTexture(world, scene.camera)).rgba).toEqual(pixels);
});

test("explicit layouts refuse spec-level instance data and name materialLayout as its place", () => {
    expect(() =>
        materialType({
            name: "WrongInstanceBindingPlace",
            layout: Own,
            instanceData: [Colors],
            defaults: { color: d.vec4f(1) },
            fragment: () => {
                "use gpu";
                return d.vec4f(1);
            },
        }),
    ).toThrow(/instanceData in materialLayout/);
    expect(() =>
        extendedMaterial(Materials, {
            name: "WrongExtensionBindingPlace",
            layout: extensionLayout(Materials, ColorFields, {}),
            instanceData: [Colors],
            fragment: () => {
                "use gpu";
                return d.vec4f(1);
            },
        }),
    ).toThrow(/instanceData in materialLayout/);
});
