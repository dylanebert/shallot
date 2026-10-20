import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    GlobalAmbientLight,
    PointLight,
    RenderContext,
    RenderLayers,
    SpotLight,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { StandardRenderer, StandardRenderingPlugin } from "./forward";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { EntityMeshInstanceInput, MeshInstanceInput } from "./material-data";
import { AlphaMode } from "./material-type";
import { MeshRenderPlugin } from "./mesh-render";
import { MeshDrawBuffers } from "./preprocess";
import { Draws } from "./registry";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] },
]);

test("layer-1 meshes draw exactly once only in their camera; two-layer meshes draw in both; edits move them next frame", async () => {
    const { world } = subjects()[0];
    const cameras = [0, 1].map((layer) => {
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 0, 5, 0] });
        world.add(eid, Camera, { antialias: 0 });
        world.add(eid, StandardRenderer);
        if (layer) world.add(eid, RenderLayers, { mask: 1 << layer });
        attachTexture(world, eid, { width: 32, height: 32 });
        return eid;
    });
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(mesh, RenderLayers, { mask: 2 });
    world.add(mesh, MeshMaterial, world.resource(Materials).add(StandardMaterial({ unlit: true })));
    const counts = async () => {
        world.step(0);
        const draw = [...world.resource(Draws).values()][0]!;
        const bytes = (
            await probeBuffer(
                world,
                world.gpu.root.unwrap(world.resource(MeshDrawBuffers).drawArgs!),
            )
        ).bytes;
        const words = new Uint32Array(bytes);
        return cameras.map(
            (eid) =>
                words[
                    (draw.args.offset! +
                        world.resource(Views).get(eid)!.slot * draw.args.viewStride!) /
                        4 +
                        1
                ],
        );
    };
    const center = async (camera: number) => {
        const { rgba } = await captureTexture(world, camera);
        return [...rgba.subarray((16 * 32 + 16) * 4, (16 * 32 + 16) * 4 + 3)];
    };
    try {
        expect(d.sizeOf(EntityMeshInstanceInput)).toBe(d.sizeOf(MeshInstanceInput));
        expect(d.memoryLayoutOf(EntityMeshInstanceInput, (row) => row.cullBounds).offset).toBe(
            d.memoryLayoutOf(MeshInstanceInput, (row) => row.cullBounds).offset,
        );
        expect(await counts()).toEqual([0, 1]);
        const hidden = await center(cameras[0]!);
        const visible = await center(cameras[1]!);
        expect(visible).not.toEqual(hidden);
        world.storage(RenderLayers).mask.set(mesh, 3);
        expect(await counts()).toEqual([1, 1]);
        expect(await center(cameras[0]!)).toEqual(visible);
        world.storage(RenderLayers).mask.set(mesh, 1);
        expect(await counts()).toEqual([1, 0]);
        expect(await center(cameras[1]!)).toEqual(hidden);
        world.storage(RenderLayers).mask.set(mesh, 0);
        expect(await counts()).toEqual([0, 0]);
        world.remove(mesh, RenderLayers);
        expect(await counts()).toEqual([1, 0]);
        world.add(mesh, RenderLayers, { mask: 0x80000000 });
        world.storage(RenderLayers).mask.set(cameras[1]!, 0x80000000);
        expect(await counts()).toEqual([0, 1]);
    } finally {
        world.destroy(mesh);
        for (const camera of cameras) world.destroy(camera);
    }
});

test("blended meshes select layers before shared transparent sorting", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(mesh, RenderLayers, { mask: 2 });
    world.add(
        mesh,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ unlit: true, baseColor: [1, 1, 1, 0.5] }), {
                alphaMode: AlphaMode.Blend,
            }),
    );
    const frame = async () => {
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    try {
        const hidden = await frame();
        world.add(camera, RenderLayers, { mask: 2 });
        const visible = await frame();
        expect(visible).not.toEqual(hidden);
        world.storage(RenderLayers).mask.set(mesh, 1);
        expect(await frame()).toEqual(hidden);
        world.storage(RenderLayers).mask.set(mesh, 3);
        expect(await frame()).toEqual(visible);
        world.storage(RenderLayers).mask.set(camera, 0);
        expect(await frame()).toEqual(hidden);
    } finally {
        world.destroy(mesh);
        world.destroy(camera);
    }
});

for (const [name, Light] of [
    ["directional", DirectionalLight],
    ["point", PointLight],
    ["spot", SpotLight],
] as const) {
    test(`${name} layer 0 leaves a layer-1 mesh unlit; a two-layer light lights both views`, async () => {
        const { world } = subjects()[0];
        world.resource(GlobalAmbientLight).brightness = 0;
        const cameras = [1, 2].map((mask) => {
            const eid = world.create();
            world.add(eid, Transform, { translation: [0, 0, 5, 0] });
            world.add(eid, Camera, { antialias: 0 });
            world.add(eid, StandardRenderer);
            world.add(eid, RenderLayers, { mask });
            attachTexture(world, eid, { width: 32, height: 32 });
            return eid;
        });
        const mesh = world.create();
        world.add(mesh, Transform);
        world.add(mesh, MeshInstance);
        world.add(mesh, RenderLayers, { mask: 3 });
        world.add(
            mesh,
            MeshMaterial,
            world.resource(Materials).add(StandardMaterial({ perceptualRoughness: 1 })),
        );
        const light = world.create();
        world.add(light, Transform, { translation: [0, 0, 3, 0] });
        world.add(light, Light);
        if (Light === DirectionalLight)
            world.storage(DirectionalLight).illuminance.set(light, 100000);
        else if (Light === PointLight) world.storage(PointLight).intensity.set(light, 500000);
        else world.storage(SpotLight).intensity.set(light, 500000);
        const brightness = async () => {
            world.step(0);
            const result: number[] = [];
            for (const camera of cameras) {
                const { rgba } = await captureTexture(world, camera);
                result.push(rgba[(16 * 32 + 16) * 4]!);
            }
            return result;
        };
        try {
            const baseline = await brightness();
            expect(baseline[0]!).toBeGreaterThan(baseline[1]! + 20);
            world.add(light, RenderLayers, { mask: 3 });
            const shared = await brightness();
            expect(shared[0]).toBe(baseline[0]);
            expect(shared[1]).toBe(shared[0]);
            world.storage(RenderLayers).mask.set(light, 0);
            expect(await brightness()).toEqual([baseline[1]!, baseline[1]!]);
            world.remove(light, RenderLayers);
            expect(await brightness()).toEqual(baseline);
        } finally {
            world.destroy(light);
            world.destroy(mesh);
            for (const camera of cameras) world.destroy(camera);
        }
    });
}

test("a layer-1 caster submits no shadow instances for layer-0 lights; shared light layers restore casting", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, RenderLayers, { mask: 3 });
    attachTexture(world, camera, { width: 16, height: 16 });
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(mesh, RenderLayers, { mask: 2 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    const point = world.create();
    world.add(point, Transform, { translation: [2, 2, 3, 0] });
    world.add(point, PointLight, { shadowMapsEnabled: 1, range: 10 });
    const shadowCounts = async () => {
        world.step(0);
        const render = world.resource(RenderContext);
        expect(render.viewCount).toBeGreaterThan(render.shadeCount);
        const draw = [...world.resource(Draws).values()][0]!;
        const words = new Uint32Array(
            (
                await probeBuffer(
                    world,
                    world.gpu.root.unwrap(world.resource(MeshDrawBuffers).drawArgs!),
                )
            ).bytes,
        );
        return Array.from(
            { length: render.viewCount - render.shadeCount },
            (_, i) =>
                words[
                    (draw.args.offset! + (i + render.shadeCount) * draw.args.viewStride!) / 4 + 1
                ]!,
        );
    };
    try {
        const hidden = await shadowCounts();
        expect(hidden.every((count) => count === 0)).toBe(true);
        world.add(sun, RenderLayers, { mask: 3 });
        world.add(point, RenderLayers, { mask: 3 });
        const shared = await shadowCounts();
        const cascades = world.storage(DirectionalLight).numCascades.get(sun);
        expect(shared.slice(0, cascades).some((count) => count === 1)).toBe(true);
        expect(shared.slice(cascades).some((count) => count === 1)).toBe(true);
        world.storage(RenderLayers).mask.set(sun, 1);
        world.storage(RenderLayers).mask.set(point, 1);
        expect(await shadowCounts()).toEqual(hidden);
    } finally {
        for (const eid of [camera, mesh, sun, point]) world.destroy(eid);
    }
});
