import { beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    createApp,
    GlobalTransform,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { loadGltfScene } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { selectScene, Viewer } from "../src/app";
import { modelPath } from "../src/scenes";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
beforeAll(async () => {
    await syncAssets();
    await setupGlobals();
});
test("the truck's accepted placement derives its five mesh and material instances", async () => {
    const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
    let entities: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [
            MeshPlugin,
            TransformPlugin,
            {
                name: "Truck",
                async initialize(world) {
                    const viewer = world.resource(Viewer);
                    viewer.camera = world.create();
                    world.add(viewer.camera, Transform);
                    world.add(viewer.camera, Orbit);
                    viewer.ground = world.create();
                    world.add(viewer.ground, Transform);
                    for (const scene of ["CesiumMilkTruck", "BoxTextured"] as const) {
                        viewer.models.set(
                            scene,
                            await loadGltfScene(
                                world,
                                new URL(modelPath(scene).slice("assets/".length), assets),
                            ),
                        );
                    }
                    selectScene(world, "CesiumMilkTruck");
                    entities = viewer.entities;
                },
            },
        ],
    }).finally(() => fetch.mockRestore());
    try {
        console.log("glTF truck placement", app.world.gpu.adapter);
        app.world.tick();
        expect(entities).toHaveLength(5);
        const global = app.world.storage(GlobalTransform);
        expect(entities.map((eid) => global.translation.y.get(eid))).toEqual([
            expect.closeTo(-0.5),
            expect.closeTo(-0.5),
            expect.closeTo(-0.5),
            expect.closeTo(-0.2005946),
            expect.closeTo(-0.2005946),
        ]);
        for (const eid of entities) {
            expect(app.world.has(eid, MeshInstance)).toBe(true);
            expect(app.world.has(eid, MeshMaterial)).toBe(true);
            expect(global.scale.x.get(eid)).toBeCloseTo(0.7);
        }
        const old = entities.map((eid) => app.world.ref(eid));
        const viewer = app.world.resource(Viewer);
        selectScene(app.world, "BoxTextured", true);
        expect(old.map((ref) => app.world.resolve(ref))).toEqual([0, 0, 0, 0, 0]);
        expect(viewer.entities).toHaveLength(1);
        expect(app.world.storage(MeshInstance).mesh.get(viewer.entities[0])).toBe(
            viewer.models.get("BoxTextured")!.primitives[0].mesh,
        );
        expect(app.world.storage(Orbit).distance.get(viewer.camera)).toBeCloseTo(4.8);
        expect(app.world.storage(Transform).translation.y.get(viewer.ground)).toBeCloseTo(-0.51);
        selectScene(app.world, "CesiumMilkTruck");
        expect(viewer.entities).toHaveLength(5);
        expect(app.world.storage(Orbit).distance.get(viewer.camera)).toBe(6);
    } finally {
        app.dispose();
    }
});
