import { beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    Camera,
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
import { frameScene, selectScene, Viewer } from "../src/app";
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
                    world.add(viewer.camera, Camera, { fov: 60 });
                    world.add(viewer.camera, Transform);
                    world.add(viewer.camera, Orbit);
                    for (const scene of ["CesiumMilkTruck", "TextureCoordinateTest"] as const) {
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
        const truckDistance = app.world.storage(Orbit).distance.get(viewer.camera);
        selectScene(app.world, "TextureCoordinateTest", 390 / 844);
        expect(old.map((ref) => app.world.resolve(ref))).toEqual([0, 0, 0, 0, 0]);
        expect(viewer.entities).toHaveLength(5);
        expect(app.world.storage(MeshInstance).mesh.get(viewer.entities[0])).toBe(
            viewer.models.get("TextureCoordinateTest")!.primitives[0].mesh,
        );
        const phoneDistance = app.world.storage(Orbit).distance.get(viewer.camera);
        frameScene(app.world, 1280 / 720);
        expect(app.world.storage(Orbit).distance.get(viewer.camera)).toBeLessThan(phoneDistance);
        selectScene(app.world, "CesiumMilkTruck");
        expect(viewer.entities).toHaveLength(5);
        expect(app.world.storage(Orbit).distance.get(viewer.camera)).toBe(truckDistance);
    } finally {
        app.dispose();
    }
});
