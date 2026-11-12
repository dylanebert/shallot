import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    type App,
    Camera,
    createApp,
    MeshPlugin,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import {
    BakedFrame,
    GltfPlugin,
    LeaderPose,
    loadGltfScene,
    SkinnedMesh,
} from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { animateScene, selectScene, Viewer } from "../src/app";
import { Crowd, crowdCount, leaderCount, setCrowdMode } from "../src/crowd";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
let app: App;
beforeAll(async () => {
    await syncAssets();
    await setupGlobals();
    const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
    try {
        app = await createApp({
            defaults: false,
            plugins: [
                MeshPlugin,
                TransformPlugin,
                {
                    name: "CrowdTest",
                    dependencies: [GltfPlugin],
                    async initialize(world) {
                        const viewer = world.resource(Viewer);
                        viewer.camera = world.create();
                        world.add(viewer.camera, Camera, { fov: 60 });
                        world.add(viewer.camera, Transform);
                        world.add(viewer.camera, Orbit);
                        const model = await loadGltfScene(
                            world,
                            new URL("Models/CesiumMan/glTF/CesiumMan.gltf", assets),
                        );
                        viewer.models.set("CesiumMan", model);
                        viewer.models.set("CesiumManCrowd", model);
                    },
                },
            ],
        });
    } finally {
        fetch.mockRestore();
    }
});
afterAll(() => app?.dispose());

test("crowd switches live, eight randomized leaders and nearest baked frames, then releases on selection", async () => {
    const world = app.world;
    await selectScene(world, "CesiumManCrowd", 390 / 844);
    const crowd = world.resource(Crowd);
    expect(crowd.instances).toHaveLength(crowdCount);
    expect(new Set(crowd.offsets.slice(0, leaderCount)).size).toBe(leaderCount);
    for (const offset of crowd.offsets) {
        expect(offset).toBeGreaterThanOrEqual(0);
        expect(offset).toBeLessThan(crowd.duration);
    }
    const bounds = world.resource(Viewer).bounds!;
    for (const aspect of [1280 / 720, 1512 / 982, 390 / 844]) {
        await selectScene(world, "CesiumManCrowd", aspect);
        const camera = world.resource(Viewer).camera;
        const orbit = world.storage(Orbit);
        const center = bounds.min.map((value, axis) => (value + bounds.max[axis]) / 2);
        expect([orbit.pan.x.get(camera), orbit.pan.y.get(camera), orbit.pan.z.get(camera)]).toEqual(
            center.map(Math.fround),
        );
        const radius = Math.hypot(...bounds.max.map((value, axis) => value - center[axis]));
        const distance = orbit.distance.get(camera);
        const tangent = Math.tan((world.storage(Camera).fov.get(camera) * Math.PI) / 360);
        expect(
            radius / Math.sqrt(distance ** 2 - radius ** 2) / (tangent * Math.min(1, aspect)),
        ).toBeCloseTo(1 / 1.3, 5);
    }
    for (const instance of crowd.instances) {
        const eid = instance.entities[0];
        const transform = world.storage(Transform);
        expect(transform.translation.x.get(eid)).toBeGreaterThanOrEqual(bounds.min[0]);
        expect(transform.translation.x.get(eid)).toBeLessThanOrEqual(bounds.max[0]);
        expect(transform.translation.z.get(eid)).toBeGreaterThanOrEqual(bounds.min[2]);
        expect(transform.translation.z.get(eid)).toBeLessThanOrEqual(bounds.max[2]);
    }
    for (const mode of ["live", "shared", "baked", "live"] as const) {
        setCrowdMode(world, mode);
        animateScene(world, 1 / 60);
        world.tick();
        expect(crowd.mode).toBe(mode);
        expect(crowd.instances).toHaveLength(crowdCount);
        expect(crowd.costs[mode]).toBeGreaterThanOrEqual(0);
        expect(crowd.bake !== null).toBe(mode === "baked");
        for (let i = 0; i < crowdCount; i++) {
            const eid = crowd.instances[i].entities[0];
            const follower = mode === "shared" && i >= leaderCount;
            expect(world.has(eid, LeaderPose)).toBe(follower);
            expect(world.has(eid, SkinnedMesh)).toBe(mode !== "baked" && !follower);
            expect(world.has(eid, BakedFrame)).toBe(mode === "baked");
            if (follower)
                expect(world.storage(LeaderPose).leader.get(eid)).toBe(
                    crowd.instances[i % leaderCount].entities[0],
                );
            if (crowd.bake)
                expect(world.storage(BakedFrame).frame.get(eid)).toBe(
                    Math.round(
                        ((crowd.time + crowd.offsets[i]) % crowd.duration) * crowd.bake.rate,
                    ),
                );
        }
    }
    setCrowdMode(world, "baked");
    const entities = crowd.instances.flatMap((instance) =>
        instance.entities.map((eid) => world.ref(eid)),
    );
    await selectScene(world, "CesiumMan");
    expect(crowd.instances).toHaveLength(0);
    expect(crowd.bake).toBeNull();
    for (const ref of entities) expect(world.resolve(ref)).toBe(0);
});
