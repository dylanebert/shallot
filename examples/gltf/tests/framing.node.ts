import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    type App,
    Camera,
    composeMat4,
    createApp,
    GlobalTransform,
    MeshPlugin,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { loadGltf, loadGltfScene } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { selectScene, Viewer } from "../src/app";
import { modelPath, type Scene, scenes } from "../src/scenes";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
const frameShare = 1 / 1.3;
let app: App;
const decoded = new Map<Scene, Awaited<ReturnType<typeof loadGltf>>>();
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
                    name: "Framing",
                    async initialize(world) {
                        const viewer = world.resource(Viewer);
                        viewer.camera = world.create();
                        world.add(viewer.camera, Camera, { fov: 60 });
                        world.add(viewer.camera, Transform);
                        world.add(viewer.camera, Orbit);
                        for (const scene of Object.keys(scenes) as Scene[]) {
                            const url = new URL(modelPath(scene).slice("assets/".length), assets);
                            decoded.set(scene, await loadGltf(url, fileFetch));
                            viewer.models.set(scene, await loadGltfScene(world, url));
                        }
                    },
                },
            ],
        });
        console.log("glTF framing", app.world.gpu.adapter);
    } finally {
        fetch.mockRestore();
    }
});
afterAll(() => app?.dispose());

for (const [width, height] of [
    [1280, 720],
    [1512, 982],
    [390, 844],
]) {
    for (const scene of Object.keys(scenes) as Scene[]) {
        test(`${scene} at ${width}x${height}: centered bounds fit with margin and sphere fills limiting axis`, () => {
            const world = app.world;
            selectScene(world, scene, width / height);
            world.tick();
            const viewer = world.resource(Viewer);
            const global = world.storage(GlobalTransform);
            const points: number[][] = [];
            // Observe the spawned placements, independently of the importer's bounds.
            for (const [index, instance] of decoded.get(scene)!.instances.entries()) {
                const eid = viewer.entities[index];
                const matrix = composeMat4(
                    global.translation.x.get(eid),
                    global.translation.y.get(eid),
                    global.translation.z.get(eid),
                    global.rotation.x.get(eid),
                    global.rotation.y.get(eid),
                    global.rotation.z.get(eid),
                    global.rotation.w.get(eid),
                    global.scale.x.get(eid),
                    global.scale.y.get(eid),
                    global.scale.z.get(eid),
                );
                const vertices = decoded.get(scene)!.meshes[instance.mesh].vertices;
                for (let i = 0; i < vertices.length; i += 8) {
                    points.push(
                        [0, 1, 2].map(
                            (axis) =>
                                matrix[axis] * vertices[i] +
                                matrix[axis + 4] * vertices[i + 1] +
                                matrix[axis + 8] * vertices[i + 2] +
                                matrix[axis + 12],
                        ),
                    );
                }
            }
            const min = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis])));
            const max = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis])));
            const center = min.map((value, axis) => (value + max[axis]) / 2);
            const radius = Math.hypot(...max.map((value, axis) => value - center[axis]));
            const orbit = world.storage(Orbit);
            const pan = [
                orbit.pan.x.get(viewer.camera),
                orbit.pan.y.get(viewer.camera),
                orbit.pan.z.get(viewer.camera),
            ];
            for (let axis = 0; axis < 3; axis++) expect(pan[axis]).toBeCloseTo(center[axis], 4);
            const distance = orbit.distance.get(viewer.camera);
            const yaw = orbit.yaw.get(viewer.camera),
                pitch = orbit.pitch.get(viewer.camera);
            const cy = Math.cos(yaw),
                sy = Math.sin(yaw),
                cp = Math.cos(pitch),
                sp = Math.sin(pitch);
            const tangent = Math.tan(
                (world.storage(Camera).fov.get(viewer.camera) * Math.PI) / 360,
            );
            const dot = (a: number[], b: number[]) =>
                a.reduce((sum, value, axis) => sum + value * b[axis], 0);
            for (const x of [min[0], max[0]])
                for (const y of [min[1], max[1]])
                    for (const z of [min[2], max[2]]) {
                        const relative = [x - pan[0], y - pan[1], z - pan[2]];
                        const depth = distance - dot(relative, [sy * cp, sp, cy * cp]);
                        expect(depth).toBeGreaterThan(
                            world.storage(Camera).near.get(viewer.camera),
                        );
                        expect(depth).toBeLessThan(world.storage(Camera).far.get(viewer.camera));
                        expect(
                            Math.abs(
                                dot(relative, [cy, 0, -sy]) / ((depth * tangent * width) / height),
                            ),
                        ).toBeLessThanOrEqual(frameShare + 1e-5);
                        expect(
                            Math.abs(dot(relative, [-sy * sp, cp, -cy * sp]) / (depth * tangent)),
                        ).toBeLessThanOrEqual(frameShare + 1e-5);
                    }
            // Exact perspective silhouette of the enclosing sphere, not a small-angle approximation.
            const coverage =
                radius /
                Math.sqrt(distance ** 2 - radius ** 2) /
                (tangent * Math.min(1, width / height));
            expect(coverage).toBeCloseTo(frameShare, 5);
        });
    }
}
