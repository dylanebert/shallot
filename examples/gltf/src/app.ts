import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    MeshPlugin,
    type Plugin,
    type Resource,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { type Gltf, loadGltfScene, spawnGltfScene } from "@dylanebert/shallot/gltf";
import { fitBounds, sceneAngles, sceneBounds } from "./framing";
import { modelPath, type Scene, scenes, truckPlacement } from "./scenes";

export const Viewer: Resource<{
    models: Map<Scene, Gltf>;
    entities: number[];
    camera: number;
    bounds: Gltf["bounds"] | null;
}> = {
    create: () => ({
        models: new Map(),
        entities: [],
        camera: 0,
        bounds: null,
    }),
};
export function frameScene(world: World, aspect: number) {
    const viewer = world.resource(Viewer);
    if (viewer.bounds) fitBounds(world, viewer.camera, viewer.bounds, aspect);
}

export function selectScene(world: World, scene: Scene, aspect = 1) {
    const viewer = world.resource(Viewer);
    for (const eid of viewer.entities) world.destroy(eid);
    viewer.entities = spawnGltfScene(
        world,
        viewer.models.get(scene)!,
        scene === "CesiumMilkTruck" ? truckPlacement : {},
    );
    viewer.bounds = sceneBounds(viewer.models.get(scene)!, scene);
    const view = sceneAngles(scene);
    const orbit = world.storage(Orbit);
    orbit.yaw.set(viewer.camera, view.yaw);
    orbit.pitch.set(viewer.camera, view.pitch);
    frameScene(world, aspect);
}
export const AppPlugin: Plugin = {
    name: "GltfApp",
    dependencies: [OrbitPlugin, MeshPlugin, DiagnosticsOverlayPlugin],
    async initialize(world, progress) {
        const viewer = world.resource(Viewer);
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = 380;
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000 });
        world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
        viewer.camera = world.create();
        world.add(viewer.camera, Camera, { clearColor: 0x191816 });
        world.add(viewer.camera, StandardRenderer);
        world.add(viewer.camera, Transform);
        world.add(viewer.camera, Orbit);
        const names = Object.keys(scenes) as Scene[];
        let loaded = 0;
        await Promise.all(
            names.map(async (scene) => {
                viewer.models.set(
                    scene,
                    await loadGltfScene(world, new URL(modelPath(scene), document.baseURI)),
                );
                progress?.(++loaded / names.length);
            }),
        );
        const canvas = document.querySelector("canvas")!;
        selectScene(world, "CesiumMilkTruck", canvas.clientWidth / canvas.clientHeight);
    },
};
