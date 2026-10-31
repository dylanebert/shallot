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
import { modelPath, type Scene, scenes, truckPlacement } from "./scenes";

export const Viewer: Resource<{
    models: Map<Scene, Gltf>;
    entities: number[];
    camera: number;
}> = {
    create: () => ({
        models: new Map(),
        entities: [],
        camera: 0,
    }),
};
export function selectScene(world: World, scene: Scene, phone = false) {
    const viewer = world.resource(Viewer);
    for (const eid of viewer.entities) world.destroy(eid);
    viewer.entities = spawnGltfScene(
        world,
        viewer.models.get(scene)!,
        scene === "CesiumMilkTruck" ? truckPlacement : {},
    );
    const view = scenes[scene];
    const orbit = world.storage(Orbit);
    orbit.distance.set(viewer.camera, view.distance * (phone ? 1.6 : 1));
    orbit.yaw.set(viewer.camera, view.yaw);
    orbit.pitch.set(viewer.camera, view.pitch);
    orbit.pan.set(viewer.camera, view.target[0], view.target[1], view.target[2]);
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
        selectScene(world, "CesiumMilkTruck", matchMedia("(max-width: 480px)").matches);
    },
};
