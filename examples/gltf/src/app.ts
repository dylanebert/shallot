import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    type Plugin,
    type Resource,
    StandardMaterial,
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
    ground: number;
}> = {
    create: () => ({
        models: new Map(),
        entities: [],
        camera: 0,
        ground: 0,
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
    const transform = world.storage(Transform);
    transform.translation.set(viewer.ground, 0, view.ground, 0);
    if (scene === "CesiumMilkTruck") transform.scale.set(viewer.ground, 4, 0.2, 4);
    else transform.scale.set(viewer.ground, view.distance, 0.02, view.distance);
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
        viewer.ground = world.create();
        world.add(viewer.ground, MeshInstance);
        world.add(
            viewer.ground,
            MeshMaterial,
            world.resource(Materials).add(
                StandardMaterial({
                    baseColor: [0.085, 0.08, 0.072, 1],
                    perceptualRoughness: 0.82,
                }),
            ),
        );
        world.add(viewer.ground, Transform);
        const names = Object.keys(scenes) as Scene[];
        for (const [i, scene] of names.entries()) {
            viewer.models.set(
                scene,
                await loadGltfScene(world, new URL(modelPath(scene), document.baseURI)),
            );
            progress?.((i + 1) / names.length);
        }
        selectScene(world, "CesiumMilkTruck", matchMedia("(max-width: 480px)").matches);
    },
};
