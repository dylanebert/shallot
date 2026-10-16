import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";

export function authorWorld(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 499.04787;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -1.2, -1, -0.1) });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 3449.1713,
    });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.6, pitch: 0.25 });
    world.add(camera, Transform);
    const cube = world.create();
    world.add(cube, MeshInstance);
    world.add(cube, Transform, { translation: [0, 0, 0] });
    const materials = world.resource(Materials);
    const material = materials.add(
        StandardMaterial({
            baseColor: [0.72, 0.4, 0.05, 1],
            perceptualRoughness: 0.38,
        }),
    );
    world.add(cube, MeshMaterial, material);
    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(ground, Transform, { translation: [0, -0.6, 0], scale: [4, 0.2, 4] });
    world.add(
        ground,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.085, 0.08, 0.072, 1],
                perceptualRoughness: 0.82,
            }),
        ),
    );
}

export const LoadingWorld: Plugin = {
    name: "LoadingWorld",
    dependencies: [OrbitPlugin],
    initialize: authorWorld,
};
