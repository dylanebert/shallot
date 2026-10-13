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
    const sunRotation = lookAtRotation(0, 0, 0, -0.4, -1, -0.55);
    world.add(sun, Transform, {
        rotation: [sunRotation.x, sunRotation.y, sunRotation.z, sunRotation.w],
    });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 3449.1713,
    });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x1c1917 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.6, pitch: 0.25 });
    world.add(camera, Transform);
    const cube = world.create();
    world.add(cube, MeshInstance);
    world.add(cube, Transform, { translation: [0, 0, 0, 0] });
    const materials = world.resource(Materials);
    const material = materials.add(
        StandardMaterial({
            baseColor: [0.621407, 0.26791815, 0.11756791, 1],
            perceptualRoughness: 0.38,
        }),
    );
    world.add(cube, MeshMaterial, material);
    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(ground, Transform, { translation: [0, -0.6, 0, 0], scale: [4, 0.2, 4, 0] });
    world.add(
        ground,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.38498746, 0.3374035, 0.26791645, 1],
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
