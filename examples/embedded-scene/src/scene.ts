import {
    AlphaMode,
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
    Tonemapping,
    TonemappingMethod,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";

export function authorWorld(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd0dcec;
    ambient.brightness = 499.04787;
    const sun = world.create();
    const sunRotation = lookAtRotation(0, 0, 0, -0.4, -1, -0.55);
    world.add(sun, Transform, {
        rotation: [sunRotation.x, sunRotation.y, sunRotation.z, sunRotation.w],
    });
    world.add(sun, DirectionalLight, {
        color: 0xfff4e0,
        illuminance: 3449.1713,
    });
    const camera = world.create();
    // Neutral + sRGB inverse for --bg2 (#1c1917): one red level lower after 8-bit quantization.
    world.add(camera, Camera, { clearColor: 0x383736 });
    world.add(camera, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.6, pitch: 0.25 });
    world.add(camera, Transform);
    const cube = world.create();
    world.add(cube, MeshInstance);
    world.add(cube, Transform, { translation: [0, 0, 0, 0] });
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.85, 0.55, 0.35, 0.72] }), {
            alphaMode: AlphaMode.Blend,
        });
    world.add(cube, MeshMaterial, material);
}

export const LoadingWorld: Plugin = {
    name: "LoadingWorld",
    dependencies: [OrbitPlugin],
    initialize: authorWorld,
};
