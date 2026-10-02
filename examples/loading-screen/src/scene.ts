import {
    AmbientLight,
    Camera,
    DirectionalLight,
    Materials,
    Mesh3d,
    MeshMaterial3d,
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
    const ambient = world.create();
    world.add(ambient, AmbientLight, { color: 0xd0dcec, intensity: 0.5 });
    const sun = world.create();
    world.add(sun, DirectionalLight, {
        direction: [-0.4, -1, -0.55, 0],
        color: 0xfff4e0,
        intensity: 1.1,
    });
    const camera = world.create();
    // Neutral + sRGB inverse for --bg2 (#1c1917): one red level lower after 8-bit quantization.
    world.add(camera, Camera, { clearColor: 0x383736 });
    world.add(camera, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.6, pitch: 0.25 });
    world.add(camera, Transform);
    const cube = world.create();
    world.add(cube, Mesh3d);
    world.add(cube, Transform, { translation: [0, 0, 0, 0] });
    world.add(cube, MeshMaterial3d, {
        material: world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0.85, 0.55, 0.35, 1] })),
    });
}

export const LoadingWorld: Plugin = {
    name: "LoadingWorld",
    dependencies: [OrbitPlugin],
    initialize: authorWorld,
};
