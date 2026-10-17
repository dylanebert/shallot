import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";
import { layout, type Placement, POPULATION, SEED } from "./layout";

export const COLORS = [
    [0.085, 0.08, 0.072, 1],
    [0.15, 0.14, 0.125, 1],
    [0.042, 0.039, 0.035, 1],
    [0.72, 0.4, 0.05, 1],
    [0.04, 0.43, 0.53, 1],
    [0.658, 0.301, 0.117, 1],
] as const;
export const City: Resource<{ camera: number }> = { create: () => ({ camera: 0 }) };

function spawn(world: World, placement: Placement, materials: MaterialHandle[]): number {
    const eid = world.create();
    world.add(eid, Transform, {
        translation: placement.translation,
        rotation: [0, Math.sin(placement.yaw / 2), 0, Math.cos(placement.yaw / 2)],
        scale: placement.scale,
    });
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, materials[placement.material]);
    return eid;
}

export function cityPlugin(seed = SEED, population = POPULATION): Plugin {
    return {
        name: "City",
        dependencies: [OrbitPlugin],
        initialize(world) {
            const materials = COLORS.map((baseColor) =>
                world
                    .resource(Materials)
                    .add(StandardMaterial({ baseColor, perceptualRoughness: 0.7 })),
            );
            for (const placement of layout(seed, population)) spawn(world, placement, materials);
            Object.assign(world.resource(GlobalAmbientLight), { color: 0xd8e4ff, brightness: 380 });
            const sun = world.create();
            world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
            world.add(sun, DirectionalLight, {
                color: 0xffe8cc,
                illuminance: 6000,
                shadowMapsEnabled: 1,
                maximumDistance: 600,
            });
            const camera = world.create();
            world.add(camera, Camera, { clearColor: 0x191816, far: 2000 });
            world.add(camera, StandardRenderer);
            world.add(camera, Transform);
            world.add(camera, Orbit, { distance: 260, yaw: 0.65, pitch: 0.6, flySpeed: 35 });
            world.resource(City).camera = camera;
        },
    };
}
export const CityPlugin = cityPlugin();
