import { MeshInstance } from "../../core/mesh";
import { DirectionalLight, GlobalAmbientLight, VolumetricLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { Materials, MeshMaterial, StandardMaterial } from "../../standard/rendering";
import { Fog } from ".";

// Carbon walls around a 6.6 × 3.2 × 6.6 m interior, a 2 × 1.5 m rear window and a low sun.
// Ambient is the sun-only interreflection balance, ρΦ / (πA(1 − ρ)), in cd/m².
export function room(world: World): number {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 9.223178581445604;
    const carbon = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1] }));
    for (const [translation, scale] of [
        [
            [0, -0.25, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 3.45, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 1.6, 3.5],
            [7, 3.2, 0.4],
        ],
        [
            [-3.5, 1.6, 0],
            [0.4, 3.2, 7],
        ],
        [
            [3.5, 1.6, 0],
            [0.4, 3.2, 7],
        ],
        [
            [-2.25, 1.6, -3.5],
            [2.5, 3.2, 0.4],
        ],
        [
            [2.25, 1.6, -3.5],
            [2.5, 3.2, 0.4],
        ],
        [
            [0, 0.65, -3.5],
            [2, 1.3, 0.4],
        ],
        [
            [0, 3, -3.5],
            [2, 0.4, 0.4],
        ],
    ] as const) {
        const block = world.create();
        world.add(block, Transform, { translation, scale });
        world.add(block, MeshInstance);
        world.add(block, MeshMaterial, carbon);
    }
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 4, -8, 0, 0, 0) });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 24000,
        shadowMapsEnabled: 1,
        maximumDistance: 20,
    });
    world.add(sun, VolumetricLight);
    world.add(world.create(), Fog, {
        brightness: 0,
        density: 0.025,
        absorption: 0,
        scattering: 1,
        lightIntensity: 1 / Math.PI,
        scatteringAsymmetry: 0.4,
        heightFalloff: 0,
    });
    return sun;
}
