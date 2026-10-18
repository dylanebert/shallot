import {
    Body,
    BodyType,
    Camera,
    Character,
    DirectionalLight,
    Exposure,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
    VolumetricLight,
    type World,
} from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, Fog, FogPlugin } from "@dylanebert/shallot/extras";
import { LocalPlayer, Player, PlayerPlugin } from "./player";

export const Room: Resource<{ camera: number; player: number }> = {
    create: () => ({ camera: 0, player: 0 }),
};
function block(
    world: World,
    translation: readonly [number, number, number],
    scale: readonly [number, number, number],
): void {
    const eid = world.create();
    world.add(eid, Transform, { translation, scale });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1] })),
    );
}
export const RoomPlugin: Plugin = {
    name: "Room",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, FogPlugin, DiagnosticsOverlayPlugin],
    initialize(world) {
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        // Light the walls (luminance 0.08) to the average luminance EV100 8 meters for, 2^8 · 12.5 / 100 cd/m².
        ambient.brightness = (2 ** 8 * 0.125) / 0.08;
        const camera = world.create();
        world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        world.add(camera, Exposure, { ev100: 8 });
        world.add(camera, Transform, { translation: [0, 1.7, 2] });
        const player = world.create();
        world.add(player, Transform, { translation: [0, 1, 2] });
        world.add(player, Body, { type: BodyType.Kinematic });
        world.add(player, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.3, 0],
            capsuleCenter2: [0, 0.3, 0],
            radius: 0.3,
        });
        world.add(player, Character);
        world.add(player, Player, { camera });
        world.add(player, LocalPlayer);
        Object.assign(world.resource(Room), { camera, player });
        for (const [position, size] of [
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
        ] as const)
            block(world, position, size);
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 5, -8, 0, 1, 0) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 24000,
            shadowMapsEnabled: 1,
            maximumDistance: 20,
        });
        world.add(sun, VolumetricLight);
        const fog = world.create();
        world.add(fog, Fog, {
            // The room's ambient light is the light the dust scatters; the medium has none of its own.
            brightness: 0,
            density: 0.025,
            absorption: 0,
            scattering: 1,
            lightIntensity: 1 / Math.PI,
            scatteringAsymmetry: 0.4,
            heightFalloff: 0,
        });
    },
};
