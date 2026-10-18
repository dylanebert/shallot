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
const WALL: readonly [number, number, number] = [0.085, 0.08, 0.072];
const SUN_ILLUMINANCE = 24000;
// From the room toward the sun, which shines through the window in the back wall (−z).
const SUN_FROM = [0, 4, -8] as const;
// Interior width, height and depth; walls are 0.4 thick; the window is 2 wide and 1.5 high.
const INTERIOR = [6.6, 3.2, 6.6] as const;
const WALL_THICKNESS = 0.4;
const WINDOW = [2, 1.5] as const;

/** The room's ambient radiance in cd/m², from the sun's flux through the window alone, interreflected by
 * the walls: L = ρΦ / (πA(1 − ρ)), the integrating-sphere estimate of average interreflected light. */
function interreflectedRadiance(): number {
    const [x, y, z] = SUN_FROM;
    const length = Math.hypot(x, y, z);
    // The window's depth shades its sill from a sun this high.
    const lit = WINDOW[0] * (WINDOW[1] - (WALL_THICKNESS * y) / -z);
    const flux = SUN_ILLUMINANCE * (-z / length) * lit;
    const [w, h, d] = INTERIOR;
    const area = 2 * w * d + 2 * h * (w + d) - WINDOW[0] * WINDOW[1];
    const reflectance = 0.2126 * WALL[0] + 0.7152 * WALL[1] + 0.0722 * WALL[2];
    return (reflectance * flux) / (Math.PI * area * (1 - reflectance));
}
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
        world.resource(Materials).add(StandardMaterial({ baseColor: [...WALL, 1] })),
    );
}
export const RoomPlugin: Plugin = {
    name: "Room",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, FogPlugin, DiagnosticsOverlayPlugin],
    initialize(world) {
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = interreflectedRadiance();
        const camera = world.create();
        world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        // EV5 keeps the shaded wall 2.68× above the clear without clipping a shaft mote.
        world.add(camera, Exposure, { ev100: 5 });
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
        world.add(sun, Transform, { rotation: lookAtRotation(...SUN_FROM, 0, 0, 0) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: SUN_ILLUMINANCE,
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
