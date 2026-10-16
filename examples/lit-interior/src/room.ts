import {
    Body,
    BodyType,
    Camera,
    Character,
    GlobalAmbientLight,
    internText,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    PointLight,
    type Resource,
    Shape,
    ShapeKind,
    SpotLight,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Text,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Billboard } from "@dylanebert/shallot/rendering";
import { Flashlight, FlashlightPlugin } from "./flashlight";
import { InteractionInput, InteractPlugin, LAMP_EMISSION, Usable } from "./interact";
import { LocalPlayer, Player, PlayerPlugin } from "./player";

const GROUND = [0.085, 0.08, 0.072, 1] as const;
const AMBER = [0.72, 0.4, 0.05, 1] as const;
const CYAN = [0.04, 0.43, 0.53, 1] as const;
const ORANGE = [0.658, 0.301, 0.117, 1] as const;
type Color = readonly [number, number, number, number];
type Vec3 = readonly [number, number, number];
function block(world: World, position: Vec3, size: Vec3, color: Color = GROUND): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale: size });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: color })),
    );
    return eid;
}
export const Room: Resource<{
    camera: number;
    player: number;
    targets: number[];
    lamps: number[];
    object: number;
    shade: number[];
}> = {
    create: () => ({ camera: 0, player: 0, targets: [], lamps: [], object: 0, shade: [] }),
};
function room(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 30;
    const lamps: number[] = [];
    const shade: number[] = [];
    for (const [x, shaded] of [
        [-1.8, true],
        [1.8, false],
    ] as const) {
        block(world, [x, 3.16, -0.5], [0.4, 0.08, 0.4]);
        block(world, [x, 3.02, -0.5], [0.06, 0.2, 0.06]);
        const light = block(world, [x, 2.9, -0.5], [0.24, 0.14, 0.24], AMBER);
        world.add(light, NotShadowCaster);
        world.resource(Materials).update(world.storage(MeshMaterial).material.get(light), {
            emissive: LAMP_EMISSION,
        });
        world.add(light, PointLight, {
            color: 0xffe8cc,
            intensity: 80000,
            range: 8,
            shadowMapsEnabled: 1,
        });
        lamps.push(light);
        if (shaded) {
            for (const [offset, size] of [
                [
                    [-0.31, -0.17, 0],
                    [0.08, 0.56, 0.7],
                ],
                [
                    [0.31, -0.17, 0],
                    [0.08, 0.56, 0.7],
                ],
                [
                    [0, -0.17, -0.31],
                    [0.54, 0.56, 0.08],
                ],
                [
                    [0, 0.04, 0.31],
                    [0.54, 0.14, 0.08],
                ],
                [
                    [0, -0.43, 0.31],
                    [0.54, 0.06, 0.08],
                ],
                [
                    [0, 0.11, 0],
                    [0.7, 0.04, 0.04],
                ],
                [
                    [0, 0.11, 0],
                    [0.04, 0.04, 0.7],
                ],
            ] as const) {
                const panel = block(
                    world,
                    [x + offset[0], 2.89 + offset[1], -0.5 + offset[2]],
                    size,
                    AMBER,
                );
                world.add(panel, NotShadowCaster);
                shade.push(panel);
            }
        }
    }
    for (const x of [-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4]) {
        const lamp = block(world, [x, 0.6, -3.15], [0.16, 0.16, 0.16], AMBER);
        world.add(lamp, NotShadowCaster);
        world.add(lamp, PointLight, { color: 0xffc773, intensity: 900, range: 4 });
    }
    const camera = world.create();
    world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Transform, { translation: [0, 1.7, 2] });
    world.add(camera, InteractionInput);
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
    const surfaces = [
        [
            [0, -0.25, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 3.45, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 1.6, -3.5],
            [7, 3.2, 0.4],
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
    ] as const;
    for (const [position, size] of surfaces) block(world, position, size);
    const object = block(world, [-1.8, 0.5, -0.5], [0.65, 1, 0.65], ORANGE);
    const button = block(world, [0, 1.4, -3.23], [0.3, 0.4, 0.08], CYAN);
    world.add(button, Usable, {
        lamp: lamps[0],
        active: 1,
        label: internText(world, "Switch on"),
        activeLabel: internText(world, "Switch off"),
    });
    for (const [target, name] of [[button, "Ceiling lamp"]] as const) {
        const label = world.create();
        world.add(label, Transform, {
            translation: [world.storage(Transform).translation.x.get(target), 1.9, -3.2],
        });
        world.add(label, Text, {
            content: internText(world, name),
            fontSize: 0.2,
            anchor: [0.5, 0],
            billboard: Billboard.Screen,
            color: 0xeee9df,
        });
    }
    const flashlight = world.create();
    world.add(flashlight, Transform);
    world.add(flashlight, SpotLight, {
        color: 0xd8e4ff,
        intensity: 180000,
        range: 10,
        innerAngle: 0.18,
        outerAngle: 0.35,
        shadowMapsEnabled: 1,
    });
    Object.assign(world.resource(Flashlight), { camera, light: flashlight });
    Object.assign(world.resource(Room), {
        camera,
        player,
        targets: [button],
        lamps,
        object,
        shade,
    });
}
export const RoomPlugin: Plugin = {
    name: "Room",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, InteractPlugin, FlashlightPlugin],
    initialize: room,
};
