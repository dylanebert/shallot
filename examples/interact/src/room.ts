import {
    Body,
    BodyType,
    Camera,
    Character,
    DirectionalLight,
    GlobalAmbientLight,
    internText,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    PointLight,
    type Resource,
    registerImage,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Text,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Billboard } from "@dylanebert/shallot/rendering";
import { Action, InteractionInput, InteractPlugin, makePrompt, Usable } from "./interact";
import { LocalPlayer, Player, PlayerPlugin } from "./player";

const GROUND = [0.38498746, 0.3374035, 0.26791645, 1] as const;
const ORANGE = [0.621407, 0.26791815, 0.11756791, 1] as const;
const BLUE = [0.15270135, 0.36628627, 0.70424525, 1] as const;
const GOLD = [0.41696574, 0.35106622, 0.08533862, 1] as const;
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
export const Room: Resource<{ camera: number; player: number; targets: number[]; prompt: number }> =
    {
        create: () => ({ camera: 0, player: 0, targets: [], prompt: 0 }),
    };
function room(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 520;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.5, -1, -0.6) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const camera = world.create();
    world.add(camera, Camera, { fov: 70, clearColor: 0x1c1917 });
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
    block(world, [0, -0.25, 0], [12, 0.5, 12]);
    block(world, [0, 2, -3.3], [12, 4, 0.4]);
    block(world, [-6, 2, 0], [0.4, 4, 9]);
    block(world, [6, 2, 0], [0.4, 4, 9]);
    block(world, [-2.8, 0.6, -2.5], [0.18, 1.2, 0.18]);
    block(world, [0, 0.6, -2.7], [0.8, 1.2, 0.5]);
    block(world, [2.8, 0.4, -2.5], [1.2, 0.8, 1]);
    const lamp = block(world, [-2.8, 1.5, -2.5], [0.6, 0.6, 0.6], ORANGE);
    const light = world.create();
    world.add(light, Transform, { translation: [-2.8, 2.1, -1.9] });
    world.add(light, PointLight, { color: 0xffe8cc, intensity: 0, range: 5 });
    world.add(lamp, Usable, {
        action: Action.Lamp,
        light,
        label: internText(world, "Switch on"),
        activeLabel: internText(world, "Switch off"),
    });
    const button = block(world, [0, 1.3, -2.4], [0.45, 0.45, 0.25], BLUE);
    world.add(button, Usable, {
        action: Action.Button,
        label: internText(world, "Press"),
        activeLabel: internText(world, "Press"),
    });
    const held = block(world, [2.8, 1.2, -2.5], [0.8, 0.8, 0.6], GOLD);
    world.add(held, Usable, {
        action: Action.Hold,
        label: internText(world, "Hold to lift"),
        activeLabel: internText(world, "Hold to lower"),
    });
    for (const [target, name] of [
        [lamp, "Lamp"],
        [button, "Button"],
        [held, "Held block"],
    ] as const) {
        const label = world.create();
        world.add(label, Transform, {
            translation: [world.storage(Transform).translation.x.get(target), 2.65, -2.5],
        });
        world.add(label, Text, {
            content: internText(world, name),
            fontSize: 0.2,
            anchor: [0.5, 0],
            billboard: Billboard.Screen,
            color: 0xefe3cf,
        });
    }
    const image = registerImage(world, "/key.png");
    const ring = registerImage(world, "/ring.png");
    Object.assign(world.resource(Room), {
        camera,
        player,
        targets: [lamp, button, held],
        prompt: makePrompt(world, image, ring),
    });
}
export const RoomPlugin: Plugin = {
    name: "Room",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, InteractPlugin],
    initialize: room,
};
