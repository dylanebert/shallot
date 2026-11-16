import {
    Body,
    BodyType,
    Character,
    CharacterPlugin,
    component,
    type Plugin,
    type Resource,
    Shape,
    ShapeKind,
    setTargetTransform,
    Transform,
    vec4,
    type World,
} from "@dylanebert/shallot";
import { Player, PlayerPlugin } from "./player";

export const Tint = component("Tint", { color: vec4 });
export const Lift = component("Lift", {});
export const Route: Resource<{ entities: ReturnType<typeof route> | null }> = {
    create: () => ({ entities: null }),
};

function block(
    world: World,
    at: readonly [number, number, number],
    size: readonly [number, number, number],
    color: readonly [number, number, number, number],
    type: BodyType = BodyType.Static,
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: at, scale: size });
    world.add(eid, Body, { type });
    world.add(eid, Shape);
    world.add(eid, Tint, { color });
    return eid;
}
function avatar(world: World, x: number, z: number) {
    const eye = world.create();
    world.add(eye, Transform);
    const player = world.create();
    world.add(player, Transform, { translation: [x, 1.45, z] });
    world.add(player, Body, { type: BodyType.Kinematic });
    world.add(player, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, -0.6, 0],
        capsuleCenter2: [0, 0.6, 0],
        radius: 0.3,
    });
    world.add(player, Character);
    world.add(player, Player, { camera: eye });
    world.add(player, Tint, { color: [0.04, 0.43, 0.53, 1] });
    return { player, eye };
}
export function route(world: World) {
    const players = [avatar(world, 2, 15), avatar(world, -2, 10)];
    block(world, [0, 0, -5], [32, 1, 52], [0.085, 0.08, 0.072, 1]);
    for (const [at, size] of [
        [
            [0, 0.75, 3],
            [6, 0.5, 3],
        ],
        [
            [0, 1.25, 0],
            [6, 0.5, 3],
        ],
        [
            [0, 1.75, -3],
            [6, 0.5, 3],
        ],
    ] as const)
        block(world, at, size, [0.15, 0.14, 0.125, 1]);
    const lift = block(
        world,
        [0, 1.75, -6.5],
        [6, 0.5, 4],
        [0.658, 0.301, 0.117, 1],
        BodyType.Kinematic,
    );
    world.add(lift, Lift);
    for (const [at, size, color] of [
        [
            [0.7, 2.5, -10],
            [4.4, 1, 2.4],
            [0.06, 0.057, 0.052, 1],
        ],
        [
            [-0.7, 3.5, -12.3],
            [3.6, 1, 2.4],
            [0.06, 0.057, 0.052, 1],
        ],
        [
            [0.7, 4.5, -14.5],
            [4.4, 1, 3],
            [0.11, 0.105, 0.095, 1],
        ],
    ] as const)
        block(world, at, size, color);
    const box = block(world, [2, 0.9, 12], [0.8, 0.8, 0.8], [0.72, 0.4, 0.05, 1], BodyType.Dynamic);
    return { players, box, lift };
}
export const Demo: Plugin = {
    name: "Route",
    dependencies: [PlayerPlugin],
    components: [Tint, Lift],
    recovery: "stateless",
    initialize(world) {
        world.resource(Route).entities = route(world);
    },
    systems: [
        {
            name: "lift",
            group: "fixed",
            before: CharacterPlugin.systems,
            update(world) {
                const y = 1.75 + 0.75 * (1 - Math.cos(2 * world.time.elapsed * 0.65));
                for (const eid of world.query([Lift]))
                    setTargetTransform(world, eid, [0, y, -6.5], [0, 0, 0, 1]);
            },
        },
    ],
};
