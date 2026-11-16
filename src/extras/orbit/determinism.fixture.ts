import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { Camera } from "../../core/rendering";
import { GlobalTransform, Transform } from "../../core/transform";
import { component, createApp, f32 as field, type Plugin, u32 } from "../../engine";
import {
    Character,
    CharacterPlugin,
    computeCosSin,
    hashPhysics,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
} from "../../standard/physics";
import { Orbit, OrbitPlugin } from "./index";

export const actions = Array.from({ length: 240 }, (_, tick) => ({
    yaw: (tick % 120) * 0.025,
    speed: tick < 180 ? 3 : -2,
    jump: tick === 45 || tick === 150,
}));

const Player = component("DeterminismPlayer", { yaw: field, ticks: u32 });

export async function composition(workerCount = 1) {
    let player = 0;
    const plugin: Plugin = {
        name: "DeterminismPlayer",
        components: [Player],
        systems: [
            {
                name: "RecordedPlayer",
                group: "fixed",
                before: CharacterPlugin.systems,
                update(world) {
                    const p = world.storage(Player);
                    const tick = p.ticks.get(player);
                    const action = actions[tick];
                    if (!action) throw new Error(`No action for tick ${tick}`);
                    p.ticks.set(player, tick + 1);
                    p.yaw.set(player, action.yaw);
                    const [c, s] = computeCosSin(world, action.yaw);
                    const character = world.storage(Character);
                    const vy = action.jump ? 5 : character.velocity.y.get(player) - 9.8 / 60;
                    character.velocity.set(player, s * action.speed, vy, c * action.speed, 0);
                },
            },
        ],
    };
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, CharacterPlugin, OrbitPlugin, plugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).workerCount = workerCount;
        },
    });
    const { world } = app;
    const bodies: number[] = [];
    for (let i = 0; i < 5; i++) {
        const eid = world.create();
        bodies.push(eid);
        world.add(eid, Transform, { translation: i === 0 ? [0, -0.5, 0] : [i - 2, 2 + i, 2] });
        world.add(eid, Body, { type: i === 0 ? BodyType.Static : BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Hull, scale: i === 0 ? [40, 1, 40] : [1, 1, 1] });
    }
    const rotated = bodies[1]!;
    world
        .storage(Shape)
        .hullRotation.set(rotated, ...rotationFromAxisAngle(world, { x: 0, y: 1, z: 0 }, 0.4));
    player = world.create();
    world.add(player, Transform, { translation: [0, 1, 0] });
    world.add(player, Body, { type: BodyType.Kinematic });
    world.add(player, Shape, { kind: ShapeKind.Capsule, radius: 0.3 });
    world.storage(Shape).capsuleCenter1.set(player, 0, -0.5, 0, 0);
    world.storage(Shape).capsuleCenter2.set(player, 0, 0.5, 0, 0);
    world.add(player, Character);
    world.add(player, Player);
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, Orbit, { target: player });
    return {
        ...app,
        bodies,
        player,
        camera,
        hash() {
            let hash = hashPhysics(world);
            const bytes = new DataView(new ArrayBuffer(8));
            const fold = (value: number) => {
                bytes.setFloat64(0, value, true);
                for (let i = 0; i < 8; i++)
                    hash = BigInt.asUintN(64, (hash ^ BigInt(bytes.getUint8(i))) * 1099511628211n);
            };
            const fields = (value: unknown): void => {
                if (typeof value === "number") fold(value);
                else if (Array.isArray(value)) value.forEach(fields);
                else if (value && typeof value === "object") {
                    const record = value as Record<string, unknown>;
                    for (const key of Object.keys(record).sort()) fields(record[key]);
                } else throw new Error(`Unexpected gameplay field: ${value}`);
            };
            for (const eid of [...bodies, player]) {
                fields(world.get(eid, Transform));
                fields(world.get(eid, GlobalTransform));
                fields(world.get(eid, Body));
                fields(world.get(eid, Shape));
            }
            fields(world.get(player, Character));
            fields(world.get(player, Player));
            return hash;
        },
    };
}

const approximated = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "cbrt",
    "hypot",
] as const;

export function trappedTick(world: { tick(): void }) {
    const originals = approximated.map((name) => Object.getOwnPropertyDescriptor(Math, name)!);
    try {
        for (const name of approximated)
            Math[name] = () => {
                throw new Error(`Math.${name} in fixed tick`);
            };
        world.tick();
    } finally {
        approximated.forEach((name, i) => {
            Object.defineProperty(Math, name, originals[i]!);
        });
    }
}
