import { expect, test } from "bun:test";
import { Body, PhysicsPlugin, Shape, ShapeKind } from "../../core/physics";
import { Time, World } from "../../engine";
import { Character, CharacterPlugin } from "./character";
import gold from "./character-duration.gold.json";
import { BodyType } from "./common/types";
import { physicsState, StandardPhysicsPlugin } from "./runtime";
import { hashWorldState } from "./world/hash";

const values = new Float32Array(3);
const words = new Uint32Array(values.buffer);

test("Character's production fixed duration matches native mover free flight", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const plugin of [PhysicsPlugin, CharacterPlugin])
        for (const component of plugin.components!) world.registry.register(component);
    for (const system of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!])
        world.addSystem(system);
    try {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Kinematic });
        world.add(eid, Shape, { kind: ShapeKind.Capsule });
        world.storage(Shape).capsuleA.set(eid, 0, -0.5, 0, 0);
        world.storage(Shape).capsuleB.set(eid, 0, 0.5, 0, 0.3);
        world.add(eid, Character, { velocity: [3, 0, 0] });
        world.step(Time.FIXED_DT);
        const position = world.storage(Body).position;
        values.set([position.x.get(eid), position.y.get(eid), position.z.get(eid)]);
        console.log("character-duration", {
            position: [...words],
            hash: `0x${hashWorldState(physicsState(world)!).toString(16)}`,
        });
        expect([...words]).toEqual(gold.position);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
