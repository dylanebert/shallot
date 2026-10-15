import gold from "../../../crates/physics/tests/character.gold.json";
import { Body, PhysicsPlugin, Shape, ShapeKind } from "../../core/physics";
import { Time, World } from "../../engine";
import { Character, CharacterPlugin, characterScratch } from "./character";
import { BodyType, defaultBodyDef } from "./common/types";
import { quaternion } from "./kernel/math";
import { physicsState, StandardPhysicsPlugin } from "./runtime";
import { makeBoxHull } from "./shapes/build-hull";
import { createHullShape } from "./shapes/shape";
import { testShapeDef } from "./testing";
import { bodySetAngularVelocity, bodySetLinearVelocity, createBody } from "./world/body";

export { gold as characterGold };
export async function visitCharacterGold(
    cases: typeof gold,
    observe: (actual: number[], expected: number[], name: string) => void,
): Promise<void> {
    const duration = Time.FIXED_DT;
    const scalar = new Float32Array(1);
    const word = new Uint32Array(scalar.buffer);
    const f = (value: number) => {
        word[0] = value;
        return scalar[0];
    };
    try {
        for (const row of cases) {
            const world = new World();
            await StandardPhysicsPlugin.initialize!(world);
            await StandardPhysicsPlugin.warm!(world);
            for (const plugin of [PhysicsPlugin, CharacterPlugin])
                for (const c of plugin.components!) world.registry.register(c);
            for (const s of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!])
                world.addSystem(s);
            try {
                const state = physicsState(world)!;
                const box = (
                    p: number[],
                    half: number[],
                    angle = 0,
                    dynamic = false,
                    awake = false,
                ) => {
                    const body = createBody(state, {
                        ...defaultBodyDef(),
                        position: { x: p[0], y: p[1], z: p[2] },
                        rotation: quaternion.fromAxisAngle({ x: 0, y: 0, z: 1 }, angle),
                        type: dynamic ? BodyType.Dynamic : BodyType.Static,
                        isAwake: awake,
                    });
                    createHullShape(
                        state,
                        body,
                        testShapeDef(),
                        makeBoxHull(half[0], half[1], half[2]),
                    );
                    return body;
                };
                const name = row.name;
                if (!name.startsWith("free")) box([0, -0.5, 0], [10, 0.5, 10]);
                if (name === "wall" || name === "corner") box([0.8, 1, 0], [0.1, 1, 10]);
                if (name === "corner") box([0, 1, 0.8], [10, 1, 0.1]);
                if (name === "low-step" || name === "high-step") {
                    const height = name === "low-step" ? 0.04 : 0.15;
                    box([0.8, height, 0], [0.5, height, 2]);
                }
                if (name === "slope") box([0, 0.1, 0], [3, 0.1, 3], Math.fround(0.4));
                if (name.startsWith("dynamic")) {
                    const id = box([0.6, 1, 0], [0.3, 0.3, 0.3], 0, true, name !== "dynamic-sleep");
                    if (name === "dynamic-double")
                        createHullShape(state, id, testShapeDef(), makeBoxHull(0.3, 0.3, 0.3));
                    if (name === "dynamic-spin") {
                        bodySetLinearVelocity(state, id, { x: -2, y: Math.fround(0.1), z: 1 });
                        bodySetAngularVelocity(state, id, { x: 1, y: 2, z: -3 });
                    }
                }
                const input = row.input.map(f);
                const eid = world.create();
                world.add(eid, Body, {
                    type: BodyType.Kinematic,
                    position: [input[0], input[1], input[2]],
                    rotation: [input[25], input[26], input[27], input[28]],
                });
                world.add(eid, Shape, { kind: ShapeKind.Capsule });
                world.storage(Shape).capsuleCenter1.set(eid, input[9], input[10], input[11]);
                world.storage(Shape).capsuleCenter2.set(eid, input[12], input[13], input[14]);
                world.storage(Shape).radius.set(eid, input[15]);
                world.add(eid, Character, {
                    velocity: [input[3], input[4], input[5]],
                    up: [input[6], input[7], input[8]],
                    pogoVelocity: input[17],
                    pogoRest: input[18],
                });
                Object.defineProperty(Time, "FIXED_DT", { value: input[16] });
                world.step(Time.FIXED_DT);
                observe(Array.from(world.resource(characterScratch).outputU), row.output, name);
            } finally {
                await StandardPhysicsPlugin.dispose!(world);
                world.dispose();
            }
        }
    } finally {
        Object.defineProperty(Time, "FIXED_DT", { value: duration });
    }
}
