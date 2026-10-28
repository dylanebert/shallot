import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    pressButton,
    pressKey,
    qRotate,
    releaseKey,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { RigidBodyPlugin, RigidBodyState } from "../src/rigid-body";
import { controlValues, createScene } from "../src/scenes";
import { setSceneCamera } from "../src/view";

setDefaultTimeout(TIMEOUT.node);
test("Rigid Body keeps all six keys live and follows its character without T or V modes", async () => {
    let camera = 0;
    const app = await createApp({
        defaults: false,
        plugins: [RigidBodyPlugin, OrbitPlugin],
        setup(world) {
            createScene(world, "rigidBody");
            camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            world.add(camera, Orbit);
            setSceneCamera(world, "rigidBody", camera);
        },
    });
    try {
        const world = app.world,
            state = world.resource(RigidBodyState);
        world.step(Time.FIXED_DT);
        expect(controlValues("rigidBody")).toEqual({});
        const start = { ...state.character.position };
        pressButton(world, "right");
        for (const key of ["W", "A", "S", "D", "Space", "Shift"] as const) {
            const code = key === "Shift" ? "ShiftLeft" : key === "Space" ? "Space" : `Key${key}`;
            pressKey(world, code);
            for (let i = 0; i < 60; i++) {
                world.step(Time.FIXED_DT);
                expect(state.keys[key]).toBe(1);
                expect(world.get(camera, Orbit)!.target).toBe(state.character.body);
                const pose = world.get(camera, GlobalTransform)!;
                const forward = qRotate(...pose.rotation, 0, 0, -1);
                const target = pose.translation.map((value, axis) => value + 6 * forward[axis]!);
                const actor = world.get(state.character.body, GlobalTransform)!.translation;
                for (const [axis, value] of actor.entries())
                    expect(target[axis]).toBeCloseTo(value, 4);
            }
            releaseKey(world, code);
            world.step(Time.FIXED_DT);
            expect(state.keys[key]).toBe(0);
        }
        for (const code of ["KeyT", "KeyV"]) {
            pressKey(world, code);
            world.step(Time.FIXED_DT);
            expect(world.get(camera, Orbit)!.target).toBe(state.character.body);
            expect(Object.keys(state.keys)).toEqual(["W", "A", "S", "D", "Space", "Shift"]);
            releaseKey(world, code);
        }
        expect(state.character.position).not.toEqual(start);
    } finally {
        app.dispose();
    }
});
