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
import { GeometricMoverPlugin, GeometricMoverState } from "../src/geometric-mover";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { setSceneCamera } from "../src/view";

setDefaultTimeout(TIMEOUT.node);
test("Geometric Mover's follow camera stays attached with every movement key live and Clip Velocity toggled", async () => {
    let camera = 0;
    let bodies: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [GeometricMoverPlugin, OrbitPlugin],
        setup(world) {
            bodies = createScene(world, "geometricMover");
            camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            world.add(camera, Orbit);
            setSceneCamera(world, "geometricMover", camera);
        },
    });
    try {
        const world = app.world,
            s = world.resource(GeometricMoverState);
        world.step(Time.FIXED_DT);
        const start = { ...s.mover.position };
        const values = controlValues("geometricMover");
        for (const value of [0, 1]) {
            changeControl(world, "geometricMover", bodies, values, "Clip Velocity", value);
            expect(s.clipVelocity).toBe(!!value);
        }
        pressButton(world, "right");
        for (const key of ["W", "A", "S", "D", "Space", "Shift"] as const) {
            const code = key === "Shift" ? "ShiftLeft" : key === "Space" ? "Space" : `Key${key}`;
            pressKey(world, code);
            for (let i = 0; i < 60; i++) {
                const previous = { ...s.mover.position };
                world.step(Time.FIXED_DT);
                expect(s.keys[key]).toBe(1);
                expect(world.get(camera, Orbit)!.target).toBe(s.entity);
                const pose = world.get(camera, GlobalTransform)!;
                const forward = qRotate(...pose.rotation, 0, 0, -1);
                const target = pose.translation.map((value, axis) => value + 6 * forward[axis]!);
                for (const [axis, value] of [previous.x, previous.y, previous.z].entries())
                    expect(target[axis]).toBeCloseTo(value, 4);
            }
            releaseKey(world, code);
            world.step(Time.FIXED_DT);
            expect(s.keys[key]).toBe(0);
        }
        expect(s.mover.position).not.toEqual(start);
    } finally {
        app.dispose();
    }
});
