import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, pressKey, Time } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { drawingHistory } from "../src/drawing-history";
import { GeometricMoverPlugin, GeometricMoverState } from "../src/geometric-mover";
import { drawPogoRay, pogoDrawing } from "../src/mover-presentation";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("CharacterMover records its native pogo hit, draws only the support ray, and draws its full ray when jumping", async () => {
    const app = await createApp({ defaults: false, plugins: [GeometricMoverPlugin] });
    try {
        const world = app.world;
        createScene(world, "geometricMover");
        const mover = world.resource(GeometricMoverState).mover;
        expect(pogoDrawing(mover)).toEqual([]);
        for (let i = 0; i < 180; i++) world.tick();
        expect(mover.onGround).toBe(true);
        expect(mover.capsule.radius).toBeCloseTo(0.3, 6);
        const ray = mover.pogoRay!;
        expect(ray.origin.y - ray.end.y).toBeCloseTo(3 * mover.capsule.radius, 1);
        expect(
            mover.position.y + mover.capsule.center1.y - mover.capsule.radius - ray.end.y,
        ).toBeGreaterThan(0.55);
        expect(
            mover.position.y + mover.capsule.center1.y - mover.capsule.radius - ray.end.y,
        ).toBeLessThan(0.65);
        const calls: {
            a: number[];
            b: number[];
            color: number;
            options: { width: number; depthBias: number };
        }[] = [];
        drawPogoRay(pogoDrawing(mover), (a, b, color, options) =>
            calls.push({ a, b, color, options }),
        );
        expect(calls).toEqual([
            {
                a: [ray.origin.x, ray.origin.y, ray.origin.z],
                b: [ray.end.x, ray.end.y, ray.end.z],
                color: 0xeee9df,
                options: { width: 1, depthBias: -1 },
            },
        ]);
        const before = { ...mover.position };
        world.resource(GeometricMoverState).keys.Space = 1;
        world.tick();
        expect(mover.onGround).toBe(false);
        expect(mover.pogoRay!.origin).toEqual({
            x: before.x,
            y: Math.fround(before.y + mover.capsule.center1.y),
            z: before.z,
        });
        expect(mover.pogoRay!.origin.y - mover.pogoRay!.end.y).toBeCloseTo(
            4 * mover.capsule.radius,
            6,
        );
        calls.length = 0;
        drawPogoRay(pogoDrawing(mover), (a, b, color, options) =>
            calls.push({ a, b, color, options }),
        );
        expect(calls).toHaveLength(1);
        expect(calls[0]!.color).toBe(0x797369);
        expect(calls[0]!.options).toEqual({ width: 1, depthBias: -1 });
    } finally {
        app.dispose();
    }
});

test("the pogo ray uses the capsule's fixed-alpha cadence at 120 Hz, without changing query records", async () => {
    const app = await createApp({ defaults: false, plugins: [GeometricMoverPlugin] });
    try {
        const world = app.world;
        createScene(world, "geometricMover");
        const mover = world.resource(GeometricMoverState).mover;
        const history = drawingHistory(() => pogoDrawing(mover), "pogo");
        for (const system of history.systems) world.addSystem(system);
        for (let i = 0; i < 180; i++) world.tick();
        pressKey(world, "KeyW");
        for (let i = 0; i < 3; i++) world.step(Time.FIXED_DT);
        const previous = pogoDrawing(mover)[0]!;
        world.step(Time.FIXED_DT);
        const current = pogoDrawing(mover)[0]!;
        expect(current[3]! - previous[3]!).toBeLessThan(-0.09);
        expect(history.records(world)[0]).toEqual(previous);
        world.step(Time.FIXED_DT / 2);
        expect(history.records(world)[0]).toEqual(
            current.map((value, i) => (i === 0 ? value : (value + previous[i]!) / 2)),
        );
        expect(pogoDrawing(mover)[0]).toEqual(current);
        pressKey(world, "Space");
        world.step(Time.FIXED_DT / 2);
        expect(mover.onGround).toBe(false);
        const airborne = history.records(world)[0]!;
        expect(airborne).toEqual(pogoDrawing(mover)[0]);
        expect(airborne[2]! - airborne[5]!).toBeCloseTo(4 * mover.capsule.radius, 6);
    } finally {
        app.dispose();
    }
});
