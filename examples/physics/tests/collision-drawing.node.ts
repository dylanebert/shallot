import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape } from "@dylanebert/shallot";
import { emptyQueryHit, PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CollisionPlugin, CollisionState, collisionNames } from "../src/collision";
import { collisionDrag } from "../src/collision-drag";
import { collisionDrawing } from "../src/collision-presentation";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
const INK = 0xeee9df,
    AXIS = 0x797369,
    ORANGE = 0xff8135;
for (const scene of collisionNames)
    test(`${scene}: scene framing, role bodies and query geometry, not debug axes`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [CollisionPlugin],
            setup: (w) => {
                createScene(w, scene);
            },
        });
        try {
            app.world.tick();
            const s = app.world.resource(CollisionState),
                drawing = collisionDrawing(app.world);
            expect(scenes[scene].camera.phoneDistance).toBeGreaterThan(
                scenes[scene].camera.distance,
            );
            expect(scenes[scene].gridSpacing).toBe(scene === "castWorld" ? 2 : 1);
            expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(
                scene !== "meshScale",
            );
            expect(drawing.segments.length + drawing.marks.length).toBeGreaterThan(0);
            expect(drawing.segments.every((s) => [INK, AXIS, ORANGE].includes(s.color))).toBe(true);
            expect(drawing.marks.every((s) => [INK, AXIS, ORANGE].includes(s.color))).toBe(true);
            expect(drawing.segments).not.toContainEqual({ a: [0, 0, 0], b: [1, 0, 0], color: INK });
            if (scene === "rayCurtain") {
                expect(s.sweeps).toHaveLength(161);
                expect(
                    s.bodies.every(
                        (eid) => app.world.storage(Body).type.get(eid) === BodyType.Kinematic,
                    ),
                ).toBe(true);
                expect(drawing.marks).toHaveLength(322);
            }
            if (scene === "capsuleCastRay") {
                expect(drawing.segments).toEqual([{ a: [-1, 0.5, 0], b: [1, 0.5, 0], color: INK }]);
                expect(drawing.marks).toHaveLength(3);
            }
            if (scene === "shapeCast") expect(s.sweeps).toHaveLength(12);
            if (scene === "overlapWorld") {
                expect(s.sweeps).toHaveLength(15);
                expect(drawing.segments.every((s) => s.color === AXIS)).toBe(true);
                s.sweeps[0]!.overlap = true;
                expect(collisionDrawing(app.world).segments.some((s) => s.color === INK)).toBe(
                    true,
                );
            }
            if (scene === "meshScale") {
                expect(drawing.segments).toContainEqual({
                    a: [-2, 0, 0],
                    b: [2, 0, 0],
                    color: INK,
                });
                const h = s.sweeps[0]!.hits[0]!;
                // The native sample's sphere uses fraction * translation, without its ray origin.
                expect(
                    Math.min(...drawing.segments.slice(0, 96).flatMap((s) => [s.a[0], s.b[0]])),
                ).toBeCloseTo(4 * h.fraction - 0.25);
            }
        } finally {
            app.dispose();
        }
    });

test("Cast World's excluded bounds keep axis ink without changing body roles", async () => {
    let bodies: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [CollisionPlugin],
        setup: (w) => {
            bodies = createScene(w, "castWorld");
        },
    });
    try {
        const values = controlValues("castWorld");
        changeControl(app.world, "castWorld", bodies, values, "Spheres", 1);
        app.world.tick();
        const s = app.world.resource(CollisionState);
        expect(s.ignored).toHaveLength(1);
        const body = s.ignored[0]!,
            type = app.world.storage(Body).type.get(body),
            shape = app.world.get(body, Shape)!;
        expect(collisionDrawing(app.world).segments.filter((s) => s.color === AXIS)).toHaveLength(
            12,
        );
        expect(app.world.storage(Body).type.get(body)).toBe(type);
        expect(app.world.get(body, Shape)).toEqual(shape);
    } finally {
        app.dispose();
    }
});

test("Long Ray Cast draws accurate, drifting, accuracy-miss and geometric-miss states and retains 180 unfaded hits", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [CollisionPlugin],
        setup: (w) => {
            createScene(w, "longRayCast");
        },
    });
    try {
        for (let i = 0; i < 200; i++) app.world.tick();
        const s = app.world.resource(CollisionState);
        expect(s.trails.every((t) => t.length <= 180)).toBe(true);
        expect(s.trails.some((t) => t.length === 180)).toBe(true);
        const hit = {
            ...emptyQueryHit(),
            point: { x: 1, y: 2, z: 3 },
            normal: { x: 0, y: 1, z: 0 },
        };
        for (let i = 0; i < 5; i++) {
            s.trails[i] = [];
            s.sweeps[2 * i]!.hits = [];
            s.sweeps[2 * i + 1]!.hits = [];
        }
        s.sweeps[1]!.hits = [hit];
        s.sweeps[1]!.color = INK;
        s.sweeps[3]!.hits = [hit];
        s.sweeps[3]!.color = ORANGE;
        s.sweeps[4]!.hits = [hit];
        s.sweeps[5]!.color = ORANGE;
        s.trails[0] = [
            { point: [4, 5, 6], color: INK },
            { point: [7, 8, 9], color: ORANGE },
        ];
        const drawing = collisionDrawing(app.world);
        expect(drawing.marks).toContainEqual({ point: [1, 2, 3], color: INK });
        expect(drawing.marks).toContainEqual({ point: [1, 2, 3], color: ORANGE });
        expect(drawing.marks).toContainEqual({ point: [4, 5, 6], color: INK });
        expect(drawing.marks).toContainEqual({ point: [7, 8, 9], color: ORANGE });
        expect(drawing.segments.filter((s) => s.color === AXIS)).toHaveLength(2);
    } finally {
        app.dispose();
    }
});

for (const scene of ["shapeCast", "overlapWorld"] as const)
    test(`${scene}: query shapes take pointer drag instead of body grab`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [CollisionPlugin],
            setup: (w) => {
                createScene(w, scene);
            },
        });
        try {
            app.world.tick();
            const s = app.world.resource(CollisionState),
                origin = { x: -6, y: 3, z: -15 },
                translation = { x: 0, y: 0, z: 20 };
            expect(
                collisionDrag(app.world, { x: 100, y: 100, z: 100 }, translation, true, true),
            ).toBe(false);
            expect(collisionDrag(app.world, origin, translation, true, true)).toBe(true);
            expect(
                collisionDrag(app.world, { ...origin, x: -5, y: 4 }, translation, false, true),
            ).toBe(true);
            expect(s.offset).toEqual({ x: 1, y: 1, z: 0 });
            app.world.tick();
            expect(s.sweeps[0]!.points[0]).toEqual({ x: -5, y: 4, z: -5 });
            expect(collisionDrag(app.world, origin, translation, false, false)).toBe(false);
        } finally {
            app.dispose();
        }
    });
