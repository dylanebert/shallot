import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    Compounds,
    createApp,
    Hulls,
    Shape,
    ShapeKind,
    Transform,
} from "@dylanebert/shallot";
import { PhysicsEvents, setTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { drawEventMark, type EventMark } from "../src/event-presentation";
import { EventSample, EventSamplePlugin, persistentContact } from "../src/events";

setDefaultTimeout(TIMEOUT.node);

test("Persistent Contact draws each native step's manifold point, impulse line and two-decimal label in scene ink", async () => {
    const gold: EventMark[][] = (
        await Bun.file(new URL("./persistentContact.drawing.gold.txt", import.meta.url)).text()
    )
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
    const app = await createApp({
        defaults: false,
        plugins: [EventSamplePlugin],
        setup: persistentContact,
    });
    let points = 0;
    try {
        for (const expected of gold) {
            app.world.tick();
            const marks = app.world.resource(EventSample).contactMarks;
            expect(marks).toEqual(
                expected.map((mark) => ({
                    ...mark,
                    point: mark.point.map(Math.fround) as [number, number, number],
                    end: mark.end.map(Math.fround) as [number, number, number],
                })),
            );
            for (const mark of marks) {
                const lines: unknown[] = [];
                drawEventMark(mark, (a, b, color, options) => lines.push({ a, b, color, options }));
                expect(lines).toEqual([
                    {
                        a: mark.end,
                        b: mark.point,
                        color: 0xffb62e,
                        options: { width: 2, depthBias: -1 },
                    },
                    {
                        a: [mark.point[0] - 0.2, mark.point[1], mark.point[2]],
                        b: [mark.point[0] + 0.2, mark.point[1], mark.point[2]],
                        color: 0xffb62e,
                        options: { width: 3, depthBias: -1 },
                    },
                ]);
                points++;
            }
        }
        expect(gold).toHaveLength(600);
        expect(points).toBeGreaterThan(0);
    } finally {
        app.dispose();
    }
});

test("Persistent Contact retains the first begin's child when another contact of that pair ends", async () => {
    let body = 0;
    const app = await createApp({
        defaults: false,
        plugins: [EventSamplePlugin],
        setup(world) {
            world.resource(EventSample).scene = "persistentContact";
            const ground = world.create();
            world.add(ground, Body);
            world.add(ground, Shape, {
                kind: ShapeKind.Compound,
                geometry: world.resource(Compounds).add({
                    name: "two spheres",
                    spheres: [
                        { center: [-1, 0, 0], radius: 0.5 },
                        { center: [1, 0, 0], radius: 0.5 },
                    ],
                }),
            });
            body = world.create();
            world.add(body, Body, { type: BodyType.Dynamic });
            world.add(body, Transform, { translation: [0, 1, 0] });
            world.add(body, Shape, {
                enableContactEvents: 1,
                geometry: world
                    .resource(Hulls)
                    .add({ name: "box", halfExtents: [1.75, 0.5, 0.75] }),
            });
        },
    });
    try {
        const world = app.world;
        world.tick();
        const events = world.resource(PhysicsEvents);
        expect(events.contactBegin.count).toBe(2);
        const first = events.contactBegin.records[0]!;
        const sample = world.resource(EventSample);
        const tracked: NonNullable<typeof sample.contactPair> = [
            first.shapeA,
            first.shapeB,
            first.childIndex,
        ];
        expect(sample.contactPair).toEqual(tracked);
        setTransform(world, body, [first.childIndex === 0 ? -1.4 : 1.4, 1, 0], [0, 0, 0, 1]);
        world.tick();
        expect(events.contactEnd.count).toBe(1);
        expect(events.contactEnd.records[0]!.childIndex).not.toBe(tracked[2]);
        expect(sample.contactPair).toEqual(tracked);
        expect(sample.contactMarks.length).toBeGreaterThan(0);
        setTransform(world, body, [10, 1, 0], [0, 0, 0, 1]);
        world.tick();
        expect(sample.contactPair).toBeNull();
        expect(sample.contactMarks).toEqual([]);
    } finally {
        app.dispose();
    }
});
