import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
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
