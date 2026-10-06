import { expect, test } from "bun:test";
import type { AABB, WorldTransform } from "../common/math";
import type { Sphere } from "../shapes/geometry";
import expected from "./draw-output.gold.json";
import { drawScene } from "./draw-scene";

test("debug draw callback order and values match the pre-kernel walk", () => {
    const { world, draw } = drawScene();
    const calls: unknown[] = [];
    for (const key of [
        "drawSolidSphere",
        "drawSolidCapsule",
        "drawSolidHull",
        "drawSolidMesh",
        "drawSolidHeightField",
        "drawSegment",
        "drawPoint",
        "drawTransform",
        "drawAabb",
        "drawString",
    ] as const) {
        draw[key] = (...args: unknown[]) => {
            calls.push(structuredClone([key, ...args]));
        };
    }
    world.draw(draw);
    expect(
        JSON.parse(
            JSON.stringify(calls, (_key, value) =>
                typeof value === "bigint" ? `${value}n` : value,
            ),
        ),
    ).toEqual(expected);
    world.destroy();
});

test("draw snapshots flags and bounds and hands out fresh callback values", () => {
    const { world, draw } = drawScene();
    const bounds: AABB[] = [];
    const frames: WorldTransform[] = [];
    const spheres: Sphere[] = [];
    draw.drawSolidSphere = (t, s) => {
        frames.push(t);
        spheres.push(s);
        draw.drawBounds = false;
        draw.drawingBounds.lowerBound.x = 1000;
    };
    draw.drawAabb = (aabb) => {
        bounds.push(aabb);
    };
    world.draw(draw);
    expect(bounds.length).toBe(6);
    expect(new Set(bounds).size).toBe(bounds.length);
    const center = spheres[0].center.x;
    spheres[0].center.x = 500;
    frames[0].p.x = 500;
    draw.drawBounds = true;
    draw.drawingBounds.lowerBound.x = -100;
    world.draw(draw);
    expect(spheres[1].center.x).toBe(center);
    expect(frames[1].p.x).toBe(1);
    world.destroy();
});
