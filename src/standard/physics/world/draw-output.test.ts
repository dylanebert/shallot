import { expect, test } from "bun:test";
import type { AABB, WorldTransform } from "../common/math";
import { kernel } from "../kernel/kernel";
import type { Sphere } from "../shapes/geometry";
import { defaultDebugDraw } from "./draw";
import expected from "./draw-output.gold.json";
import reactions from "./draw-reactions.gold.json";
import { drawScene, reactionScene } from "./draw-scene";

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

test("stepped joint reactions and the complete callback stream match the archived pre-kernel walk", () => {
    const { world, draw } = reactionScene();
    const calls: unknown[][] = [];
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
    try {
        world.draw(draw);
        const labels = calls.filter(
            (call) => call[0] === "drawString" && String(call[2]).startsWith("f = "),
        );
        expect(labels.length).toBe(7);
        expect(labels.some((call) => Number(/f = ([^,]+)/.exec(String(call[2]))![1]) > 0)).toBe(
            true,
        );
        expect(labels.some((call) => Number(/t = (.+)/.exec(String(call[2]))![1]) > 0)).toBe(true);
        expect(
            JSON.parse(
                JSON.stringify(calls, (_key, value) =>
                    typeof value === "bigint" ? `${value}n` : value,
                ),
            ),
        ).toEqual(reactions);
    } finally {
        world.destroy();
    }
});

test("a shapes-only nested draw from the first callback preserves the full outer stream", () => {
    const { world, draw } = drawScene();
    const nested = defaultDebugDraw();
    nested.drawShapes = true;
    const calls: unknown[] = [];
    let entered = false;
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
            if (!entered) {
                entered = true;
                world.draw(nested);
            }
        };
    }
    try {
        world.draw(draw);
        expect(calls.length).toBe(expected.length);
        expect(
            JSON.parse(
                JSON.stringify(calls, (_key, value) =>
                    typeof value === "bigint" ? `${value}n` : value,
                ),
            ),
        ).toEqual(expected);
    } finally {
        world.destroy();
    }
});

test("a throwing nested callback releases both pending streams", () => {
    const { world, draw } = drawScene();
    const nested = defaultDebugDraw();
    nested.drawShapes = true;
    const error = new Error("nested draw callback");
    nested.drawSolidHull = () => {
        throw error;
    };
    draw.drawSolidHull = () => {
        world.draw(nested);
    };
    const k = kernel(world.state.ecsState);
    try {
        expect(() => world.draw(draw)).toThrow(error);
        expect(k.worldDrawLen()).toBe(0);
        expect(() => world.draw(defaultDebugDraw())).not.toThrow();
        expect(k.worldDrawLen()).toBe(0);
    } finally {
        world.destroy();
    }
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
