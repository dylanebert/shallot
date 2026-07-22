// 4b.3a / T4 — the resident fat-AABB column mirrors each shape's fatAABB. Every TS site that writes
// `shape.fatAABB` mirrors it into the column inline (create, refit commit, CCD, moves), so the column the
// in-kernel recycle + finalize escape tests read stays current with no step-top flush. This exercises the
// whole write path end-to-end: a static shape written once at create (never refits), dynamic shapes refit
// while falling, all landing bit-for-bit in the column.

import { describe, expect, test } from "bun:test";
import { makeBoxHull } from "./hull";
import { BodyType, World } from "./index";

describe("fat-AABB column residency", () => {
    test("column mirrors shape.fatAABB across create + refits", () => {
        const world = new World();
        const box = makeBoxHull(0.5, 0.5, 0.5);

        // Static ground: its fat AABB is written once at create and never refits.
        const ground = world.createBody({ type: BodyType.Static, position: { x: 0, y: 0, z: 0 } });
        ground.createHull({}, makeBoxHull(20, 0.5, 20));

        // A stack of dynamic boxes that fall and settle, refitting their proxies as they move.
        for (let i = 0; i < 8; ++i) {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: f(1.5 + i * 1.2), z: 0 },
            });
            b.createHull({ density: 1 }, box);
        }

        const state = world.state;
        // Assert the column matches every live shape's fatAABB. The inline writes keep it current after
        // every step; just refresh the view (a grow this step may have detached it) before reading.
        const check = () => {
            state.fatAabbStore.refreshViews();
            const col = state.fatAabbStore.fatF;
            for (const shape of state.shapes) {
                const o = shape.id * 6;
                const fat = shape.fatAABB;
                expect(col[o]).toBe(fat.lowerBound.x);
                expect(col[o + 1]).toBe(fat.lowerBound.y);
                expect(col[o + 2]).toBe(fat.lowerBound.z);
                expect(col[o + 3]).toBe(fat.upperBound.x);
                expect(col[o + 4]).toBe(fat.upperBound.y);
                expect(col[o + 5]).toBe(fat.upperBound.z);
            }
        };

        for (let step = 0; step < 40; ++step) {
            world.step(1 / 60, 4);
            check();
        }

        world.destroy();
    });
});

/** exactly-representable helper so the test positions aren't f64 literals feeding the engine. */
function f(x: number): number {
    return Math.fround(x);
}
