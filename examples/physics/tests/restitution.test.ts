import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { RestitutionState, restitutionScenes } from "../src/restitution";
import { restitutionLines } from "../src/restitution-presentation";
import type { Controls } from "../src/scenes";

test("Bounce Impulse readouts distinguish approach, measured and expected impulse after the hit", () => {
    const world = new World();
    const controls: Controls = restitutionScenes.bounceImpulse.controls;
    const read = () =>
        Object.fromEntries(
            Object.entries(controls)
                .filter(([, c]) => c.read)
                .map(([key, c]) => [key, c.read!(world)]),
        );
    expect(read()).toEqual({ Approach: null, Measured: null, Expected: null, Steps: null });
    Object.assign(world.resource(RestitutionState), {
        hit: true,
        approachSpeed: 3,
        totalImpulse: 4,
        expectedImpulse: 5,
        contactSteps: 2,
    });
    expect(read()).toEqual({ Approach: 3, Measured: 4, Expected: 5, Steps: 2 });
});
test("Restitution Propagation retains drop markers and adds only observed apex markers", () => {
    const world = new World();
    const s = world.resource(RestitutionState);
    s.scene = "restitutionPropagation";
    expect(restitutionLines(world)).toEqual([
        { a: [-5, 5.5, 0], b: [-3, 5.5, 0] },
        { a: [-1, 6.5, 0], b: [1, 6.5, 0] },
        { a: [3, 7.5, 0], b: [5, 7.5, 0] },
    ]);
    s.apex[1] = Math.fround(3.1);
    const lines = restitutionLines(world);
    expect(lines).toHaveLength(4);
    expect(lines[2]).toEqual({
        a: [-1, Math.fround(1.5 + s.apex[1]!), 0],
        b: [1, Math.fround(1.5 + s.apex[1]!), 0],
    });
});
