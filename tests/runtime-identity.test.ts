import { expect, spyOn, test } from "bun:test";
import { Physics, State } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { linearToSrgb1 } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import { State as CanonicalState } from "../src/engine";
import { installHarness as canonicalHarness } from "../src/harness/runtime";
import { Physics as CanonicalPhysics } from "../src/standard/physics";

test("public development entries share canonical declarations, values and physics effects", () => {
    expect(State, "one executable State definition").toBe(CanonicalState);
    expect(Physics, "one mutable physics singleton").toBe(CanonicalPhysics);
    expect(installHarness).toBe(canonicalHarness);
    const state: CanonicalState = new State();
    const publicState: State = state;
    let y = 2;
    const read = spyOn(CanonicalPhysics, "readBody").mockImplementation(() => ({
        pos: [1, y, 3],
        quat: [0, 0, 0, 1],
        vel: [0, 3, 0],
    }));
    try {
        const harness = installHarness(publicState);
        expect(harness.read!(1)?.pos[1]).toBe(2);
        y = 5;
        expect(harness.read!(1)?.pos[1]).toBe(5);
        state.step();
        expect(harness.ready).toBe(true);
    } finally {
        read.mockRestore();
        state.dispose();
    }
});

test("public TGSL executes and resolves through the root preload", () => {
    expect(linearToSrgb1(0.003)).toBeCloseTo(0.03876, 6);
    const code = tgpu.resolve([linearToSrgb1]);
    expect(code).toContain("12.92");
});
