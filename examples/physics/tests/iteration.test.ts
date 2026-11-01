import { expect, test } from "bun:test";
import type { World } from "@dylanebert/shallot";
import { scenes } from "../src/scenes";
import { advanceIteration } from "./iteration";

test("a sample iteration defaults to one tick", () => {
    let ticks = 0;
    advanceIteration({ tick: () => ticks++ } as unknown as World, "boxStack");
    expect(ticks).toBe(1);
});
test("a scene declares completion of a multi-tick iteration", () => {
    let ticks = 0;
    const entry = scenes.boxStack as typeof scenes.boxStack & {
        iterationComplete?: (world: World) => boolean;
    };
    entry.iterationComplete = () => ticks % 2 === 0;
    try {
        const world = { tick: () => ticks++ } as unknown as World;
        advanceIteration(world, "boxStack");
        expect(ticks).toBe(2);
        advanceIteration(world, "boxStack");
        expect(ticks).toBe(4);
    } finally {
        delete entry.iterationComplete;
    }
});
test("a stalled iteration fails with its scene name after four ticks", () => {
    let ticks = 0;
    const entry = scenes.boxStack as typeof scenes.boxStack & {
        iterationComplete?: (world: World) => boolean;
    };
    entry.iterationComplete = () => false;
    try {
        expect(() =>
            advanceIteration({ tick: () => ticks++ } as unknown as World, "boxStack"),
        ).toThrow("boxStack: sample iteration did not complete within 4 ticks");
        expect(ticks).toBe(4);
    } finally {
        delete entry.iterationComplete;
    }
});
