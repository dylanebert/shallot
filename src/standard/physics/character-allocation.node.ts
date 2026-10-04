import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

for (const [claim, path, input] of [
    [
        "a warm fixed step with a character allocates no JavaScript heap",
        "character-allocation.entry.ts",
        "",
    ],
    [
        "warm published plane solver calls with out allocate no JavaScript heap",
        "collision/mover-allocation.entry.ts",
        "",
    ],
    [
        "warm Character SolveMove with dynamic push allocates no JavaScript heap before the rigid solver",
        "character-allocation.entry.ts",
        "push",
    ],
]) {
    test(claim, async () => {
        const sample = await sampleAllocation(resolve(import.meta.dir, path), {
            // Pose-driven collider refits need a longer warmup than the isolated solvers.
            warm: path === "collision/mover-allocation.entry.ts" ? 1800 : 12000,
            input,
        });
        expect(sample.control.length).toBeGreaterThan(0);
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    });
}
