import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { controlledReadback } from "./readback.fixture";

setDefaultTimeout(CEILING.node);

test("a stalled readback rejects with its copy label, frame and tick within the GPU ceiling", async () => {
    await controlledReadback(async (world, slots) => {
        world.step(1 / 60);
        const pending = world.readback.request(4, "stalled counter", () => {});
        await expect(pending).rejects.toThrow(
            "stalled counter: frame 1 tick 1 readback map exceeded 750 ms",
        );
        expect(slots[0].destroyed).toBe(true);
        expect(world.readback.allocated).toBe(0);
    });
});
