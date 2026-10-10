import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { controlledReadback } from "./readback.fixture";

setDefaultTimeout(CEILING.node);

test("a stalled non-real-adapter readback rejects with its copy label, frame and tick", async () => {
    await controlledReadback(async (world, slots) => {
        world.step(1 / 60);
        const pending = world.readback.request(4, "stalled counter", () => {});
        await expect(pending).rejects.toThrow(
            "stalled counter: frame 1 tick 1 readback map exceeded 2000 ms",
        );
        expect(slots[0].destroyed).toBe(true);
        expect(world.readback.allocated).toBe(0);
    });
});
