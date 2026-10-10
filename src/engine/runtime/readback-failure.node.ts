import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { controlledReadback } from "./readback.fixture";

setDefaultTimeout(CEILING.node);

test("an uncaptured GPU error rejects a pending readback with its copy context", async () => {
    await controlledReadback(async (world, slots, errors) => {
        world.step(1 / 60);
        const pending = world.readback.request(4, "counter snapshot", () => {});
        const outcome = pending.then(
            () => new Error("readback unexpectedly resolved"),
            (error: unknown) => error,
        );
        const event = new Event("uncapturederror");
        Object.defineProperty(event, "error", { value: new Error("validation failure") });
        errors.dispatchEvent(event);

        const failure = await outcome;
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toBe(
            "counter snapshot: frame 1 tick 1 readback failed: validation failure",
        );
        expect(slots[0].destroyed).toBe(true);
        expect(world.readback.allocated).toBe(0);
    });
});
