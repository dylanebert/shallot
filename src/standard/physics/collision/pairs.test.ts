import { afterAll, beforeAll, expect, test } from "bun:test";
import { BodyType, init, makeBoxHull, PhysicsWorld, shutdown } from "../api/index";
import { defaultFilter, type Filter } from "../common/types";

beforeAll(async () => {
    await init(undefined);
});
afterAll(() => shutdown(undefined));

function touches(a: Filter, b: Filter): boolean {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        for (const filter of [a, b]) {
            world
                .createBody({ type: BodyType.Dynamic })
                .createSphere(
                    { filter, enableContactEvents: true },
                    { center: { x: 0, y: 0, z: 0 }, radius: 1 },
                );
        }
        world.step(1 / 60);
        return world.getContactEvents().beginEvents.length !== 0;
    } finally {
        world.destroy();
    }
}
const filter = (categoryBits: bigint, maskBits: bigint, groupIndex = 0): Filter => ({
    categoryBits,
    maskBits,
    groupIndex,
});

test("pair filtering requires both masks, preserves both u64 halves, and gives shared nonzero groups precedence", () => {
    expect(touches(defaultFilter(), defaultFilter())).toBe(true);
    expect(touches(filter(1n, 2n), filter(2n, 2n))).toBe(false);
    expect(touches(filter(1n, 1n), filter(2n, 2n))).toBe(false);
    expect(touches(filter(1n, 2n), filter(2n, 1n))).toBe(true);
    const high = 1n << 40n;
    expect(touches(filter(high, high), filter(high, high))).toBe(true);
    expect(touches(filter(high, high), filter(1n << 8n, 1n << 8n))).toBe(false);
    const top = (1n << 63n) | (1n << 31n);
    expect(touches(filter(top, top), filter(top, top))).toBe(true);
    expect(touches(filter(1n, 1n, 7), filter(2n, 2n, 7))).toBe(true);
    expect(touches(filter(255n, 255n, -3), filter(255n, 255n, -3))).toBe(false);
    expect(touches(filter(1n, 1n, 1), filter(2n, 2n, 2))).toBe(false);
});

test("contact pair creation from the broad phase emits an overlapping pair once per moved side instead of deduplicating it, or re-creates a pair already in the pair set on a later step, so a single touching pair fires two begin-touch events", () => {
    // Two dynamic boxes overlap along x and both drift +y (zero gravity, sleep off) so both sit in
    // the move buffer every step. Step 1: both moved, the pair is found from both sides — dedup
    // must emit it exactly once. Step 2: the pair persists in the pair set, so the re-query must
    // reject it — no duplicate contact, no new begin.
    const physicsWorld = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    const drift = { x: 0, y: 0.5, z: 0 };

    const a = physicsWorld.createBody({
        type: BodyType.Dynamic,
        position: { x: -0.25, y: 0, z: 0 },
        linearVelocity: drift,
    });
    a.createHull({ enableContactEvents: true }, makeBoxHull(0.5, 0.5, 0.5));
    const b = physicsWorld.createBody({
        type: BodyType.Dynamic,
        position: { x: 0.25, y: 0, z: 0 },
        linearVelocity: drift,
    });
    b.createHull({ enableContactEvents: true }, makeBoxHull(0.5, 0.5, 0.5));

    let beginTotal = 0;
    let endTotal = 0;

    physicsWorld.step(1 / 60);
    let ev = physicsWorld.getContactEvents();
    beginTotal += ev.beginEvents.length;
    endTotal += ev.endEvents.length;
    // Dedup: the overlapping pair emitted exactly once, not once per moved side.
    expect(beginTotal).toBe(1);

    physicsWorld.step(1 / 60);
    ev = physicsWorld.getContactEvents();
    beginTotal += ev.beginEvents.length;
    endTotal += ev.endEvents.length;
    // Membership: the pair persisted, so step 2 creates no duplicate contact and fires no new
    // begin; the overlap is deep enough that they stay touching (no end event).
    expect(beginTotal).toBe(1);
    expect(endTotal).toBe(0);

    physicsWorld.destroy();
});
