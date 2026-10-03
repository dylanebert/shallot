import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../../scripts/test-tiers";
import { World } from "../../../engine";
import { BodyType, PhysicsWorld } from "../api/index";
import { init, shutdown } from "../kernel/kernel";
import { shouldBodiesCollide } from "./pairs";

setDefaultTimeout(CEILING.node);

test("fast bodies and bullets pass static shapes jointed without collideConnected, but stop when enabled", () => {
    for (const isBullet of [false, true]) {
        for (const collideConnected of [false, true]) {
            const world = new PhysicsWorld();
            try {
                world.setGravity({ x: 0, y: 0, z: 0 });
                const target = world.createBody();
                target.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
                const fast = world.createBody({
                    type: BodyType.Dynamic,
                    position: { x: -4, y: 0, z: 0 },
                    linearVelocity: { x: 480, y: 0, z: 0 },
                    isBullet,
                });
                fast.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.25 });
                const free = world.createBody({
                    type: BodyType.Dynamic,
                    position: { x: -4, y: 10, z: 0 },
                    linearVelocity: { x: 480, y: 0, z: 0 },
                    isBullet,
                });
                free.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.25 });
                world.createFilterJoint(fast, target, { collideConnected });
                world.step(1 / 60, 1);
                expect(free.getPosition().x).toBeGreaterThan(1);
                if (collideConnected) expect(fast.getPosition().x).toBeLessThan(-1);
                else expect(fast.getPosition().x).toBe(free.getPosition().x);
            } finally {
                world.destroy();
            }
        }
    }
});

test("direct joint filters survive parallel joints, toggles, destroy, growth and snapshot restore without filtering transitive neighbors", async () => {
    const owner = new World();
    await init(owner, { threads: 0 });
    const world = new PhysicsWorld({}, owner);
    try {
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody();
        const c = world.createBody();
        const allows = (x: typeof a, y: typeof a) =>
            shouldBodiesCollide(
                world.state,
                world.state.bodies[x.id.index1 - 1],
                world.state.bodies[y.id.index1 - 1],
            );
        expect(allows(a, b)).toBe(true);
        const first = world.createFilterJoint(a, b);
        const second = world.createFilterJoint(b, a);
        world.createFilterJoint(b, c);
        world.createFilterJoint(a, c, { collideConnected: true });
        expect(allows(a, b)).toBe(false);
        expect(allows(b, a)).toBe(false);
        expect(allows(a, c)).toBe(true);
        first.setCollideConnected(true);
        first.setCollideConnected(true);
        expect(allows(a, b)).toBe(false);
        second.destroy();
        expect(allows(a, b)).toBe(true);
        first.setCollideConnected(false);
        expect(allows(a, b)).toBe(false);
        const saved = world.snapshot();
        first.destroy();
        expect(allows(a, b)).toBe(true);
        // Cross the resident column's capacity, relocating the existing tree/set columns too.
        const others = Array.from({ length: 40 }, () => world.createBody());
        for (const body of others.reverse()) world.createFilterJoint(a, body);
        for (const body of others) expect(allows(a, body)).toBe(false);
        expect(allows(a, b)).toBe(true);
        world.restore(saved);
        expect(allows(a, b)).toBe(false);
        expect(allows(a, c)).toBe(true);
        first.setCollideConnected(true);
        expect(allows(a, b)).toBe(true);
        first.setCollideConnected(false);
        b.destroy();
        const replacement = world.createBody();
        expect(allows(a, replacement)).toBe(true);
        expect(allows(c, replacement)).toBe(false);
    } finally {
        world.destroy();
        await shutdown(owner);
        owner.dispose();
    }
});
