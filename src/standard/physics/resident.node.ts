import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import { hash } from "./api";
import { PhysicsWorld } from "./api/world";
import { moveCount } from "./collision/broadphase";
import { BodyType } from "./common/types";

setDefaultTimeout(CEILING.node);

function scene(filtered: boolean) {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    const mover = world.createBody({ type: BodyType.Kinematic, position: { x: 4, y: 0, z: 0 } });
    mover.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    const probe = world.createBody({ type: BodyType.Dynamic, position: { x: 5.25, y: 0, z: 0 } });
    probe.createSphere(
        { enableContactEvents: true },
        { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
    );
    const bodies = filtered
        ? [0, 0.25].map((x) => {
              const body = world.createBody({
                  type: BodyType.Dynamic,
                  position: { x, y: 0, z: 0 },
              });
              body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
              return body;
          })
        : [];
    if (filtered) world.createFilterJoint(bodies[0], bodies[1]);
    world.step(1 / 60);
    for (let i = 0; i < bodies.length; i++) {
        const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };
        bodies[i].setTransform({ x: i * 0.25 + 2, y: 0, z: 0 }, rotation);
        bodies[i].setTransform({ x: i * 0.25, y: 0, z: 0 }, rotation);
    }
    mover.setTransform({ x: 5, y: 0, z: 0 }, { v: { x: 0, y: 0, z: 0 }, s: 1 });
    return world;
}

for (const filtered of [false, true]) {
    test(`empty World construction preserves the owner's pending proxies and moves${filtered ? " and non-colliding joint pair" : ""}`, () => {
        const solo = scene(filtered);
        let expected: bigint;
        try {
            solo.step(1 / 60);
            expect(solo.getContactEvents().beginEvents.length).toBe(1);
            expected = hash(solo);
        } finally {
            solo.destroy();
        }
        const a = scene(filtered);
        let b: PhysicsWorld | undefined;
        try {
            expect(a.castRayClosest({ x: 5, y: 2, z: 0 }, { x: 0, y: -4, z: 0 }).hit).toBe(true);
            b = new PhysicsWorld();
            expect(a.castRayClosest({ x: 5, y: 2, z: 0 }, { x: 0, y: -4, z: 0 }).hit).toBe(true);
            a.step(1 / 60);
            // The pending mover must discover the previously separated probe, while the
            // overlapping joint-filtered pair must remain non-colliding.
            expect(a.getContactEvents().beginEvents.length).toBe(1);
            expect(hash(a)).toBe(expected);
        } finally {
            // B has never claimed or stepped; this does not assert interleaved stepping support.
            b?.destroy();
            a.destroy();
        }
    });
}

function ray(world: PhysicsWorld) {
    const { hit, fraction, point, normal } = world.castRayClosest(
        { x: 0, y: 2, z: 0 },
        { x: 0, y: -4, z: 0 },
    );
    return { hit, fraction, point, normal };
}

for (const claimed of [false, true]) {
    for (const previouslyOwned of [false, true]) {
        test(`snapshot reinstates resident initialization, proxies, moves, ray and tick hashes: ${claimed ? "claimed" : "unclaimed"}, ${previouslyOwned ? "previous owner" : "fresh kernel"}`, () => {
            const owner = new World();
            if (previouslyOwned) {
                const prior = new PhysicsWorld({}, owner);
                const body = prior.createBody({ type: BodyType.Dynamic });
                body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
                prior.step(1 / 60);
                prior.destroy();
            }
            const source = new PhysicsWorld(
                { gravity: { x: 0, y: 0, z: 0 }, enableSleep: false },
                owner,
            );
            if (claimed) {
                const bodies = [0, 0.25, 4, 4.25].map((x) => {
                    const body = source.createBody({
                        type: BodyType.Dynamic,
                        position: { x, y: 0, z: 0 },
                    });
                    body.createSphere(
                        { enableContactEvents: true },
                        { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                    );
                    return body;
                });
                source.createFilterJoint(bodies[2], bodies[3]);
            }
            // Capture before a query or step can claim an empty source.
            const saved = source.snapshot();
            const proxies = source.state.broadPhase.trees.map((tree) => tree.proxyCount);
            const moves = moveCount(source.state.broadPhase);
            expect(proxies).toEqual([0, 0, claimed ? 4 : 0]);
            expect(moves).toBe(claimed ? 4 : 0);
            const expectedRay = ray(source);
            expect(expectedRay.hit).toBe(claimed);
            const expected: bigint[] = [];
            try {
                for (let tick = 0; tick < 12; tick++) {
                    source.step(1 / 60);
                    if (tick === 0)
                        expect(source.getContactEvents().beginEvents.length).toBe(claimed ? 1 : 0);
                    expected.push(hash(source));
                }
            } finally {
                source.destroy();
            }
            const target = new PhysicsWorld({}, owner);
            try {
                target.restore(saved);
                expect(target.state.broadPhase.trees.map((tree) => tree.proxyCount)).toEqual(
                    proxies,
                );
                expect(moveCount(target.state.broadPhase)).toBe(moves);
                expect(ray(target)).toEqual(expectedRay);
                for (let tick = 0; tick < expected.length; tick++) {
                    target.step(1 / 60);
                    expect(hash(target), `tick ${tick}`).toBe(expected[tick]);
                    if (tick === 0)
                        expect(target.getContactEvents().beginEvents.length).toBe(claimed ? 1 : 0);
                }
            } finally {
                target.destroy();
            }
        });
    }
}
