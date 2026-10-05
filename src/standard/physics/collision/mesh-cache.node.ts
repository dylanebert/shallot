import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../../scripts/test-tiers";
import { BodyType, createMesh, hash, PhysicsWorld } from "../api";
import { kernel } from "../kernel/kernel";

setDefaultTimeout(CEILING.node);

test("mesh contact cache replays through sleep, wake, contact destruction and recycled ids", () => {
    const world = new PhysicsWorld();
    const k = kernel(world.state.ecsState);
    const id = world.state.worldId;
    try {
        const ground = world.createBody();
        ground.createMesh(
            {},
            createMesh({
                vertices: [
                    { x: -4, y: 0, z: -4 },
                    { x: 4, y: 0, z: -4 },
                    { x: 4, y: 0, z: 4 },
                    { x: -4, y: 0, z: 4 },
                ],
                indices: [0, 2, 1, 0, 3, 2],
            })!,
        );
        const spawn = () => {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: 0.5, z: 0 },
            });
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            return body;
        };
        const body = spawn();
        for (let i = 0; i < 30; ++i) world.step(1 / 60);
        expect(world.state.contacts.some((c) => c !== null && c.manifoldCount > 0)).toBe(true);
        expect(k.meshCacheCapacity(id)).toBeGreaterThan(0);
        const saved = world.snapshot();
        const run = () => {
            const hashes: bigint[] = [];
            body.setAwake(false);
            for (let i = 0; i < 3; ++i) {
                world.step(1 / 60);
                hashes.push(hash(world));
            }
            body.setAwake(true);
            for (let i = 0; i < 3; ++i) {
                world.step(1 / 60);
                hashes.push(hash(world));
            }
            body.destroy();
            spawn();
            for (let i = 0; i < 10; ++i) {
                world.step(1 / 60);
                hashes.push(hash(world));
            }
            return hashes;
        };
        const expected = run();
        world.restore(saved);
        expect(run()).toEqual(expected);
    } finally {
        world.destroy();
    }
    expect(k.meshCacheCapacity(id)).toBe(0);
});
