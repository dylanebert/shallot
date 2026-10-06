import { expect, setDefaultTimeout, test } from "bun:test";
import { BodyType, hash, PhysicsWorld } from "../../src/standard/physics/api";
import { nativeBinary, run } from "./native";

setDefaultTimeout(180_000);

test("sleeping-set joint creation and subsequent wake match native step hashes for either survivor and ties", () => {
    const native = nativeBinary();
    for (const sizes of [[3, 4], [4, 3], [3, 3]]) {
        const expected = run([native, "sleeping_merge", "1", "5"], {
            MERGE_SIZES: sizes.join(","),
        }).trim().split("\n");
        const world = new PhysicsWorld();
        try {
            const groups = sizes.map((size, group) => {
                const bodies = Array.from({ length: size }, (_, i) => {
                    const body = world.createBody({
                        type: BodyType.Dynamic,
                        position: { x: group * 10 + i, y: 0, z: 0 },
                    });
                    body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.25 });
                    return body;
                });
                for (let i = 1; i < size; ++i)
                    world.createDistanceJoint(bodies[i - 1], bodies[i], { length: 1 });
                bodies[0].setAwake(false);
                return bodies;
            });
            world.createDistanceJoint(groups[0][1], groups[1][0], { length: 9 });
            const actual: string[] = [];
            for (let i = 0; i < 5; ++i) {
                if (i === 1) groups[1][1].applyLinearImpulseToCenter({ x: 1, y: 2, z: 3 }, true);
                world.step(1 / 60);
                actual.push(`${i} 0x${hash(world).toString(16).padStart(16, "0")}`);
            }
            expect(actual).toEqual(expected);
        } finally {
            world.destroy();
        }
    }
});
