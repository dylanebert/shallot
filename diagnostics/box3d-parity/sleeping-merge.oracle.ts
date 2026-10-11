import { expect, setDefaultTimeout, test } from "bun:test";
import { BodyType } from "../../src/standard/physics/common/types";
import { createSphereShape } from "../../src/standard/physics/shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../../src/standard/physics/solver/distanceJoint";
import { bodyApplyLinearImpulseToCenter, bodySetAwake } from "../../src/standard/physics/world/body";
import { defaultJointDef } from "../../src/standard/physics/solver/joint";
import { hashWorldState } from "../../src/standard/physics/world/hash";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { nativeBinary, run } from "./native";

setDefaultTimeout(180_000);

test("sleeping-set joint creation and subsequent wake match native step hashes for either survivor and ties", () => {
    const native = nativeBinary();
    for (const sizes of [[3, 4], [4, 3], [3, 3]]) {
        const expected = run([native, "sleeping_merge", "1", "5"], {
            MERGE_SIZES: sizes.join(","),
        }).trim().split("\n");
        const world = createTestWorld();
        try {
            const groups = sizes.map((size, group) => {
                const bodies = Array.from({ length: size }, (_, i) => {
                    const body = createTestBody(world, {
                        type: BodyType.Dynamic,
                        position: { x: group * 10 + i, y: 0, z: 0 },
                    });
                    createSphereShape(world, body, testShapeDef(), {
                        center: { x: 0, y: 0, z: 0 },
                        radius: 0.25,
                    });
                    return body;
                });
                for (let i = 1; i < size; ++i)
                    createDistanceJoint(
                        world,
                        defaultDistanceJointDef({
                            ...defaultJointDef(),
                            bodyIdA: bodies[i - 1]!,
                            bodyIdB: bodies[i]!,
                        }),
                    );
                bodySetAwake(world, bodies[0]!, false);
                return bodies;
            });
            createDistanceJoint(
                world,
                Object.assign(
                    defaultDistanceJointDef({
                        ...defaultJointDef(),
                        bodyIdA: groups[0]![1]!,
                        bodyIdB: groups[1]![0]!,
                    }),
                    { length: 9 },
                ),
            );
            const actual: string[] = [];
            for (let i = 0; i < 5; ++i) {
                if (i === 1)
                    bodyApplyLinearImpulseToCenter(world, groups[1]![1]!, { x: 1, y: 2, z: 3 }, true);
                stepTestWorld(world, 1 / 60, 4);
                actual.push(`${i} 0x${hashWorldState(world).toString(16).padStart(16, "0")}`);
            }
            expect(actual).toEqual(expected);
        } finally {
            destroyTestWorld(world);
        }
    }
});
