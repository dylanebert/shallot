import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";
import { BodyType } from "../../src/standard/physics/common/types";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { readStateAngularVelocity } from "../../src/standard/physics/kernel/bodycolumns";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import { createSphereShape, createCapsuleShape, createHullShape } from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { bodyGetLinearVelocity, getBodyState } from "../../src/standard/physics/world/body";

setDefaultTimeout(0);
await init(undefined, { threads: 0 });
const bits = new Uint32Array(1);
const floats = new Float32Array(bits.buffer);
function hex(n: number): string { floats[0] = n; return bits[0]!.toString(16).padStart(8, "0"); }

test("explosion radius, falloff, masks and shape scales equal native velocity bits", () => {
    const expected = nativeSseOutput("explosion.c", "").trim().split("\n");
    const world = createTestWorld();
    try {
        const bodies = Array.from({ length: 12 }, (_, i) => {
            const body = createTestBody(world, { type: BodyType.Dynamic, position: { x: (i % 4) * 2, y: 0.25, z: Math.floor(i / 4) * 2 } });
            const def = testShapeDef({ explosionScale: Math.floor(i / 3) % 3 });
            def.filter.categoryBits = 1n << BigInt(i % 2);
            if (i % 3 === 0) createSphereShape(world, body, def, { center: { x: 0.2, y: 0, z: 0 }, radius: 0.5 });
            else if (i % 3 === 1) createCapsuleShape(world, body, def, { center1: { x: 0, y: -0.5, z: 0 }, center2: { x: 0, y: 0.5, z: 0 }, radius: 0.25 });
            else createHullShape(world, body, def, makeBoxHull(0.5, 0.75, 0.25));
            return body;
        });
        const actual: string[] = [];
        for (let j = 0; j < 3; j++) {
            kernel(world.ecsState).worldExplode(world.worldId, j === 2 ? 0 : 0xffffffff, j === 2 ? 2 : 0xffffffff, 0, 0, 0, j === 0 ? 0.5 : 3, j === 0 ? 0 : 4, j === 2 ? -3 : 12);
            for (const body of bodies) {
                const v = bodyGetLinearVelocity(world, body, { x: 0, y: 0, z: 0 });
                const w = readStateAngularVelocity(world, getBodyState(world, body)!, { x: 0, y: 0, z: 0 });
                actual.push([v.x, v.y, v.z, w.x, w.y, w.z].map(hex).join(" "));
            }
        }
        expect(actual).toEqual(expected);
    } finally { destroyTestWorld(world); }
});
