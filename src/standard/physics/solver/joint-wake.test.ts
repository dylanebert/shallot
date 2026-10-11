import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { setSplitIslandCandidate, splitIslandCandidate } from "../kernel/islandcolumns";
import { queryColumns } from "../kernel/querycolumns";
import { createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodyIsAwake, bodySetAwake, bodySetLinearVelocity } from "../world/body";
import { createFilterJoint, defaultJointDef, destroyJointInternal } from "./joint";

function castRayClosest(
    world: ReturnType<typeof createTestWorld>,
    origin: { x: number; y: number; z: number },
    translation: { x: number; y: number; z: number },
) {
    const q = queryColumns(world);
    const k = q.prepare(origin);
    q.translation(translation);
    k.worldQuery(world.worldId, 3, 0);
    const shape = q.resultU[0]!;
    if (shape === 0xffffffff)
        return { hit: false, shape, point: { x: 0, y: 0, z: 0 }, fraction: 0 };
    return {
        hit: true,
        shape,
        point: {
            x: origin.x + q.resultF[6]!,
            y: origin.y + q.resultF[7]!,
            z: origin.z + q.resultF[8]!,
        },
        fraction: q.resultF[5]!,
    };
}

test("joint creation and destruction wake every sleeping endpoint-set body and keep their shape-query outputs current", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = [0, 10, 20].map((x) =>
            createTestBody(world, { type: BodyType.Dynamic, position: { x, y: 0, z: 0 } }),
        );
        const shapes = bodies.map(
            (body) =>
                createSphereShape(world, body, testShapeDef(), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 0.5,
                })!,
        );
        const first = createFilterJoint(world, {
            ...defaultJointDef(),
            bodyIdA: bodies[0]!,
            bodyIdB: bodies[1]!,
        }).joint;
        bodySetAwake(world, bodies[0]!, false);
        expect(bodyIsAwake(world, bodies[0]!)).toBe(false);
        expect(bodyIsAwake(world, bodies[1]!)).toBe(false);
        const connector = createFilterJoint(world, {
            ...defaultJointDef(),
            bodyIdA: bodies[1]!,
            bodyIdB: bodies[2]!,
        }).joint;
        const verify = (offset: number) => {
            for (const body of bodies) {
                expect(bodyIsAwake(world, body)).toBe(true);
                bodySetLinearVelocity(world, body, { x: 60, y: 0, z: 0 });
            }
            stepTestWorld(world, 1 / 60);
            for (let i = 0; i < bodies.length; ++i) {
                const ray = castRayClosest(
                    world,
                    { x: i * 10 + offset, y: -2, z: 0 },
                    { x: 0, y: 4, z: 0 },
                );
                expect(ray).toEqual({
                    hit: true,
                    shape: shapes[i],
                    point: { x: i * 10 + offset, y: -0.5, z: 0 },
                    fraction: 0.375,
                });
            }
        };
        verify(1);
        bodySetAwake(world, bodies[0]!, false);
        for (const body of bodies) expect(bodyIsAwake(world, body)).toBe(false);
        destroyJointInternal(world, connector, true);
        verify(2);
        destroyJointInternal(world, first, true);
    } finally {
        destroyTestWorld(world);
    }
});

test("a kernel joint merge clears a destroyed island's split candidate and snapshots retain the candidate without touching a sibling", () => {
    const world = createTestWorld();
    const sibling = createTestWorld();
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic });
        const b = createTestBody(world, { type: BodyType.Dynamic });
        const other = createTestBody(sibling, { type: BodyType.Dynamic });
        const candidate = bodyField(world, b, BodyField.islandId);
        const otherCandidate = bodyField(sibling, other, BodyField.islandId);
        setSplitIslandCandidate(world, candidate);
        setSplitIslandCandidate(sibling, otherCandidate);
        const saved = snapshot(world);
        createFilterJoint(world, { ...defaultJointDef(), bodyIdA: a, bodyIdB: b });
        expect(splitIslandCandidate(world)).toBe(-1);
        expect(splitIslandCandidate(sibling)).toBe(otherCandidate);
        restore(world, saved);
        expect(splitIslandCandidate(world)).toBe(candidate);
        expect(splitIslandCandidate(sibling)).toBe(otherCandidate);
        createFilterJoint(world, { ...defaultJointDef(), bodyIdA: a, bodyIdB: b });
        expect(splitIslandCandidate(world)).toBe(-1);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});
