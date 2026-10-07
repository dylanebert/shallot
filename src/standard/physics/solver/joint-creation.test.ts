import { expect, test } from "bun:test";
import { type Body, BodyType, type Joint, PhysicsWorld } from "../api/index";
import * as J from "../kernel/joint-layout";
import { kernel } from "../kernel/kernel";

const range = (start: number, count: number) => Array.from({ length: count }, (_, i) => start + i);
const flagWord = (field: number) => (field & ~(1 << 16)) >>> 2;
// Definition fields moved among impulses and scratch in the typed per-kind payloads.
const kinds: [string, (world: PhysicsWorld, a: Body, b: Body) => Joint, number[]][] = [
    [
        "distance",
        (w, a, b) => w.createDistanceJoint(a, b),
        [...range(J.DJ_LENGTH, 9), flagWord(J.DJ_ENABLE)],
    ],
    ["filter", (w, a, b) => w.createFilterJoint(a, b), []],
    ["motor", (w, a, b) => w.createMotorJoint(a, b), range(J.MJ_LINEAR_VELOCITY, 14)],
    ["parallel", (w, a, b) => w.createParallelJoint(a, b), range(J.PLJ_HERTZ, 3)],
    [
        "prismatic",
        (w, a, b) => w.createPrismaticJoint(a, b),
        [...range(J.PJ_HERTZ, 7), flagWord(J.PJ_ENABLE)],
    ],
    [
        "revolute",
        (w, a, b) => w.createRevoluteJoint(a, b),
        [...range(J.RJ_HERTZ, 7), flagWord(J.RJ_ENABLE)],
    ],
    [
        "spherical",
        (w, a, b) => w.createSphericalJoint(a, b),
        [...range(J.SJ_HERTZ, 13), flagWord(J.SJ_ENABLE)],
    ],
    ["weld", (w, a, b) => w.createWeldJoint(a, b), range(J.WJ_LINEAR_HERTZ, 4)],
    [
        "wheel",
        (w, a, b) => w.createWheelJoint(a, b),
        [
            ...range(J.WHJ_MAX_SPIN_TORQUE, 2),
            ...range(J.WHJ_LOWER_SUSPENSION_LIMIT, 4),
            ...range(J.WHJ_LOWER_STEERING_LIMIT, 6),
            flagWord(J.WHJ_ENABLE),
            flagWord(J.WHJ_ENABLE) + 1,
        ],
    ],
];

for (const [name, create, fields] of kinds) {
    test(`${name} creation rebuilds its definition and zeroes impulses and solver scratch in a reused slot`, () => {
        const world = new PhysicsWorld();
        try {
            const a = world.createBody();
            const b = world.createBody({ type: BodyType.Dynamic });
            const k = kernel(world.state.ecsState);
            const words = (joint: Joint) => {
                k.bodySetActiveWorld(world.state.worldId);
                return new Uint32Array(
                    k.memory.buffer,
                    k.jointSimPtr(joint.id.index1 - 1),
                    J.JOINT_STRIDE,
                );
            };
            const scratch = range(J.J_PAYLOAD, J.JOINT_STRIDE - J.J_PAYLOAD).filter(
                (i) => !fields.includes(i),
            );
            const definition = (joint: Joint) => fields.map((i) => words(joint)[i]);
            const zeroes = (joint: Joint) => scratch.map((i) => words(joint)[i]);
            const first = create(world, a, b);
            const expectedDefinition = definition(first);
            expect(zeroes(first)).toEqual(Array(scratch.length).fill(0));
            words(first).fill(0x41200000, J.J_PAYLOAD, J.JOINT_STRIDE);
            const id = first.id.index1;
            first.destroy();
            const second = create(world, a, b);
            expect(second.id.index1).toBe(id);
            expect(definition(second)).toEqual(expectedDefinition);
            expect(zeroes(second)).toEqual(Array(scratch.length).fill(0));
        } finally {
            world.destroy();
        }
    });
}
