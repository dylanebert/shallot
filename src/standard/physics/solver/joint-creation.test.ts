import { expect, test } from "bun:test";
import { type Body, BodyType, type Joint, PhysicsWorld } from "../api/index";
import {
    DJ_IMPULSE,
    J_FORCE_THRESHOLD,
    DJ_LENGTH as J_PAYLOAD,
    JOINT_STRIDE,
    MJ_LINEAR_VELOCITY_IMPULSE,
    PJ_PERP_IMPULSE,
    PLJ_PERP_IMPULSE,
    RJ_LINEAR_IMPULSE,
    SJ_LINEAR_IMPULSE,
    WHJ_LINEAR_IMPULSE,
    WJ_LINEAR_IMPULSE,
} from "../kernel/columns";
import { kernel } from "../kernel/kernel";

const kinds: [string, (world: PhysicsWorld, a: Body, b: Body) => Joint, number][] = [
    ["distance", (w, a, b) => w.createDistanceJoint(a, b), DJ_IMPULSE],
    ["filter", (w, a, b) => w.createFilterJoint(a, b), J_PAYLOAD],
    ["motor", (w, a, b) => w.createMotorJoint(a, b), MJ_LINEAR_VELOCITY_IMPULSE],
    ["parallel", (w, a, b) => w.createParallelJoint(a, b), PLJ_PERP_IMPULSE],
    ["prismatic", (w, a, b) => w.createPrismaticJoint(a, b), PJ_PERP_IMPULSE],
    ["revolute", (w, a, b) => w.createRevoluteJoint(a, b), RJ_LINEAR_IMPULSE],
    ["spherical", (w, a, b) => w.createSphericalJoint(a, b), SJ_LINEAR_IMPULSE],
    ["weld", (w, a, b) => w.createWeldJoint(a, b), WJ_LINEAR_IMPULSE],
    ["wheel", (w, a, b) => w.createWheelJoint(a, b), WHJ_LINEAR_IMPULSE],
];

for (const [name, create, impulse] of kinds) {
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
                    JOINT_STRIDE,
                );
            };
            const first = create(world, a, b);
            const definition = Array.from(words(first).slice(J_PAYLOAD, impulse));
            expect(Array.from(words(first).slice(impulse, J_FORCE_THRESHOLD))).toEqual(
                Array(J_FORCE_THRESHOLD - impulse).fill(0),
            );
            words(first).fill(0x41200000, J_PAYLOAD, J_FORCE_THRESHOLD);
            const id = first.id.index1;
            first.destroy();
            const second = create(world, a, b);
            expect(second.id.index1).toBe(id);
            expect(Array.from(words(second).slice(J_PAYLOAD, impulse))).toEqual(definition);
            expect(Array.from(words(second).slice(impulse, J_FORCE_THRESHOLD))).toEqual(
                Array(J_FORCE_THRESHOLD - impulse).fill(0),
            );
        } finally {
            world.destroy();
        }
    });
}
