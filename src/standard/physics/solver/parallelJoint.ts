import { FLT_MAX, f32, type Quat, quat, type Vec3, vec3 } from "../common/math";
import {
    PLJ_DAMPING_RATIO,
    PLJ_HERTZ,
    PLJ_MAX_TORQUE,
    PLJ_PERP_AXIS_X,
    PLJ_PERP_AXIS_Y,
    PLJ_PERP_IMPULSE,
    PLJ_QUAT_A,
    PLJ_QUAT_B,
} from "../kernel/columns";
import {
    readJointQuat,
    readJointVec2,
    readJointVec3,
    writeJointFloat,
    writeJointQuat,
    writeJointVec2,
    writeJointVec3,
} from "../kernel/jointcolumns";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Parallel joint payload (b3ParallelJoint). Impulse persists across steps for warm starting. */

/** Parallel joint definition (b3ParallelJointDef), body handles resolved to a base JointDef. */
export type ParallelJointDef = {
    base: JointDef;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** @returns the ported parallel joint definition defaults (b3DefaultParallelJointDef). */
export function defaultParallelJointDef(base: JointDef): ParallelJointDef {
    return {
        base,
        hertz: 1,
        dampingRatio: 1,
        maxTorque: FLT_MAX,
    };
}
const identityQuat = (): Quat => ({
    v: {
        x: 0,
        y: 0,
        z: 0,
    },
    s: 1,
});

/** Create a parallel joint (b3CreateParallelJoint). @returns the joint handle. */
export function createParallelJoint(
    world: WorldState,
    def: ParallelJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Parallel);
    writeJointVec2(world, pair.joint, PLJ_PERP_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointFloat(world, pair.joint, PLJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, PLJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, PLJ_MAX_TORQUE, def.maxTorque);
    writeJointQuat(world, pair.joint, PLJ_QUAT_A, identityQuat());
    writeJointQuat(world, pair.joint, PLJ_QUAT_B, identityQuat());
    writeJointVec3(world, pair.joint, PLJ_PERP_AXIS_X, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec3(world, pair.joint, PLJ_PERP_AXIS_Y, {
        x: 0,
        y: 0,
        z: 0,
    });
    return pair;
}

// The two perpendicular collinearity axes in world space, from the relative rotation (relQ) of the
// two joint frames. relQ = inv(quatA) * quatB; the axes are half the rotated imaginary parts.
function perpAxes(
    qA: Quat,
    relQ: Quat,
): {
    x: Vec3;
    y: Vec3;
} {
    return {
        x: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
            ),
        ),
        y: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
            ),
        ),
    };
}
export function getParallelJointTorque(world: WorldState, sim: Joint): Vec3 {
    const relQ = quat.invMul(
        readJointQuat(world, sim, PLJ_QUAT_A),
        readJointQuat(world, sim, PLJ_QUAT_B),
    );
    const axes = perpAxes(readJointQuat(world, sim, PLJ_QUAT_A), relQ);
    writeJointVec3(world, sim, PLJ_PERP_AXIS_X, axes.x);
    writeJointVec3(world, sim, PLJ_PERP_AXIS_Y, axes.y);
    const angularImpulse = vec3.blend2(
        readJointVec2(world, sim, PLJ_PERP_IMPULSE).x,
        readJointVec3(world, sim, PLJ_PERP_AXIS_X),
        readJointVec2(world, sim, PLJ_PERP_IMPULSE).y,
        readJointVec3(world, sim, PLJ_PERP_AXIS_Y),
    );
    return vec3.scale(world.invH, angularImpulse);
}
