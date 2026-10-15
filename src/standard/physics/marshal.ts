import { Body, BodyMotionLock, type BodyType } from "../../core/physics";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import type { EntityId } from "./common/ids";
import { defaultBodyDef } from "./common/types";
import { createBody, makeBodyId } from "./world/body";
import type { WorldState } from "./world/world";

/** The sole ECS-to-Box3D conversion for authored body definitions. */
export function marshalBody(world: World, state: WorldState, eid: number): EntityId {
    const body = world.storage(Body);
    const pose = world.storage(Transform);
    const locks = body.motionLocks.get(eid);
    const bodyId = createBody(state, {
        ...defaultBodyDef(),
        type: body.type.get(eid) as BodyType,
        position: {
            x: pose.translation.x.get(eid),
            y: pose.translation.y.get(eid),
            z: pose.translation.z.get(eid),
        },
        rotation: {
            v: {
                x: pose.rotation.x.get(eid),
                y: pose.rotation.y.get(eid),
                z: pose.rotation.z.get(eid),
            },
            s: pose.rotation.w.get(eid),
        },
        linearVelocity: {
            x: body.linearVelocity.x.get(eid),
            y: body.linearVelocity.y.get(eid),
            z: body.linearVelocity.z.get(eid),
        },
        angularVelocity: {
            x: body.angularVelocity.x.get(eid),
            y: body.angularVelocity.y.get(eid),
            z: body.angularVelocity.z.get(eid),
        },
        linearDamping: body.linearDamping.get(eid),
        angularDamping: body.angularDamping.get(eid),
        gravityScale: body.gravityScale.get(eid),
        sleepThreshold: body.sleepThreshold.get(eid),
        motionLocks: {
            linearX: (locks & BodyMotionLock.linearX) !== 0,
            linearY: (locks & BodyMotionLock.linearY) !== 0,
            linearZ: (locks & BodyMotionLock.linearZ) !== 0,
            angularX: (locks & BodyMotionLock.angularX) !== 0,
            angularY: (locks & BodyMotionLock.angularY) !== 0,
            angularZ: (locks & BodyMotionLock.angularZ) !== 0,
        },
        enableSleep: body.enableSleep.get(eid) !== 0,
        isAwake: body.isAwake.get(eid) !== 0,
        isBullet: body.isBullet.get(eid) !== 0,
        isEnabled: body.isEnabled.get(eid) !== 0,
        allowFastRotation: body.allowFastRotation.get(eid) !== 0,
        enableContactRecycling: body.enableContactRecycling.get(eid) !== 0,
    });
    return makeBodyId(state, bodyId);
}
