import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { contactIds, createContact, destroyContact } from "../collision/contact";
import { BodyType } from "../common/types";
import { readStateAngularVelocity } from "../kernel/bodycolumns";
import { physicsState } from "../runtime";
import { makeBoxHull } from "../shapes/hull";
import { createHullShape } from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef } from "../solver/joint";
import { createTestBody, testShapeDef } from "../testing";
import { PhysicsWorldDefinition } from "../world-definition";
import {
    bodyApplyAngularImpulse,
    bodyApplyForce,
    bodyApplyForceToCenter,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyApplyTorque,
    bodyDisable,
    bodyEnable,
    bodyGetLinearVelocity,
    bodyGetTransform,
    bodySetAngularVelocity,
    bodySetAwake,
    bodySetLinearVelocity,
    bodySetTargetTransform,
    bodySetTransform,
    getBodyState,
} from "./body";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const state = physicsState(app.world)!;
    app.world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const a = createTestBody(state, { type: BodyType.Dynamic });
    const b = createTestBody(state, { type: BodyType.Dynamic, position: { x: 0.9, y: 0, z: 0 } });
    const shapeA = createHullShape(
        state,
        a,
        testShapeDef({ density: 1, enableContactEvents: false }),
        hull,
    )!;
    const shapeB = createHullShape(
        state,
        b,
        testShapeDef({ density: 1, enableContactEvents: false }),
        hull,
    )!;
    const sleeper = createTestBody(state, {
        type: BodyType.Dynamic,
        position: { x: 10, y: 0, z: 0 },
    });
    createHullShape(state, sleeper, testShapeDef({ density: 1 }), hull);
    const partner = createTestBody(state, {
        type: BodyType.Dynamic,
        position: { x: 11, y: 0, z: 0 },
    });
    createDistanceJoint(
        state,
        defaultDistanceJointDef({ ...defaultJointDef(), bodyIdA: sleeper, bodyIdB: partner }),
    );
    app.world.step(1 / 60);
    const ids = contactIds(state);
    if (ids.length !== 1) throw new Error("body allocation requires one contact");
    const contact = ids[0];
    const first = shapeA;
    const second = shapeB;
    const velocity = { x: 0.1, y: 0.2, z: 0.3 };
    const angular = { x: 0.2, y: 0.3, z: 0.4 };
    const force = { x: 0.1, y: 0.1, z: 0.1 };
    const impulse = { x: 0.001, y: 0.001, z: 0.001 };
    const point = { x: 0.2, y: 0, z: 0 };
    const pose = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const readVelocity = { x: 0, y: 0, z: 0 };
    const allocating = input === "allocating";
    return {
        step: () => {
            for (let i = 0; i < 16; ++i) {
                bodySetLinearVelocity(state, a, velocity);
                bodySetAngularVelocity(state, a, angular);
                bodyGetLinearVelocity(state, a, readVelocity);
                readStateAngularVelocity(state, getBodyState(state, a)!, readVelocity);
                bodyGetTransform(state, a, pose);
                bodySetTransform(state, a, pose.p, pose.q);
                bodySetTargetTransform(state, a, pose, 1 / 60, false);
                bodyApplyForce(state, a, force, point, true);
                bodyApplyForceToCenter(state, a, force, true);
                bodyApplyTorque(state, a, force, true);
                bodyApplyLinearImpulse(state, a, impulse, point, true);
                bodyApplyLinearImpulseToCenter(state, a, impulse, true);
                bodyApplyAngularImpulse(state, a, impulse, true);
                bodySetAwake(state, sleeper, false);
                bodySetAwake(state, sleeper, true);
                bodyDisable(state, sleeper);
                bodyEnable(state, sleeper);
                destroyContact(state, contact, false);
                createContact(state, first, second, -1);
                if (allocating)
                    controlSink = {
                        velocity: bodyGetLinearVelocity(state, a, { x: 0, y: 0, z: 0 }),
                    };
            }
        },
        dispose: () => app.dispose(),
    };
}
