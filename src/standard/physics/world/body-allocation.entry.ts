import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { contactIds, createContact, destroyContact } from "../collision/contact";
import { BodyType } from "../common/types";
import { readStateAngularVelocity } from "../kernel/bodycolumns";
import { makeBoxHull } from "../shapes/hull";
import { bodyDisable, bodyEnable, getBodyState } from "./body";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const physics = physicsWorld(app.world)!;
    physics.setGravity({ x: 0, y: 0, z: 0 });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const a = physics.createBody({ type: BodyType.Dynamic });
    const b = physics.createBody({ type: BodyType.Dynamic, position: { x: 0.9, y: 0, z: 0 } });
    const shapeA = a.createHull({ density: 1, enableContactEvents: false }, hull);
    const shapeB = b.createHull({ density: 1, enableContactEvents: false }, hull);
    const sleeper = physics.createBody({ type: BodyType.Dynamic, position: { x: 10, y: 0, z: 0 } });
    sleeper.createHull({ density: 1 }, hull);
    const partner = physics.createBody({ type: BodyType.Dynamic, position: { x: 11, y: 0, z: 0 } });
    physics.createDistanceJoint(sleeper, partner, { length: 1 });
    app.world.step(1 / 60);
    const state = physics.state;
    const ids = contactIds(state);
    if (ids.length !== 1) throw new Error("body allocation requires one contact");
    const contact = ids[0];
    const first = shapeA.id.index1 - 1;
    const second = shapeB.id.index1 - 1;
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
                a.setLinearVelocity(velocity);
                a.setAngularVelocity(angular);
                a.getLinearVelocity(readVelocity);
                readStateAngularVelocity(
                    state,
                    getBodyState(state, a.id.index1 - 1)!,
                    readVelocity,
                );
                a.getTransform(pose);
                a.setTransform(pose.p, pose.q);
                a.setTargetTransform(pose, 1 / 60);
                a.applyForce(force, point, true);
                a.applyForceToCenter(force, true);
                a.applyTorque(force, true);
                a.applyLinearImpulse(impulse, point, true);
                a.applyLinearImpulseToCenter(impulse, true);
                a.applyAngularImpulse(impulse, true);
                sleeper.setAwake(false);
                sleeper.setAwake(true);
                bodyDisable(state, sleeper.id.index1 - 1);
                bodyEnable(state, sleeper.id.index1 - 1);
                destroyContact(state, contact, false);
                createContact(state, first, second, -1);
                if (allocating) controlSink = { velocity: a.getLinearVelocity() };
            }
        },
        dispose: () => app.dispose(),
    };
}
