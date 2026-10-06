import { expect, test } from "bun:test";
import { BodyType, defaultSurfaceMaterial, makeBoxHull, PhysicsWorld } from "../api";

function output(): unknown[] {
    const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    const records: unknown[] = [];
    const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 0.5 };
    const frame = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    try {
        world.createBody({ position: { x: 0, y: -0.5, z: 0 } }).createHull(
            {
                baseMaterial: {
                    ...defaultSurfaceMaterial(),
                    userMaterialId: 0x123456789abcdef0n,
                },
            },
            makeBoxHull(10, 0.5, 10),
        );
        const falling = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            userData: "falling",
        });
        falling.createSphere(
            { density: 1, enableContactEvents: true, enableHitEvents: true },
            sphere,
        );
        const anchor = world.createBody({ position: { x: 30, y: 5, z: 0 } });
        const bob = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 30, y: 5, z: 0 },
            userData: "bob",
        });
        bob.createSphere({ density: 1 }, sphere);
        const joint = world.createWeldJoint(anchor, bob, {
            localFrameA: frame,
            localFrameB: frame,
            forceThreshold: 0,
            userData: "joint",
        });
        for (let step = 0; step < 180; ++step) {
            if (step === 90) falling.destroy();
            if (step === 120) joint.destroy();
            world.step(1 / 60, 4);
            const contacts = world.getContactEvents();
            const joints = world.getJointEvents();
            const moves = world.getBodyEvents();
            records.push({
                step,
                begin: contacts.beginEvents.map((e) => [
                    e.shapeA.id.index1,
                    e.shapeA.id.generation,
                    e.shapeB.id.index1,
                    e.shapeB.id.generation,
                    e.contact.id.index1,
                    e.contact.id.generation,
                    e.normalImpulse,
                ]),
                end: contacts.endEvents.map((e) => [
                    e.shapeA.id.index1,
                    e.shapeA.id.generation,
                    e.shapeB.id.index1,
                    e.shapeB.id.generation,
                    e.contact.id.index1,
                    e.contact.id.generation,
                    e.normalImpulse,
                ]),
                hit: contacts.hitEvents.map((e) => [
                    e.shapeA.id.index1,
                    e.shapeB.id.index1,
                    e.contact.id.index1,
                    e.point,
                    e.normal,
                    e.approachSpeed,
                    e.userMaterialIdA.toString(),
                    e.userMaterialIdB.toString(),
                ]),
                joints: joints.map((e) => [e.joint.id.index1, e.joint.id.generation, e.userData]),
                moves: moves.moveEvents.slice(0, moves.count).map((e) => [
                    e.body.id.index1,
                    e.body.id.generation,
                    {
                        p: { ...e.transform.p },
                        q: { v: { ...e.transform.q.v }, s: e.transform.q.s },
                    },
                    e.fellAsleep,
                    e.userData,
                ]),
            });
        }
    } finally {
        world.destroy();
    }
    return records;
}

test("contact, hit, joint and move event records match the archived fixed stepped output in order", async () => {
    expect(output()).toEqual(
        await Bun.file(new URL("event-output.gold.json", import.meta.url)).json(),
    );
});
