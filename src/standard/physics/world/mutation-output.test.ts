import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init } from "../kernel/kernel";
import { bodyDisable, bodyEnable } from "./body";
import jointExpected from "./joint-mutation-output.json";
import expected from "./mutation-output.json";
import shapeExpected from "./shape-mutation-output.json";

await init(undefined, { threads: 0 });
const zero = { x: 0, y: 0, z: 0 };

export function mutationOutput() {
    const world = new PhysicsWorld({ gravity: zero });
    const a = world.createBody({ type: BodyType.Static, userData: "a" });
    const b = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 1.5, z: 0 },
        userData: "b",
    });
    a.createSphere({ enableContactEvents: true, userData: "shape-a" }, { center: zero, radius: 1 });
    b.createSphere({ enableContactEvents: true, userData: "shape-b" }, { center: zero, radius: 1 });
    const output: unknown[] = [];
    const plain = (value: unknown): unknown => {
        if (Array.isArray(value)) return value.map(plain);
        if (value && typeof value === "object") {
            if ("id" in value) return { id: plain(value.id) };
            return Object.fromEntries(
                Object.entries(value)
                    .filter(([key]) => key !== "world0")
                    .map(([key, child]) => [key, plain(child)]),
            );
        }
        return value;
    };
    function observe(name: string) {
        output.push(
            plain({
                name,
                bodies: [a, b].map((body) =>
                    body.isValid()
                        ? {
                              valid: true,
                              type: body.getType(),
                              awake: body.isAwake(),
                              pose: body.getTransform(),
                              velocity: body.getLinearVelocity(),
                              angular: body.getAngularVelocity(),
                              mass: body.getMassData(),
                              shapes: body.getShapeCount(),
                              userData: body.getUserData(),
                          }
                        : { valid: false },
                ),
                sensors: world.getSensorEvents(),
                contacts: world.getContactEvents(),
                moves: world.getBodyEvents(),
                joints: world.getJointEvents(),
            }),
        );
    }
    try {
        world.step(1 / 60);
        observe("contact");
        a.setType(BodyType.Dynamic);
        observe("static-to-dynamic");
        world.step(1 / 60);
        observe("dynamic-step");
        a.setType(BodyType.Static);
        observe("dynamic-to-static");
        b.setAwake(false);
        observe("sleep");
        b.setAwake(true);
        observe("wake");
        bodyDisable(world.state, b.id.index1 - 1);
        observe("disable");
        bodyEnable(world.state, b.id.index1 - 1);
        observe("enable");
        world.step(1 / 60);
        observe("enabled-step");
        const joint = world.createDistanceJoint(a, b);
        world.step(1 / 60);
        observe("joint-attached");
        b.destroy();
        observe("destroy");
        output.push({ name: "destroyed-joint", valid: joint.isValid() });
        world.step(1 / 60);
        observe("destroyed-step");
        return output;
    } finally {
        world.destroy();
    }
}

function jointMutationOutput() {
    const world = new PhysicsWorld({ gravity: zero });
    const a = world.createBody({ type: BodyType.Dynamic, userData: "a" });
    const b = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 1.5, y: 0, z: 0 },
        userData: "b",
    });
    const c = world.createBody({ position: { x: 3, y: 0, z: 0 }, userData: "c" });
    for (const body of [a, b, b, c])
        body.createSphere({ enableContactEvents: true }, { center: zero, radius: 1 });
    const joint = world.createDistanceJoint(a, b, { collideConnected: true });
    const output: unknown[] = [];
    function plain(value: unknown): unknown {
        if (Array.isArray(value)) return value.map(plain);
        if (value && typeof value === "object") {
            if ("id" in value) return { id: plain(value.id) };
            return Object.fromEntries(
                Object.entries(value)
                    .filter(([key]) => key !== "world0")
                    .map(([key, child]) => [key, plain(child)]),
            );
        }
        return value;
    }
    function observe(name: string) {
        output.push(
            plain({
                name,
                collide: joint.getCollideConnected(),
                valid: joint.isValid(),
                bodies: [a, b, c].map((body) => ({
                    pose: body.getTransform(),
                    awake: body.isAwake(),
                    linear: body.getLinearVelocity(),
                    angular: body.getAngularVelocity(),
                })),
                sensors: world.getSensorEvents(),
                contacts: world.getContactEvents(),
                moves: world.getBodyEvents(),
                joints: world.getJointEvents(),
            }),
        );
    }
    try {
        world.step(1 / 60);
        observe("contacts");
        joint.setCollideConnected(false);
        observe("collision-off");
        joint.setCollideConnected(false);
        observe("collision-off-again");
        world.step(1 / 60);
        observe("off-step");
        joint.setCollideConnected(true);
        observe("collision-on");
        joint.setCollideConnected(true);
        observe("collision-on-again");
        world.step(1 / 60);
        observe("on-step");
        a.setAwake(false);
        observe("sleep");
        joint.setMotorSpeed(2);
        observe("motor-input");
        world.step(1 / 60);
        observe("motor-step");
        return output;
    } finally {
        world.destroy();
    }
}

test("cold joint mutations preserve public values and ordered events", () => {
    const actual = jointMutationOutput();
    expect(actual).toEqual(jointExpected);
});

function shapeMutationOutput() {
    const world = new PhysicsWorld({ gravity: zero });
    const a = world.createBody({ userData: "ground" });
    const b = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 1.5, z: 0 },
        userData: "visitor",
    });
    const ground = a.createSphere(
        {
            enableContactEvents: true,
            enableSensorEvents: true,
            name: "ground",
            userData: "ground-shape",
        },
        { center: zero, radius: 1 },
    );
    const visitor = b.createSphere(
        {
            density: 2,
            enableContactEvents: true,
            enableSensorEvents: true,
            name: "visitor",
            userData: "visitor-shape",
        },
        { center: zero, radius: 1 },
    );
    const sensor = a.createSphere(
        { isSensor: true, enableSensorEvents: true, name: "sensor", userData: "sensor-shape" },
        { center: zero, radius: 3 },
    );
    const output: unknown[] = [];
    function plain(value: unknown): unknown {
        if (Array.isArray(value)) return value.map(plain);
        if (value && typeof value === "object") {
            if ("id" in value) return { id: plain(value.id) };
            return Object.fromEntries(
                Object.entries(value)
                    .filter(([key]) => key !== "world0")
                    .map(([key, child]) => [key, plain(child)]),
            );
        }
        return value;
    }
    function observe(name: string) {
        output.push(
            plain({
                name,
                shapes: [ground, visitor, sensor].map((shape) =>
                    shape.isValid()
                        ? {
                              valid: true,
                              type: shape.getType(),
                              aabb: shape.getAABB(),
                              density: shape.getDensity(),
                              mass: shape.computeMassData(),
                              name: shape.getName(),
                              userData: shape.getUserData(),
                              sensor: shape.isSensor(),
                              overlaps: shape.getSensorOverlaps(),
                              sensorEvents: shape.areSensorEventsEnabled(),
                              contactEvents: shape.areContactEventsEnabled(),
                              hitEvents: shape.areHitEventsEnabled(),
                          }
                        : { valid: false },
                ),
                body: {
                    mass: b.getMassData(),
                    pose: b.getTransform(),
                    velocity: b.getLinearVelocity(),
                    awake: b.isAwake(),
                    shapes: b.getShapeCount(),
                },
                sensors: world.getSensorEvents(),
                contacts: world.getContactEvents(),
                moves: world.getBodyEvents(),
                joints: world.getJointEvents(),
            }),
        );
    }
    try {
        world.step(1 / 60);
        observe("touches");
        visitor.setFilter({ categoryBits: 2n, maskBits: 0n, groupIndex: 0 });
        observe("filter-off");
        world.step(1 / 60);
        observe("filter-off-step");
        visitor.setFilter({ categoryBits: 2n, maskBits: 0xffffffffffffffffn, groupIndex: 0 });
        observe("filter-on");
        world.step(1 / 60);
        observe("filter-on-step");
        visitor.enableSensorEvents(false);
        observe("visitor-events-off");
        world.step(1 / 60);
        observe("visitor-off-step");
        visitor.enableSensorEvents(true);
        sensor.enableSensorEvents(false);
        observe("sensor-events-off");
        world.step(1 / 60);
        observe("sensor-off-step");
        sensor.enableSensorEvents(true);
        observe("sensor-events-on");
        world.step(1 / 60);
        observe("sensor-on-step");
        ground.enableContactEvents(false);
        visitor.enableContactEvents(false);
        visitor.enableHitEvents(true);
        observe("contact-off-hit-on");
        visitor.setFilter({ categoryBits: 4n, maskBits: 0xffffffffffffffffn, groupIndex: 0 });
        world.step(1 / 60);
        observe("changed-event-flags-step");
        visitor.enableContactEvents(true);
        visitor.enableHitEvents(false);
        observe("contact-on-hit-off");
        sensor.destroy();
        observe("destroy-sensor");
        world.step(1 / 60);
        observe("destroy-sensor-step");
        visitor.destroy();
        observe("destroy-visitor");
        world.step(1 / 60);
        observe("destroy-visitor-step");
        return output;
    } finally {
        world.destroy();
    }
}

test("cold shape mutations preserve public values and ordered events", () => {
    const actual = shapeMutationOutput();
    expect(actual).toEqual(shapeExpected);
});

test("cold body mutations preserve public values and ordered events", () => {
    const actual = mutationOutput();
    expect(actual).toEqual(expected);
});
