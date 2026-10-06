import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init } from "../kernel/kernel";
import { bodyDisable, bodyEnable } from "./body";
import jointExpected from "./joint-mutation-output.json";
import expected from "./mutation-output.json";

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
    if (process.env.CAPTURE_JOINT_OUTPUT) console.log(`JOINT_OUTPUT=${JSON.stringify(actual)}`);
    else expect(actual).toEqual(jointExpected);
});

test("cold body mutations preserve public values and ordered events", () => {
    const actual = mutationOutput();
    if (process.env.CAPTURE_MUTATION_OUTPUT) console.log(JSON.stringify(actual, null, 2));
    else expect(actual).toEqual(expected);
});
