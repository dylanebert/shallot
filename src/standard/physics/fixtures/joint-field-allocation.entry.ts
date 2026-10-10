import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { Body, BodyType, RevoluteJoint } from "@dylanebert/shallot/physics";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { SyncPhysicsConstraintsSystem } from "../runtime";

export let controlSink: { frame: number } | undefined;
export function control() {
    controlSink = { frame: 0 };
}

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const anchor = world.create();
    const body = world.create();
    world.add(anchor, Body, { type: BodyType.Static });
    world.add(body, Body, { type: BodyType.Dynamic });
    const joint = world.create();
    world.add(joint, RevoluteJoint, {
        a: anchor,
        b: body,
        enableMotor: 1,
        localRotationA: [0, 0, 0, 1],
        localRotationB: [0, 0, 0, 1],
        maxMotorTorque: 10,
    });
    world.tick();
    // Exact ticks omit the frame upload that normally clears these marks.
    world.clearChanges();
    if (physicsWorld(world)!.getCounters().jointCount !== 1)
        throw new Error("allocation subject did not create its revolute joint");
    for (const system of StandardPhysicsPlugin.systems!) world.removeSystem(system);
    world.addSystem(SyncPhysicsConstraintsSystem, StandardPhysicsPlugin.name);
    const motorSpeed = world.storage(RevoluteJoint).motorSpeed;
    const speeds = [{ value: 1.25 }, { value: 2.5 }, { value: 4.75 }, { value: 8.125 }];
    let frame = 0;
    return {
        step() {
            motorSpeed.set(joint, speeds[frame++ & 3].value);
            world.tick();
            world.clearChanges();
            if (input === "allocating") control();
        },
        dispose: () => app.dispose(),
    };
}
