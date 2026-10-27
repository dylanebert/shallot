import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, RevoluteJoint } from "@dylanebert/shallot";
import { getJointLinearSeparation } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DoorState, JointSamplePlugin } from "../src/joints";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("Door read-only rows keep each joint's running separation maximum and Impulse resets both", async () => {
    let bodies: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [JointSamplePlugin],
        setup(world) {
            bodies = createScene(world, "door");
        },
    });
    try {
        const world = app.world;
        const values = controlValues("door");
        const sequence = (await Bun.file(new URL("./door.controls.txt", import.meta.url)).text())
            .trim()
            .split("\n")
            .map((line) => {
                const [step, name, value] = line.split("\t");
                return { step: Number(step), name: name!, value: Number(value) };
            });
        const maxima = [0, 0];
        let grew = false,
            retained = false;
        const rows = ["Translation error 1", "Translation error 2"].map(
            (name) => scenes.door.controls[name]!,
        );
        for (let step = 0; step < 600; step++) {
            for (const change of sequence.filter((change) => change.step === step)) {
                const previous = world.resource(DoorState).joints.map((eid) => world.ref(eid));
                changeControl(world, "door", bodies, values, change.name, change.value);
                if (change.name === "Impulse") {
                    maxima.fill(0);
                    expect(rows[0]!.read!(world)).toBe(0);
                    if (values["Two joints"]) expect(rows[1]!.read!(world)).toBe(0);
                }
                if (change.name === "Two joints") {
                    expect(previous.every((ref) => world.resolve(ref) === 0)).toBe(true);
                    expect([...world.query([RevoluteJoint])]).toHaveLength(change.value ? 2 : 1);
                    // CreateJoints enables limits even when the Limit checkbox remains off.
                    for (const joint of world.resource(DoorState).joints)
                        expect(world.storage(RevoluteJoint).enableLimit.get(joint)).toBe(1);
                }
            }
            world.tick();
            for (let i = 0; i < 2; i++) {
                const joint = world.resource(DoorState).joints[i];
                if (joint === undefined) {
                    expect(rows[i]!.read!(world)).toBeNull();
                    continue;
                }
                const separation = getJointLinearSeparation(world, joint)!;
                grew ||= separation > maxima[i]!;
                retained ||= separation < maxima[i]!;
                maxima[i] = Math.max(maxima[i]!, separation);
                expect(rows[i]!.read!(world), `step ${step}, row ${i + 1}`).toBe(maxima[i]!);
                expect(() =>
                    changeControl(world, "door", bodies, values, rows[i]!.label, 10),
                ).toThrow();
            }
        }
        expect(grew).toBe(true);
        expect(retained).toBe(true);
    } finally {
        app.dispose();
    }
});
