import { expect, setDefaultTimeout, test } from "bun:test";
import {
    createApp,
    DistanceJoint,
    GlobalTransform,
    qRotate,
    RevoluteJoint,
    SphericalJoint,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { drawJoints, JOINT_DOT_DIAMETER, JOINT_LINE } from "../src/joint-presentation";
import { JointSamplePlugin } from "../src/joints";
import { changeControl, controlValues, createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

for (const [scene, kind] of [
    ["distance", DistanceJoint],
    ["spherical", SphericalJoint],
    ["revolute", RevoluteJoint],
    ["door", RevoluteJoint],
] as const) {
    test(`${scene}: joint segments and dots follow the bodies' GlobalTransform and runtime joint membership without a GPU`, async () => {
        const app = await createApp({ defaults: false, plugins: [JointSamplePlugin] });
        const world = app.world;
        const bodies = createScene(world, scene);
        const values = controlValues(scene);
        try {
            const check = () => {
                world.step(0);
                const segments: { a: number[]; b: number[] }[] = [];
                const dots: number[][] = [];
                drawJoints(
                    world,
                    scene,
                    (a, b) => segments.push({ a, b }),
                    (point) => dots.push(point),
                );
                const expected = [...world.query([kind])].map((eid) => {
                    const joint = world.get(eid, kind)!;
                    const anchors = (["A", "B"] as const).map((side) => {
                        const body = side === "A" ? joint.a : joint.b;
                        const pose = world.get(body, GlobalTransform)!;
                        const offset = qRotate(...pose.rotation, ...joint[`localAnchor${side}`]);
                        return pose.translation.map((value, axis) => value + offset[axis]!);
                    });
                    return { a: anchors[0], b: anchors[1] };
                });
                expect(segments).toEqual(expected);
                expect(dots).toEqual(expected.flatMap(({ a, b }) => [a, b]));
                expect(JOINT_DOT_DIAMETER).toBe(4);
                expect(JOINT_LINE.width).toBe(1);
                return segments;
            };
            world.tick();
            const before = check();
            const joint = world.get([...world.query([kind])][0]!, kind)!;
            world.storage(GlobalTransform).translation.set(joint.b, 7, 11, 13);
            expect(check()).not.toEqual(before);
            if (scene === "distance" || scene === "door") {
                const name = scene === "distance" ? "Count" : "Two joints";
                for (const value of scene === "distance" ? [5, 1] : [0, 1]) {
                    changeControl(world, scene, bodies, values, name, value);
                    world.tick();
                    expect(check()).toHaveLength(scene === "distance" ? value : value ? 2 : 1);
                }
            }
        } finally {
            app.dispose();
        }
    });
}

test("non-Joints scenes draw no joint anchors even when they contain joints", async () => {
    const app = await createApp({ defaults: false, plugins: [JointSamplePlugin] });
    try {
        createScene(app.world, "door");
        let calls = 0;
        drawJoints(
            app.world,
            "joint",
            () => calls++,
            () => calls++,
        );
        expect(calls).toBe(0);
    } finally {
        app.dispose();
    }
});
