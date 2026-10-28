import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    DistanceJoint,
    GlobalTransform,
    qRotate,
    RevoluteJoint,
    resizeViewport,
    SphericalJoint,
} from "@dylanebert/shallot";
import { type View, Views, worldToViewport } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    drawJoints,
    drawJointViews,
    JOINT_DOT_DIAMETER,
    JOINT_LINE,
} from "../src/joint-presentation";
import { JointSamplePlugin } from "../src/joints";
import { RagdollPlugin } from "../src/ragdoll";
import { changeControl, controlValues, createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
for (const [scene, count] of [
    ["ragdollBox", 14],
    ["ragdollMesh", 28],
    ["ragdollPile", 280],
    ["ragdollIncline", 14],
] as const) {
    test(`${scene}: every bone joint and anchor draws one segment and two dots`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [RagdollPlugin],
            setup(world) {
                createScene(world, scene);
            },
        });
        try {
            app.world.tick();
            const segments: number[][][] = [],
                dots: number[][] = [];
            drawJoints(
                app.world,
                scene,
                (a, b) => segments.push([a, b]),
                (point) => dots.push(point),
            );
            expect(segments).toHaveLength(count);
            expect(dots).toEqual(segments.flat());
            expect(dots.flat().every(Number.isFinite)).toBe(true);
        } finally {
            app.dispose();
        }
    });
}

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

test("an anchor inside a body emits a centered four-CSS-pixel mark with draw-on-top ink", async () => {
    const app = await createApp({ defaults: false, plugins: [JointSamplePlugin] });
    const world = app.world;
    try {
        createScene(world, "spherical");
        world.tick();
        const joint = [...world.query([SphericalJoint])][0]!;
        const body = world.storage(SphericalJoint).b.get(joint);
        world.storage(SphericalJoint).localAnchorB.set(joint, 0, 0, 0);
        const anchor = [...world.get(body, GlobalTransform)!.translation] as [
            number,
            number,
            number,
        ];
        const camera = world.create();
        world.add(camera, Camera, { fov: 60, near: 0.01, mode: 0 });
        world.add(camera, GlobalTransform);
        world.storage(GlobalTransform).scale.set(camera, 1, 1, 1);
        world.resource(Views).set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
        for (const [width, height, dpr] of [
            [1280, 720, 1],
            [390, 844, 2],
        ]) {
            resizeViewport(world, 0, width!, height!, dpr!);
            for (const distance of [10, 40]) {
                for (const yaw of [0, Math.PI / 4]) {
                    const rotation = [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)] as const;
                    const offset = qRotate(...rotation, 0, 0, distance);
                    const pose = world.storage(GlobalTransform);
                    pose.translation.set(
                        camera,
                        anchor[0] + offset[0],
                        anchor[1] + offset[1],
                        anchor[2] + offset[2],
                    );
                    pose.rotation.set(camera, ...rotation);
                    const drawn: {
                        a: [number, number, number];
                        b: [number, number, number];
                        color: number;
                        options: typeof JOINT_LINE;
                    }[] = [];
                    drawJointViews(world, "spherical", camera, (a, b, color, options) =>
                        drawn.push({ a, b, color, options }),
                    );
                    expect(drawn).toHaveLength(3);
                    for (const { color, options } of drawn) {
                        expect(color).toBe(0xeee9df);
                        expect(options.depthBias).toBe(-1);
                    }
                    expect(drawn[0]!.options.width).toBe(1);
                    const mark = drawn[2]!;
                    expect(mark.options.width).toBe(4);
                    for (let axis = 0; axis < 3; axis++)
                        expect((mark.a[axis]! + mark.b[axis]!) / 2).toBeCloseTo(anchor[axis]!, 10);
                    const a = worldToViewport(world, camera, mark.a)!;
                    const b = worldToViewport(world, camera, mark.b)!;
                    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeCloseTo(4, 5);
                    expect(a[1]).toBeCloseTo(b[1], 8);
                }
            }
        }
    } finally {
        app.dispose();
    }
});

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
