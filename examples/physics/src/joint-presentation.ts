import {
    DistanceJoint,
    FilterJoint,
    GlobalTransform,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    qRotate,
    RevoluteJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";
import { drawInkPoint, INK_DOT as JOINT_DOT, INK as JOINT_INK } from "./ink-presentation";

export { INK as JOINT_INK, INK_DOT as JOINT_DOT } from "./ink-presentation";

import { categories, type Scene } from "./scenes";

const kinds = [
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
] as const;
const scenes = new Set(
    categories.find((category) => category.name === "Joints")!.scenes.map(({ id }) => id),
);
export const JOINT_DOT_DIAMETER = JOINT_DOT.width;
export const JOINT_LINE = { width: 1, depthBias: -1 };
type Point = [number, number, number];

export function drawJointViews(
    world: World,
    scene: Scene,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    drawJoints(
        world,
        scene,
        (a, b) => line(a, b, JOINT_INK, JOINT_LINE),
        (point) => drawInkPoint(world, camera, point, line),
    );
}

/** Emits one segment and two anchor dots per live joint, in world space. */
export function drawJoints(
    world: World,
    scene: Scene,
    segment: (a: Point, b: Point) => void,
    dot: (point: Point) => void,
): void {
    if (!scenes.has(scene)) return;
    const pose = world.storage(GlobalTransform);
    for (const kind of kinds) {
        const storage = world.storage(kind);
        for (const eid of world.query([kind])) {
            const bodies = [storage.a.get(eid), storage.b.get(eid)];
            if (!bodies.every((body) => body && world.has(body, GlobalTransform))) continue;
            const anchors = [storage.localAnchorA, storage.localAnchorB].map(
                (anchor, index): Point => {
                    const body = bodies[index]!;
                    const offset = qRotate(
                        pose.rotation.x.get(body),
                        pose.rotation.y.get(body),
                        pose.rotation.z.get(body),
                        pose.rotation.w.get(body),
                        anchor.x.get(eid),
                        anchor.y.get(eid),
                        anchor.z.get(eid),
                    );
                    return [
                        pose.translation.x.get(body) + offset[0],
                        pose.translation.y.get(body) + offset[1],
                        pose.translation.z.get(body) + offset[2],
                    ];
                },
            );
            segment(anchors[0]!, anchors[1]!);
            dot(anchors[0]!);
            dot(anchors[1]!);
        }
    }
}
