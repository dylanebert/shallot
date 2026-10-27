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
import { worldToViewport } from "@dylanebert/shallot/rendering";
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
export const JOINT_DOT_DIAMETER = 4;
export const JOINT_INK = 0xeee9df;
export const JOINT_LINE = { width: 1, depthBias: -1 };
export const JOINT_DOT = { width: JOINT_DOT_DIAMETER, depthBias: -1 };
type Point = [number, number, number];

export function drawJointViews(
    world: World,
    scene: Scene,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const q = world.storage(GlobalTransform).rotation;
    const right = qRotate(
        q.x.get(camera),
        q.y.get(camera),
        q.z.get(camera),
        q.w.get(camera),
        1,
        0,
        0,
    );
    drawJoints(
        world,
        scene,
        (a, b) => line(a, b, JOINT_INK, JOINT_LINE),
        (point) => {
            const center = worldToViewport(world, camera, point);
            const unit = worldToViewport(world, camera, [
                point[0] + right[0],
                point[1] + right[1],
                point[2] + right[2],
            ]);
            if (!center || !unit) return;
            const half =
                JOINT_DOT_DIAMETER / Math.hypot(unit[0] - center[0], unit[1] - center[1]) / 2;
            const a: Point = [
                point[0] - right[0] * half,
                point[1] - right[1] * half,
                point[2] - right[2] * half,
            ];
            const b: Point = [
                point[0] + right[0] * half,
                point[1] + right[1] * half,
                point[2] + right[2] * half,
            ];
            line(a, b, JOINT_INK, JOINT_DOT);
        },
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
