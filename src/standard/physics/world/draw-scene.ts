import { quat, xf } from "../common/math";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { createFilterJoint, defaultJointDef } from "../solver/joint";
import { createMotorJoint, defaultMotorJointDef } from "../solver/motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "../solver/parallelJoint";
import { createPrismaticJoint, defaultPrismaticJointDef } from "../solver/prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "../solver/revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "../solver/sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "../solver/weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "../solver/wheelJoint";
import { createTestBody, createTestWorld, stepTestWorld, testShapeDef } from "../testing";
import { defaultDebugDraw } from "./draw";

function jointBase(bodyIdA: number, bodyIdB: number) {
    return { ...defaultJointDef(), bodyIdA, bodyIdB };
}

export function drawScene() {
    const world = createTestWorld();
    const a = createTestBody(world, {
        type: BodyType.Dynamic,
        position: { x: 1, y: 2, z: 3 },
    });
    createSphereShape(world, a, testShapeDef({ density: 2 }), {
        center: { x: 0.2, y: 0, z: 0 },
        radius: 0.5,
    });
    createCapsuleShape(world, a, testShapeDef({ density: 1 }), {
        center1: { x: -1, y: 0, z: 0 },
        center2: { x: 1, y: 0, z: 0 },
        radius: 0.3,
    });
    createHullShape(world, a, testShapeDef({ density: 1 }), makeBoxHull(0.5, 0.7, 0.9));
    const b = createTestBody(world, { position: { x: -2, y: 0, z: 1 } });
    createMeshShape(world, b, testShapeDef(), createGridMesh(2, 2, 1, 0, true), {
        x: 1,
        y: 1,
        z: 1,
    });
    createHeightFieldShape(world, b, testShapeDef(), createGrid(2, 2, { x: 1, y: 1, z: 1 }, false));
    createCompoundShape(
        world,
        b,
        testShapeDef(),
        createCompound({
            hulls: [
                {
                    hull: makeBoxHull(0.3, 0.4, 0.5),
                    transform: xf.identity(),
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!,
    );
    const base = jointBase(a, b);
    createFilterJoint(world, base);
    createMotorJoint(world, defaultMotorJointDef(base));
    createDistanceJoint(world, defaultDistanceJointDef(base));
    createParallelJoint(world, defaultParallelJointDef(base));
    createPrismaticJoint(world, defaultPrismaticJointDef(base));
    createRevoluteJoint(world, defaultRevoluteJointDef(base));
    createSphericalJoint(world, defaultSphericalJointDef(base));
    createWeldJoint(world, defaultWeldJointDef(base));
    createWheelJoint(world, defaultWheelJointDef(base));
    const draw = defaultDebugDraw();
    draw.drawShapes =
        draw.drawBounds =
        draw.drawMass =
        draw.drawJoints =
        draw.drawJointExtras =
            true;
    return { world, draw };
}

export function reactionScene() {
    const world = createTestWorld();
    const creators = [
        (a: number, b: number) =>
            createDistanceJoint(world, defaultDistanceJointDef(jointBase(a, b))),
        (a: number, b: number) =>
            createParallelJoint(world, defaultParallelJointDef(jointBase(a, b))),
        (a: number, b: number) =>
            createPrismaticJoint(world, defaultPrismaticJointDef(jointBase(a, b))),
        (a: number, b: number) =>
            createRevoluteJoint(world, defaultRevoluteJointDef(jointBase(a, b))),
        (a: number, b: number) =>
            createSphericalJoint(world, defaultSphericalJointDef(jointBase(a, b))),
        (a: number, b: number) => createWeldJoint(world, defaultWeldJointDef(jointBase(a, b))),
        (a: number, b: number) => createWheelJoint(world, defaultWheelJointDef(jointBase(a, b))),
    ];
    for (let i = 0; i < creators.length; ++i) {
        const a = createTestBody(world, { position: { x: i * 5 - 15, y: 0, z: 0 } });
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: i * 5 - 14, y: 2, z: 0 },
            rotation: quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
            angularVelocity: { x: 1, y: 2, z: 3 },
        });
        createSphereShape(world, b, testShapeDef({ density: 2 }), {
            center: { x: 0.2, y: 0, z: 0 },
            radius: 0.5,
        });
        creators[i]!(a, b);
    }
    for (let step = 0; step < 4; ++step) stepTestWorld(world, 1 / 60, 4);
    const draw = defaultDebugDraw();
    draw.drawShapes =
        draw.drawBounds =
        draw.drawMass =
        draw.drawJoints =
        draw.drawJointExtras =
            true;
    return { world, draw };
}
