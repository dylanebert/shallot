import { PhysicsWorld } from "../api/world";
import { xf } from "../common/math";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";
import { defaultDebugDraw } from "./draw";

export function drawScene() {
    const world = new PhysicsWorld();
    const a = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 2, z: 3 } });
    a.createSphere({ density: 2 }, { center: { x: 0.2, y: 0, z: 0 }, radius: 0.5 });
    a.createCapsule(
        { density: 1 },
        { center1: { x: -1, y: 0, z: 0 }, center2: { x: 1, y: 0, z: 0 }, radius: 0.3 },
    );
    a.createHull({ density: 1 }, makeBoxHull(0.5, 0.7, 0.9));
    const b = world.createBody({ position: { x: -2, y: 0, z: 1 } });
    b.createMesh({}, createGridMesh(2, 2, 1, 0, true));
    b.createHeightField({}, createGrid(2, 2, { x: 1, y: 1, z: 1 }, false));
    b.createCompound(
        {},
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
    world.createFilterJoint(a, b);
    world.createMotorJoint(a, b);
    world.createDistanceJoint(a, b);
    world.createParallelJoint(a, b);
    world.createPrismaticJoint(a, b);
    world.createRevoluteJoint(a, b);
    world.createSphericalJoint(a, b);
    world.createWeldJoint(a, b);
    world.createWheelJoint(a, b);
    const draw = defaultDebugDraw();
    draw.drawShapes =
        draw.drawBounds =
        draw.drawMass =
        draw.drawJoints =
        draw.drawJointExtras =
            true;
    return { world, draw };
}
