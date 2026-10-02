import { createApp, physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { xf } from "../common/math";
import { defaultQueryFilter, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";
import { castMover, castRayClosest, collideMover } from "./query";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const physics = physicsWorld(app.world)!;
    const hull = makeBoxHull(1, 0.25, 1);
    const material = defaultSurfaceMaterial();
    const compound = createCompound({ hulls: [{ hull, transform: xf.identity(), material }] })!;
    const capsule = {
        center1: { x: 0, y: -0.5, z: 0 },
        center2: { x: 0, y: 0.5, z: 0 },
        radius: 0.4,
    };
    physics
        .createBody({ position: { x: 0, y: 0, z: 0 } })
        .createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
    physics.createBody({ position: { x: 8, y: 0, z: 0 } }).createCapsule({}, capsule);
    physics.createBody({ position: { x: 16, y: 0, z: 0 } }).createHull({}, hull);
    physics
        .createBody({ position: { x: 24, y: 0, z: 0 } })
        .createMesh({}, createGridMesh(4, 4, 1, 0, true));
    physics
        .createBody({ position: { x: 32, y: 0, z: 0 } })
        .createHeightField({}, createGrid(8, 8, { x: 1, y: 1, z: 1 }, false));
    physics.createBody({ position: { x: 40, y: 0, z: 0 } }).createCompound({}, compound);
    const origin = { x: 0, y: 0.75, z: 0 };
    const rayOrigin = { x: 0, y: 3, z: 0 };
    const translation = { x: 0, y: -4, z: 0 };
    const filter = defaultQueryFilter();
    const gather = () => true;
    return {
        step: () => {
            for (let i = 0; i < 6; ++i) {
                origin.x = 8 * i;
                rayOrigin.x = 8 * i;
                collideMover(physics.state, origin, capsule, filter, gather);
                castMover(physics.state, rayOrigin, capsule, translation, filter, null);
                castRayClosest(physics.state, rayOrigin, translation, filter);
            }
        },
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
