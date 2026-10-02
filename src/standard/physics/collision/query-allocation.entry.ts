import { createApp, physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { xf } from "../common/math";
import { defaultQueryFilter, defaultSurfaceMaterial } from "../common/types";
import { queryColumns } from "../kernel/querycolumns";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";

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
    const q = queryColumns(physics.state);
    return {
        step: () => {
            const k = q.prepare(origin, filter);
            q.mover(capsule.center1, capsule.center2, capsule.radius);
            q.translation(translation);
            for (let i = 0; i < 6; ++i) {
                q.headerF[10] = 8 * i;
                q.headerF[11] = origin.y;
                k.worldQuery(physics.state.worldId, 5, 0);
                q.headerF[11] = rayOrigin.y;
                k.worldQuery(physics.state.worldId, 6, 0);
                k.worldQuery(physics.state.worldId, 3, 0);
            }
        },
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
