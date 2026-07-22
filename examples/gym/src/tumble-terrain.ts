// The tumble.js `Terrain` sample (`samples/src/samples/mesh.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A sine-wave triangle-mesh terrain; a 5x5 grid of the `shape` knob's shape
// (cylinder/sphere/box) drops over the hills and settles in the troughs. No `update()`.
//
// Creation order is load-bearing for the hash: the static mesh ground, then the grid row-major (ix outer,
// iz inner) — the sample's exact order.

import {
    BodyType,
    createCylinder,
    createWaveMesh,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/** Author the Terrain scene into `world`, reading the `shape` knob (cylinder/sphere/box). */
export function buildTerrain(world: World, params: SampleParams): void {
    const mesh = createWaveMesh(24, 24, 1, 0.8, 0.15, 0.15);
    world.createBody({ type: BodyType.Static }).createMesh({}, mesh);

    const shape = params.shape as string;
    const cylinder = createCylinder(0.6, 0.4, 0.0, 12);
    const box = makeBoxHull(0.4, 0.4, 0.4);
    for (let ix = 0; ix < 5; ix++) {
        for (let iz = 0; iz < 5; iz++) {
            const x = (ix - 2) * 2.2;
            const z = (iz - 2) * 2.2;
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y: 4, z },
            });
            if (shape === "sphere") {
                body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.4 });
            } else if (shape === "box") {
                body.createHull({ density: 1 }, box);
            } else {
                body.createHull({ density: 1 }, cylinder);
            }
        }
    }
}
