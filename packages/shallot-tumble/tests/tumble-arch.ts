// The tumble.js `Arch` sample (`samples/src/samples/stacks.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A semicircular masonry arch of wedge-shaped voussoirs, held up by friction
// alone, topped with four loose boxes stacked on the keystone — the classic no-joint stability showpiece
// (quarter voussoir coordinates, Box3D's Arch). No knobs, no `update()`.
//
// Creation order is load-bearing for the hash: ground, then the 17 voussoirs (8 inner/outer pairs plus
// the keystone) in the sample's loop order, then the four boxes — the sample's exact order.

import { BodyType, createHull, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Arch scene into `world`. No knobs — `params` is unused, kept for the shared `SampleBuild`
 * signature.
 */
export function buildArch(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const s = 0.25;
    const inner: [number, number][] = [
        [16.0, 0.0],
        [14.93803712795643, 5.133601056842984],
        [13.79871746027416, 10.24928069555078],
        [12.56252963284711, 15.34107019122473],
        [11.20040987372525, 20.39856541571217],
        [9.66521217819836, 25.40369899225096],
        [7.87179930638133, 30.3179337000085],
        [5.635199558196225, 35.03820717801641],
        [2.405937953536585, 39.09554102558315],
    ].map(([x, y]) => [x * s, y * s]);
    const outer: [number, number][] = [
        [24.0, 0.0],
        [22.33619528222415, 6.02299846205841],
        [20.54936888969905, 12.00964361211476],
        [18.60854610798073, 17.9470321677465],
        [16.46769273811807, 23.81367936585418],
        [14.05325025774858, 29.57079353071012],
        [11.23551045834022, 35.13775818285372],
        [7.752568160730571, 40.30450679009583],
        [3.016931552701656, 44.28891593799322],
    ].map(([x, y]) => [x * s, y * s]);

    const d = 0.5;
    const wedge = (corners: [number, number][]): void => {
        const pts = [-d, d].flatMap((z) => corners.map(([x, y]) => ({ x, y: 1 + y, z })));
        const hull = createHull(pts, 8);
        if (hull === null) return;
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createHull({ density: 200 }, hull);
    };

    const mirror = ([x, y]: [number, number]): [number, number] => [-x, y];
    for (let i = 0; i < 8; ++i) {
        wedge([inner[i], outer[i], outer[i + 1], inner[i + 1]]);
        wedge([outer[i], inner[i], inner[i + 1], outer[i + 1]].map(mirror));
    }
    wedge([inner[8], outer[8], mirror(outer[8]), mirror(inner[8])]);

    const box = makeBoxHull(2, 0.5, d);
    for (let i = 0; i < 4; ++i) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 1 + 0.5 + outer[8][1] + i, z: 0 },
        });
        body.createHull({ density: 200 }, box);
    }
}
