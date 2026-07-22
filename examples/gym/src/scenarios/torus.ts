// mesh-torus — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Torus` sample
// (`samples/src/samples/mesh.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/mesh-torus.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildTorus } from "../tumble-torus";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildTorus,
    }),
);
