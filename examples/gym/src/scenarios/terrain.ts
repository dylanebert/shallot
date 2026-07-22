// mesh-terrain — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Terrain` sample
// (`samples/src/samples/mesh.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/mesh-terrain.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildTerrain } from "../tumble-terrain";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildTerrain,
    }),
);
