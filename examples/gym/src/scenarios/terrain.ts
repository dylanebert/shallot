// mesh-terrain — stage-4 gym twin: the upstream `Terrain` sample
// (`samples/src/samples/mesh.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/mesh-terrain.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildTerrain } from "../physics-terrain";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildTerrain,
    }),
);
