// mesh-torus — stage-4 gym twin: the upstream `Torus` sample
// (`samples/src/samples/mesh.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/mesh-torus.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildTorus } from "../physics-torus";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildTorus,
    }),
);
