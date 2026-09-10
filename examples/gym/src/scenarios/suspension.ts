// joints-suspension — stage-4 gym twin: the upstream `Suspension` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/joints-suspension.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildSuspension } from "../physics-suspension";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildSuspension,
    }),
);
