// compound-simple — stage-4 gym twin: the upstream `SimpleCompound` sample
// (`samples/src/samples/compound.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../src/standard/physics/samples/compound-simple.json";
import { register } from "../gym";
import { buildCompoundSimple } from "../physics-compound-simple";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildCompoundSimple,
    }),
);
