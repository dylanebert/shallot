// compound-spheres — stage-4 gym twin: the upstream `CompoundSpheres` sample
// (`samples/src/samples/compound.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../src/standard/physics/samples/compound-spheres.json";
import { register } from "../gym";
import { buildCompoundSpheres } from "../physics-compound-spheres";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildCompoundSpheres,
    }),
);
