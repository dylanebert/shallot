// determinism-falling-ragdolls — stage-4 gym twin: the upstream
// `FallingRagdolls` sample (`samples/src/samples/ragdoll.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/determinism-falling-ragdolls.json";
import { register } from "../gym";
import { buildFallingRagdolls } from "../physics-falling-ragdolls";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildFallingRagdolls,
    }),
);
