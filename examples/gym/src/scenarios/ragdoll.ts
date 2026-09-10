// ragdoll-ragdoll — stage-4 gym twin: the upstream `Ragdoll` sample
// (`samples/src/samples/ragdoll.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/ragdoll-ragdoll.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { buildRagdoll } from "../physics-ragdoll";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildRagdoll,
    }),
);
