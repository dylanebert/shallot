// ragdoll-ragdoll — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Ragdoll` sample
// (`samples/src/samples/ragdoll.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/ragdoll-ragdoll.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildRagdoll } from "../tumble-ragdoll";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildRagdoll,
    }),
);
