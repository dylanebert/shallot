// determinism-falling-ragdolls — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js
// `FallingRagdolls` sample (`samples/src/samples/ragdoll.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/determinism-falling-ragdolls.json";
import { register } from "../gym";
import { buildFallingRagdolls } from "../tumble-falling-ragdolls";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildFallingRagdolls,
    }),
);
