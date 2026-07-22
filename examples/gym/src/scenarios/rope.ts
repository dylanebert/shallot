// joints-rope — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Rope` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-rope.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildRope } from "../tumble-rope";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildRope,
    }),
);
