// joints-driving — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Driving` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-driving.json";
import { register } from "../gym";
import { buildDriving } from "../tumble-driving";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildDriving,
    }),
);
