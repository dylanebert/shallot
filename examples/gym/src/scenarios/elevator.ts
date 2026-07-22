// joints-elevator — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Elevator` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer. The motor-speed reversal lives
// in `update()`.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-elevator.json";
import { register } from "../gym";
import { buildElevator, updateElevator } from "../tumble-elevator";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildElevator,
        update: updateElevator,
    }),
);
