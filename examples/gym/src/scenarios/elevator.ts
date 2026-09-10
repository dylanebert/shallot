// joints-elevator — stage-4 gym twin: the upstream `Elevator` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer. The motor-speed reversal lives
// in `update()`.

import goldJson from "../../../../src/standard/physics/samples/joints-elevator.json";
import { register } from "../gym";
import { buildElevator, updateElevator } from "../physics-elevator";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildElevator,
        update: updateElevator,
    }),
);
