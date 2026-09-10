// joints-rope — stage-4 gym twin: the upstream `Rope` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/joints-rope.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { buildRope } from "../physics-rope";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildRope,
    }),
);
