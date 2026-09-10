// events-joint-break — stage-4 gym twin: the upstream `JointBreak` sample
// (`samples/src/samples/events.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The threshold-crossing
// joint cut lives in `update()`.

import goldJson from "../../../../src/standard/physics/samples/events-joint-break.json";
import { register } from "../gym";
import { buildJointBreak, renderJointBreak, updateJointBreak } from "../physics-joint-break";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildJointBreak,
        update: updateJointBreak,
        render: renderJointBreak,
    }),
);
