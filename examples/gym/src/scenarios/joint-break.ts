// events-joint-break — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `JointBreak` sample
// (`samples/src/samples/events.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The threshold-crossing
// joint cut lives in `update()`.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/events-joint-break.json";
import { register } from "../gym";
import { buildJointBreak, renderJointBreak, updateJointBreak } from "../tumble-joint-break";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildJointBreak,
        update: updateJointBreak,
        render: renderJointBreak,
    }),
);
