// bodies-motion-locks — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `MotionLocks` sample
// (`samples/src/samples/bodies.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/bodies-motion-locks.json";
import { register } from "../gym";
import { buildMotionLocks, renderMotionLocks, updateMotionLocks } from "../tumble-motion-locks";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildMotionLocks,
        update: updateMotionLocks,
        render: renderMotionLocks,
    }),
);
