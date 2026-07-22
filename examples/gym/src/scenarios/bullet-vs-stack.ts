// continuous-bullet-vs-stack — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js
// `BulletVsStack` sample (`samples/src/samples/continuous.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/continuous-bullet-vs-stack.json";
import { register } from "../gym";
import { buildBulletVsStack } from "../tumble-bullet-vs-stack";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBulletVsStack,
    }),
);
