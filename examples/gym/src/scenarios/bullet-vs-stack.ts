// continuous-bullet-vs-stack — stage-4 gym twin: the upstream
// `BulletVsStack` sample (`samples/src/samples/continuous.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/continuous-bullet-vs-stack.json";
import { register } from "../gym";
import { buildBulletVsStack } from "../physics-bullet-vs-stack";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBulletVsStack,
    }),
);
