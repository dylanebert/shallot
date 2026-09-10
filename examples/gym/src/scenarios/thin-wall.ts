// continuous-thin-wall — stage-4 gym twin: the upstream `ThinWall` sample
// (`samples/src/samples/continuous.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/continuous-thin-wall.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildThinWall } from "../physics-thin-wall";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildThinWall,
    }),
);
