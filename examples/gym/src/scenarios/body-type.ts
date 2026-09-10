// bodies-body-type — stage-4 gym twin: the upstream `BodyTypes` sample
// (`samples/src/samples/bodies.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The kinematic sweep
// lives in `update()` — the seam `physics-pilot.test.ts` proves red/green.

import goldJson from "../../../../src/standard/physics/samples/bodies-body-type.json";
import { register } from "../gym";
import { buildBodyType, updateBodyType } from "../physics-body-type";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBodyType,
        update: updateBodyType,
    }),
);
