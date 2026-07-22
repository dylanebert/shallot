// bodies-body-type — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `BodyTypes` sample
// (`samples/src/samples/bodies.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The kinematic sweep
// lives in `update()` — the seam `tumble-pilot.test.ts` proves red/green.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/bodies-body-type.json";
import { register } from "../gym";
import { buildBodyType, updateBodyType } from "../tumble-body-type";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBodyType,
        update: updateBodyType,
    }),
);
