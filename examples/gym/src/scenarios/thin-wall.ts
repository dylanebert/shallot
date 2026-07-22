// continuous-thin-wall — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `ThinWall` sample
// (`samples/src/samples/continuous.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/continuous-thin-wall.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildThinWall } from "../tumble-thin-wall";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildThinWall,
    }),
);
