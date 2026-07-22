// joints-parallel — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Parallel` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-parallel.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildParallel, renderParallel, updateParallel } from "../tumble-parallel";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildParallel,
        update: updateParallel,
        render: renderParallel,
    }),
);
