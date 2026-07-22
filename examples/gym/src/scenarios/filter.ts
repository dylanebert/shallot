// joints-filter — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Filter` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-filter.json";
import { register } from "../gym";
import { buildFilter, renderFilter } from "../tumble-filter";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildFilter,
        render: renderFilter,
    }),
);
