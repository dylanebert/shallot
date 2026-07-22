// bodies-spinning-book — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `SpinningBook`
// sample (`samples/src/samples/bodies.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/bodies-spinning-book.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildSpinningBook } from "../tumble-spinning-book";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildSpinningBook,
    }),
);
