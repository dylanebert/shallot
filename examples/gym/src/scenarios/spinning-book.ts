// bodies-spinning-book — stage-4 gym twin: the upstream `SpinningBook`
// sample (`samples/src/samples/bodies.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/bodies-spinning-book.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildSpinningBook } from "../physics-spinning-book";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildSpinningBook,
    }),
);
