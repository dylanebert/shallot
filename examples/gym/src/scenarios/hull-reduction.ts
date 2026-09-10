// geometry-hull-reduction — stage-4 gym twin: the upstream `HullReduction`
// sample (`samples/src/samples/geometry.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/geometry-hull-reduction.json";
import { register } from "../gym";
import { buildHullReduction } from "../physics-hull-reduction";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildHullReduction,
    }),
);
