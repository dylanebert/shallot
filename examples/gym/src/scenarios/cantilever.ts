// joints-cantilever — stage-4 gym twin: the upstream `Cantilever` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../src/standard/physics/samples/joints-cantilever.json";
import { register } from "../gym";
import { buildCantilever } from "../physics-cantilever";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildCantilever,
    }),
);
