// geometry-convex-primitives — stage-4 gym twin: the upstream
// `ConvexPrimitives` sample (`samples/src/samples/geometry.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/geometry-convex-primitives.json";
import { register } from "../gym";
import { buildConvexPrimitives } from "../physics-convex-primitives";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildConvexPrimitives,
    }),
);
