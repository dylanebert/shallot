// geometry-convex-hull — stage-4 gym twin: the upstream `ConvexHull` sample
// (`samples/src/samples/geometry.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../src/standard/physics/samples/geometry-convex-hull.json";
import { register } from "../gym";
import { buildConvexHull } from "../physics-convex-hull";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildConvexHull,
    }),
);
