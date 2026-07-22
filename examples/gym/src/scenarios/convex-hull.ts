// geometry-convex-hull — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `ConvexHull` sample
// (`samples/src/samples/geometry.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/geometry-convex-hull.json";
import { register } from "../gym";
import { buildConvexHull } from "../tumble-convex-hull";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildConvexHull,
    }),
);
