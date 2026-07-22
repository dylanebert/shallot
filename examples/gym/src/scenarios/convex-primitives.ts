// geometry-convex-primitives — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js
// `ConvexPrimitives` sample (`samples/src/samples/geometry.ts`) ported through the escape hatch, verified
// bit-exact against its committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/geometry-convex-primitives.json";
import { register } from "../gym";
import { buildConvexPrimitives } from "../tumble-convex-primitives";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildConvexPrimitives,
    }),
);
