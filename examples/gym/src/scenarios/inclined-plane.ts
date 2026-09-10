// shapes-inclined-plane — stage-4 gym twin: the upstream `InclinedPlane`
// sample (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. No knobs.

import goldJson from "../../../../src/standard/physics/samples/shapes-inclined-plane.json";
import { register } from "../gym";
import { buildInclinedPlane } from "../physics-inclined-plane";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildInclinedPlane }));
