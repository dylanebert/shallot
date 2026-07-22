// shapes-inclined-plane — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `InclinedPlane`
// sample (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. No knobs.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/shapes-inclined-plane.json";
import { register } from "../gym";
import { buildInclinedPlane } from "../tumble-inclined-plane";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildInclinedPlane }));
