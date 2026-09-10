// stacking-box-pyramid — stage-4 gym twin: the upstream `BoxPyramid` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The `rows` knob picks
// the pyramid's base row count.

import goldJson from "../../../../src/standard/physics/samples/stacking-box-pyramid.json";
import { register } from "../gym";
import { buildBoxPyramid } from "../physics-box-pyramid";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildBoxPyramid }));
