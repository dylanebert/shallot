// shapes-shape-soup — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `ShapeSoup` sample
// (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. `rows` is a rebuild
// knob (boundedness probe only — the gold exists at defaults).

import goldJson from "../../../../packages/shallot/tests/tumble/samples/shapes-shape-soup.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildShapeSoup } from "../tumble-shape-soup";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildShapeSoup }));
