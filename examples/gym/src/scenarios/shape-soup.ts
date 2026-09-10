// shapes-shape-soup — stage-4 gym twin: the upstream `ShapeSoup` sample
// (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. `rows` is a rebuild
// knob (boundedness probe only — the gold exists at defaults).

import goldJson from "../../../../src/standard/physics/samples/shapes-shape-soup.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildShapeSoup } from "../physics-shape-soup";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildShapeSoup }));
