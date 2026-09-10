// shapes-restitution — stage-4 gym twin: the upstream `Restitution` sample
// (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. `shape`/`count` are
// rebuild knobs (boundedness probe only — the gold exists at defaults).

import goldJson from "../../../../src/standard/physics/samples/shapes-restitution.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { buildRestitution } from "../physics-restitution";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildRestitution }));
