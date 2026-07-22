// shapes-restitution — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Restitution` sample
// (`samples/src/samples/shapes.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. `shape`/`count` are
// rebuild knobs (boundedness probe only — the gold exists at defaults).

import goldJson from "../../../../packages/shallot/tests/tumble/samples/shapes-restitution.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildRestitution } from "../tumble-restitution";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildRestitution }));
