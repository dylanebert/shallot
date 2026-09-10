// events-sensor-sweep — stage-4 gym twin: the upstream `SensorSweep` sample
// (`samples/src/samples/events.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The sine-driven
// vertical sweep lives in `update()`.

import goldJson from "../../../../src/standard/physics/samples/events-sensor-sweep.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildSensorSweep, renderSensorSweep, updateSensorSweep } from "../physics-sensor-sweep";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildSensorSweep,
        update: updateSensorSweep,
        render: renderSensorSweep,
    }),
);
