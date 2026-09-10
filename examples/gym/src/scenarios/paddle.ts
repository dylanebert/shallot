// joints-paddle — the physics sample host's pilot scenario: the Paddle sample
// ported through the escape hatch, verified bit-exact against its committed gold, rendered with the
// source-faithful debug-draw layer + mouse-grab. It replaces the dropped powered-rotor recipe as the
// motor-joint verification home (the red-first oracle proof lives in `physics-pilot.test.ts`).

import goldJson from "../../../../src/standard/physics/samples/joints-paddle.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { buildPaddle } from "../physics-paddle";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildPaddle }));
