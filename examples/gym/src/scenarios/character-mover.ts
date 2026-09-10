// character-mover — stage-4 gym twin: the upstream `Character` sample
// (`samples/src/samples/character.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold. A self-driven kinematic capsule mover patrols an arena on the plane solver — the drive
// lives in `update()`, shoving dynamic crates it leans on.

import goldJson from "../../../../src/standard/physics/samples/character-mover.json";
import { register } from "../gym";
import {
    buildCharacterMover,
    renderCharacterMover,
    updateCharacterMover,
} from "../physics-character-mover";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildCharacterMover,
        update: updateCharacterMover,
        render: renderCharacterMover,
    }),
);
