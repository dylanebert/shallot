import { beforeAll, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    assertCopiedNodeCase,
    createExample,
    example,
    manualParticleCases,
} from "./packed-install-fixture";

// Manual: packed re-runs cost 14–46 s; examples/particles/src/{fog,motes}.node.ts prove these claims automatically.
beforeAll(() => createExample("particles"), TIMEOUT.startup);

for (const [subject, claim] of manualParticleCases) {
    test(
        `particles passes ${subject}: ${claim} in its copy`,
        () => {
            assertCopiedNodeCase(example("particles").project, "particles", subject, claim);
        },
        Number.POSITIVE_INFINITY,
    );
}
