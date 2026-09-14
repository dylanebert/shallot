// Active scene parity runs the immutable official command/bundle corpus. The historical 53 scene
// fixtures remain under fixtures/ as migration evidence, but their predecessor hashes are not the
// current authority and are intentionally not asserted here.

import { check } from "../../../harness/check";
import { loadConsumerCorpus, runCommonInput } from "../oracle/consumer";
import { compareCase } from "../oracle/strict";

check(
    "official Box3D scene command/bundle parity",
    {
        claim: "all 53 official Box3D scenes execute through the immutable command corpus and match its outputs",
        size: "integration",
        budget: 20_000,
    },
    () => {
        const corpus = loadConsumerCorpus();
        const scenes = corpus.cases.filter((item) => item.family === "scenario");
        if (scenes.length !== 53)
            throw new Error(`expected 53 official scenes, got ${scenes.length}`);
        for (const scene of scenes) {
            const result = compareCase(scene, runCommonInput(scene));
            if (result.status !== "pass")
                throw new Error(`${scene.id}: ${result.firstDifference?.path ?? "scene mismatch"}`);
        }
        console.log(JSON.stringify({ corpus: "immutable box3d v6", scenes: scenes.length }));
    },
);
