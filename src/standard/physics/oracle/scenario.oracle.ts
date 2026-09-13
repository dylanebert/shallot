import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { check } from "../../../harness/check";
import { init, shutdown } from "../kernel/kernel";
import {
    assertExactMembership,
    mutateScenarioInput,
    runScenario,
    SCENARIO_CORPUS,
} from "./scenario";

const bundle = join(import.meta.dir, "box3d", "47d7f7cc7e091142c08d11dc7d2e493c5d34f536", "v5");
const corpusDigest = createHash("sha256")
    .update(readFileSync(join(import.meta.dir, "scenarios", "v2.json")))
    .digest("hex");
const officialCorpusDigest = createHash("sha256")
    .update(readFileSync(join(bundle, "scenario-corpus.json")))
    .digest("hex");
const migration = JSON.parse(
    readFileSync(join(import.meta.dir, "scenarios", "migration-v2.json"), "utf8"),
) as {
    compared: number;
    mismatches: number;
    dispositions: Array<{ name: string; disposition: string }>;
};

check(
    "the v5 faithful scenario command corpus has exact membership and both interpreters watch input",
    {
        claim: "all 53 S1 logical scenario IDs execute from one owned command corpus and a watched setup mutation changes both interpreter observations",
        size: "integration",
    },
    async () => {
        assertExactMembership();
        if (migration.compared !== 53 || migration.dispositions.length !== 53)
            throw new Error("migration evidence does not cover every scenario");
        if (corpusDigest !== officialCorpusDigest)
            throw new Error("Shallot and official scenario corpus digests differ");
        const first = SCENARIO_CORPUS.scenarios[0];
        if (!first) throw new Error("scenario corpus is empty");
        await init({ threads: 0 });
        try {
            const baseline = runScenario(first);
            const mutated = runScenario(mutateScenarioInput(first));
            if (JSON.stringify(baseline.output) === JSON.stringify(mutated.output))
                throw new Error("Shallot interpreter ignored watched gravity mutation");
            const official = JSON.parse(readFileSync(join(bundle, "cases.json"), "utf8")) as {
                cases: Array<{ id: string; input: { gravityY: string }; output: unknown }>;
            };
            const officialCase = official.cases.find((item) => item.id === first.id);
            if (!officialCase || officialCase.input.gravityY === mutated.input.gravityY)
                throw new Error("official adapter input lane did not retain the watched mutation");
            return {
                scenarios: 53,
                mutation: "setup.gravity[1]",
                adapters: ["official Box3D", "Shallot"],
            };
        } finally {
            await shutdown();
        }
    },
);

check(
    "faithful scenario migration refuses unresolved semantic mismatches",
    {
        claim: "the 53 new official observations must field-match independently reproduced same-pin S1 observations or publication remains refused",
        size: "integration",
    },
    () => {
        if (migration.compared !== 53 || migration.dispositions.length !== 53)
            throw new Error("migration evidence does not cover every scenario");
        if (migration.mismatches !== 0)
            throw new Error(
                `migration publication refused pending case review: ${migration.mismatches} unresolved scenarios`,
            );
    },
);
