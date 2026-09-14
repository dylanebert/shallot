import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { check } from "../../../harness/check";
import { executeStrictReport, readFrozenStrictReport, strictReportDigest } from "./strict";

check(
    "Box3D strict parity executes the complete frozen v6 mapped surface",
    { claim: "box3d-strict-parity", size: "integration", budget: 20_000 },
    () => {
        const frozen = readFrozenStrictReport();
        const actual = executeStrictReport();
        if (actual.population.executed !== 111 || actual.results.length !== 111)
            throw new Error("strict parity did not execute every immutable v6 case");
        if (strictReportDigest(actual) !== strictReportDigest(frozen))
            throw new Error("strict report changed since its frozen O7 capture");
        const digest = createHash("sha256")
            .update(readFileSync(new URL("./reports/strict-v6.json", import.meta.url)))
            .digest("hex");
        console.log(
            JSON.stringify({
                report: "src/standard/physics/oracle/reports/strict-v6.json",
                digest,
                canonicalDigest: strictReportDigest(frozen),
                population: frozen.population,
            }),
        );
        if (frozen.population.mismatched === 0)
            throw new Error(
                "strict parity unexpectedly became green before the ledger-only delivery",
            );
        throw new Error(
            `strict parity remains red as required: ${frozen.population.mismatched} mismatches`,
        );
    },
);
