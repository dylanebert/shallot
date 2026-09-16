import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { check } from "../../../harness/check";
import {
    executeStrictReport,
    readFrozenStrictReport,
    strictReportDigest,
    watchStrictProgress,
} from "./strict";

const current = JSON.parse(
    readFileSync(new URL("./box3d/current.json", import.meta.url), "utf8"),
) as { target: string; conformsThrough: string; bundle: string };

check(
    "Box3D strict parity executes the complete frozen v6 mapped surface",
    { claim: "box3d-strict-parity", size: "integration", budget: 20_000 },
    () => {
        const frozen = readFrozenStrictReport();
        const actual = executeStrictReport();
        if (actual.population.executed !== 111 || actual.results.length !== 111)
            throw new Error("strict parity did not execute every immutable v6 case");
        if (actual.target !== frozen.target || actual.bundle !== frozen.bundle)
            throw new Error("strict parity did not execute the selected immutable bundle");
        if (
            actual.target !== current.target ||
            actual.bundle !== current.bundle ||
            actual.conformsThrough !== current.conformsThrough
        )
            throw new Error("strict parity report disagrees with the selected current receipt");
        if (actual.population.mismatched === 0 && actual.conformsThrough !== actual.target)
            throw new Error("zero-mismatch 111/111 strict parity did not close conformance");

        // O7 is audit evidence only. Keep both digests visible, but never compare the live physics
        // result to O7's predecessor output: a newly passing case is the expected delivery signal.
        const frozenFileDigest = createHash("sha256")
            .update(readFileSync(new URL("./reports/strict-v6.json", import.meta.url)))
            .digest("hex");
        const watch = watchStrictProgress(actual, frozen);
        console.log(
            JSON.stringify({
                report: "src/standard/physics/oracle/reports/strict-v6.json",
                baselineFileDigest: frozenFileDigest,
                baselineCanonicalDigest: strictReportDigest(frozen),
                currentCanonicalDigest: strictReportDigest(actual),
                baseline: frozen.population,
                current: actual.population,
                improvements: watch.improvements.length,
                unchangedMismatches: watch.unchangedMismatches.length,
            }),
        );
        if (watch.regressions.length > 0)
            throw new Error(`strict parity introduced a regression: ${watch.regressions[0]}`);

        if (actual.population.mismatched !== 0)
            throw new Error(
                `strict parity remains red: ${actual.population.mismatched} current mismatches`,
            );
    },
);
