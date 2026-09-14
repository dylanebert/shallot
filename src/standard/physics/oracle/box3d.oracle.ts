import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { check } from "../../../harness/check";

const root = join(import.meta.dir, "box3d");
const current = JSON.parse(readFileSync(join(root, "current.json"), "utf8")) as {
    target: string;
    conformsThrough: string;
    bundle: string;
};
const bundle = join(root, current.bundle);
const manifest = JSON.parse(readFileSync(join(bundle, "manifest.json"), "utf8")) as {
    bundle: { upstreamSha: string; schema: string };
    fileDigests: Record<string, string>;
    caseCount?: number;
    inheritedCaseCount?: number;
    scenarioCaseCount?: number;
    corpusDigest?: string;
};
const cases = JSON.parse(readFileSync(join(bundle, "cases.json"), "utf8")) as {
    schema: string;
    cases: Array<{ id: string; family: string; symbol: string; input: unknown; output: unknown }>;
};
const SCENARIO_ROSTER = [
    "free-fall",
    "sphere-drop",
    "box-stack",
    "sphere-sleep",
    "box-sleep",
    "wake-drop",
    "split-slide",
    "revolute-dd",
    "revolute-pendulum",
    "revolute-motor",
    "revolute-limit",
    "revolute-chain",
    "weld-dd",
    "parallel",
    "joint-contacts",
    "motor",
    "motor-spring",
    "distance",
    "distance-spring",
    "prismatic",
    "prismatic-motor",
    "spherical",
    "spherical-limits",
    "spherical-motor",
    "wheel",
    "wheel-spin",
    "wheel-steer",
    "ragdoll",
    "ccd-drop",
    "ccd-bullet",
    "mesh-box",
    "mesh-sphere",
    "mesh-capsule",
    "mesh-ccd",
    "height-box",
    "height-sphere",
    "height-capsule",
    "height-ccd",
    "compound-hull",
    "compound-capsule",
    "compound-sphere",
    "compound-mesh",
    "compound-ccd",
    "sensor",
    "bench-pyramid",
    "bench-many-pyramids",
    "bench-joint-grid",
    "bench-washer",
    "bench-large-world",
    "bench-trees",
    "bench-junkyard",
    "bench-rain",
    "drift",
];

function digest(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

check(
    "immutable Box3D public oracle bundle is selected and intact",
    {
        claim: "Box3D public oracle bundle records target separately from conformance and all generated evidence digests match",
        size: "integration",
    },
    () => {
        if (current.target !== manifest.bundle.upstreamSha)
            throw new Error("current target does not match bundle upstream SHA");
        if (current.conformsThrough === current.target)
            throw new Error("publishing target advanced conformance");
        if (manifest.bundle.schema !== "v6")
            throw new Error("current oracle must select immutable v6");
        if (
            manifest.caseCount !== 111 ||
            manifest.inheritedCaseCount !== 58 ||
            manifest.scenarioCaseCount !== 53
        )
            throw new Error("v6 case population is not the exact 58 + 53 union");
        if (cases.schema !== "box3d-oracle/v6" || cases.cases.length !== 111)
            throw new Error("v6 case file is not the exact 111-case population");
        const scenarioCases = cases.cases.slice(58);
        if (
            JSON.stringify(scenarioCases.map((item) => item.id)) !==
                JSON.stringify(SCENARIO_ROSTER.map((name) => `s1.${name}.v1`)) ||
            scenarioCases.some(
                (item) => item.family !== "scenario" || item.symbol !== "box3d-command-interpreter",
            )
        )
            throw new Error("v6 scenario roster, IDs, or interpreter provenance is not exact");
        const ids = new Set(cases.cases.map((item) => item.id));
        if (ids.size !== 111 || cases.cases.slice(0, 58).some((item) => item.id.startsWith("s1.")))
            throw new Error("v6 contains duplicate, reordered, or non-cumulative case IDs");
        for (const [name, expected] of Object.entries(manifest.fileDigests)) {
            if (digest(join(bundle, name)) !== expected)
                throw new Error(`bundle digest mismatch: ${name}`);
        }
    },
);
