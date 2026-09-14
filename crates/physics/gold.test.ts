import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

// The retained low-level gold suites cover mechanisms that remain active. The old hull manifold and
// convex-manifold scene vectors stay committed as migration evidence; active hull parity is covered
// by the official immutable command/bundle corpus instead of those predecessor vectors.
check(
    "physics gold tests",
    {
        claim: "the physics crate's C-reference gold vectors catch native solver drift",
        size: "integration",
        budget: 10_000,
        subject: ["crates/physics"],
        requires: ["cargo"],
    },
    () =>
        runCargoTest(
            "shallot-physics",
            "--test",
            "contact_gold",
            "--test",
            "contact_wide_gold",
            "--test",
            "convex_manifold_gold",
            "--test",
            "distance_gold",
            "--test",
            "finalize_gold",
            "--test",
            "integrate_gold",
            "--test",
            "joint_gold",
            "--test",
            "manifold_gold",
            "--test",
            "math_gold",
            "--test",
            "recycle_gold",
            "--test",
            "tree_gold",
        ),
);
