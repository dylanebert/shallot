import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

// Warm run measured 3.42s on macOS arm64 (cargo test --no-run is an untimed requirement step).
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
