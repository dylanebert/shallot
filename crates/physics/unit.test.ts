import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

// Warm run measured 0.79s on macOS arm64 (cargo test --no-run is an untimed requirement step).
check(
    "physics unit tests",
    {
        claim: "the physics crate's native unit tests catch regressions in its scalar solver modules",
        size: "integration",
        budget: 2_000,
        subject: ["crates/physics"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-physics", "--lib"),
);
