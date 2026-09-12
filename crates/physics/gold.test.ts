import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

// Warm run measured 4.04s on macOS arm64 with every physics integration target (cargo test --no-run is an untimed requirement step).
check(
    "physics gold tests",
    {
        claim: "the physics crate's C-reference gold vectors catch native solver drift",
        size: "integration",
        budget: 10_000,
        subject: ["crates/physics"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-physics", "--tests"),
);
