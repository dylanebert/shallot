import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

// Warm run measured 19.95s on macOS arm64 (cargo test --no-run is an untimed requirement step).
check(
    "audio tests",
    {
        claim: "the audio crate's native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio"),
);
