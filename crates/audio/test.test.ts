import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

check(
    "audio",
    {
        claim: "a DSP change breaks the audio crate's Rust suite",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    // An empty libtest filter matches every test: the row runs the lib test target whole.
    () => runCargoTest("shallot-audio", ""),
);
