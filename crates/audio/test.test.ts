import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

check(
    "audio convolution",
    {
        claim: "the audio crate's convolution:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "convolution::"),
);
check(
    "audio delay",
    {
        claim: "the audio crate's delay:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "delay::"),
);
check(
    "audio dynamics",
    {
        claim: "the audio crate's dynamics:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "dynamics::"),
);
check(
    "audio envelope",
    {
        claim: "the audio crate's envelope:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "envelope::"),
);
check(
    "audio fft",
    {
        claim: "the audio crate's fft:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "fft::"),
);
check(
    "audio filter",
    {
        claim: "the audio crate's filter:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "filter::"),
);
check(
    "audio graph",
    {
        claim: "the audio crate's graph:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "graph::"),
);
check(
    "audio hrtf",
    {
        claim: "the audio crate's hrtf:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "hrtf::"),
);
check(
    "audio interp",
    {
        claim: "the audio crate's interp:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "interp::"),
);
check(
    "audio modulation",
    {
        claim: "the audio crate's modulation:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "modulation::"),
);
check(
    "audio oscillator",
    {
        claim: "the audio crate's oscillator:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "oscillator::"),
);
check(
    "audio sample",
    {
        claim: "the audio crate's sample:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "sample::"),
);
check(
    "audio root tests",
    {
        claim: "the audio crate's tests:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "tests::"),
);
check(
    "audio waveshaper",
    {
        claim: "the audio crate's waveshaper:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "waveshaper::"),
);
