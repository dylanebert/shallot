import { check } from "@dylanebert/shallot/harness/check";
import { runCargoTest } from "../../scripts/cargo-test";

check(
    "audio convolution",
    {
        claim: "a convolution change misapplies the impulse response, keeps stale history or clicks on an IR update",
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
        claim: "a delay line change shifts, loses or mis-mixes delayed samples",
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
        claim: "a compressor or expander change applies the wrong gain around its threshold",
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
        claim: "an envelope change mistimes attack, decay, sustain or release stages",
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
        claim: "an FFT change returns the wrong magnitudes or phases for known spectra",
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
        claim: "a filter change shifts its cutoff, resonance or stability",
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
        claim: "an audio graph change routes, orders or mixes nodes wrongly",
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
        claim: "an HRTF change gives a positioned source the wrong interaural delay or level",
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
        claim: "an interpolation change produces wrong in-between samples or parameter ramps",
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
        claim: "a modulation change lets its LFO drift out of bounds or period, or its allpass lose energy",
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
        claim: "an oscillator change produces the wrong waveform range, harmonics or band-limiting",
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
        claim: "a sample playback change reads, loops or pitches the source buffer wrongly",
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
        claim: "a voice, transport or event scheduling change mistimes gates, loops or sample-offset events",
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
        claim: "a waveshaper change applies the wrong clip or fold curve, or leaves a DC offset",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", "waveshaper::"),
);
