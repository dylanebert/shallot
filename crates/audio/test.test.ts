import { check } from "@dylanebert/shallot/harness/check";
import { discoverCargoTestPartitions, runCargoTest } from "../../scripts/cargo-test";

// Cargo --no-run and libtest --list are untimed discovery; each direct module partition has the
// ordinary 20-second integration ceiling for test execution only.
const discovered =
    process.env.SHALLOT_UNIT_ONLY === "1" ? null : discoverCargoTestPartitions("shallot-audio");
const partitionByFilter = new Map(discovered?.map((partition) => [partition.filter, partition]));
function partition(filter: string): { filter: string } {
    const found = partitionByFilter.get(filter);
    if (discovered !== null && (found === undefined || found.tests.length === 0)) {
        throw new Error(`audio declaration has no non-empty libtest partition: ${filter}`);
    }
    return { filter };
}

const convolution = partition("convolution::");
const delay = partition("delay::");
const dynamics = partition("dynamics::");
const envelope = partition("envelope::");
const fft = partition("fft::");
const filter = partition("filter::");
const graph = partition("graph::");
const hrtf = partition("hrtf::");
const interp = partition("interp::");
const modulation = partition("modulation::");
const oscillator = partition("oscillator::");
const sample = partition("sample::");
const tests = partition("tests::");
const waveshaper = partition("waveshaper::");

if (discovered !== null) {
    const declared = [
        convolution.filter,
        delay.filter,
        dynamics.filter,
        envelope.filter,
        fft.filter,
        filter.filter,
        graph.filter,
        hrtf.filter,
        interp.filter,
        modulation.filter,
        oscillator.filter,
        sample.filter,
        tests.filter,
        waveshaper.filter,
    ];
    if (
        declared.length !== discovered.length ||
        new Set(declared).size !== discovered.length ||
        [...declared].sort().join("\n") !==
            discovered
                .map((candidate) => candidate.filter)
                .sort()
                .join("\n")
    ) {
        throw new Error("audio declarations are not an exact union of discovered libtest partitions");
    }
}

check(
    "audio convolution",
    {
        claim: "the audio crate's convolution:: native DSP tests catch signal-processing regressions",
        size: "integration",
        budget: 20_000,
        subject: ["crates/audio"],
        requires: ["cargo"],
    },
    () => runCargoTest("shallot-audio", convolution.filter),
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
    () => runCargoTest("shallot-audio", delay.filter),
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
    () => runCargoTest("shallot-audio", dynamics.filter),
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
    () => runCargoTest("shallot-audio", envelope.filter),
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
    () => runCargoTest("shallot-audio", fft.filter),
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
    () => runCargoTest("shallot-audio", filter.filter),
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
    () => runCargoTest("shallot-audio", graph.filter),
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
    () => runCargoTest("shallot-audio", hrtf.filter),
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
    () => runCargoTest("shallot-audio", interp.filter),
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
    () => runCargoTest("shallot-audio", modulation.filter),
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
    () => runCargoTest("shallot-audio", oscillator.filter),
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
    () => runCargoTest("shallot-audio", sample.filter),
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
    () => runCargoTest("shallot-audio", tests.filter),
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
    () => runCargoTest("shallot-audio", waveshaper.filter),
);
