/** Categorical backstops against hangs and runaway work, not performance targets. */
export const CEILING = {
    cheap: 250,
    gpu: 1000,
    node: 20_000,
    browser: 60_000,
    startup: 120_000,
} as const;
