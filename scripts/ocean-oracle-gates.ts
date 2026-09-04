export interface CpuGate {
    name: string;
    script: string;
    /** Declared bun per-test ceiling in milliseconds. */
    timeoutMs: number;
    /** Largest recorded arm divided by the declared ceiling at the S1c baseline. */
    recordedFraction: number;
    covers: string[];
}

/** CPU-only by-path work. Each cone is transcribed from the owning oracle's Trigger header. */
export const OCEAN_CPU_GATES: CpuGate[] = [
    {
        name: "ocean realization",
        script: "test:ocean-realization",
        timeoutMs: 20_000,
        recordedFraction: 0.47,
        covers: [
            "examples/showcase/ocean/test/realization.oracle.ts",
            "examples/showcase/ocean/test/*physical-spectrum*.test.ts",
            "examples/showcase/ocean/src/ocean/{spectrum,cpu-reference,fft}.ts",
        ],
    },
    {
        name: "ocean slope",
        script: "test:ocean-slope",
        timeoutMs: 10_000,
        recordedFraction: 0.07,
        covers: [
            "examples/showcase/ocean/test/slope.oracle.ts",
            "examples/showcase/ocean/src/ocean/{slope,spectrum}.ts",
        ],
    },
    {
        name: "ocean mesh inversion",
        script: "test:ocean-mesh-inversion",
        timeoutMs: 10_000,
        recordedFraction: 0.2,
        covers: [
            "examples/showcase/ocean/test/mesh-inversion-sweep.oracle.ts",
            "examples/showcase/ocean/src/ocean/{clipmap,reconstruction,cpu-reference,spectrum}.ts",
        ],
    },
    {
        name: "ocean fold",
        script: "test:ocean-fold",
        timeoutMs: 30_000,
        recordedFraction: 0.11,
        covers: [
            "examples/showcase/ocean/test/fold-anchor.oracle.ts",
            "examples/showcase/ocean/src/ocean/{spectrum,cpu-reference,composed-fold,reconstruction,ocean}.ts",
        ],
    },
];
