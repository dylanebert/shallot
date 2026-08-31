// @dylanebert/shallot-ocean — the FFT ocean surface extension. First-party shallot package (I1):
// the water-surface spike's direct O(N²) DFT becomes a butterfly FFT here (`./gpu-fft.ts`,
// `./fft.ts`), landing in a package for the first time — no other first-party `@dylanebert/shallot-*`
// extension existed before this one. Spectrum normalization (I2), the capillary slope cascade (I3),
// and fragment-side slope-texture shading (I4) are later stages of the same spec; this package
// currently ships the substrate those stages extend.

export {
    type ComplexArray,
    type CpuStageResult,
    chop,
    idft2,
    type JacobianStats,
    jacobianStats,
    runCpuPipeline,
    spectralGradient,
    updateH,
} from "./cpu-reference";
export { directIdft2, fft1dInPlace, ifft2 } from "./fft";
export {
    type FftLayout,
    getFftKernels,
    makeColFftKernel,
    makeFftLayout,
    makeRowFftKernel,
} from "./gpu-fft";
export {
    type CascadeConfig,
    CascadeParams,
    Complex,
    getCascadeConfigs,
    getDisplacementTexture,
    getProbeBufferForCascade,
    measureFoldFraction,
    OceanPlugin,
    oceanCompute,
    PEAK_FMA_FLOPS,
    PROBE_COUNTS,
    PROBE_TOTAL,
    ProbeData,
    type ProbeRow,
    readStageBuffers,
    type StageBuffers,
    setPeakFmaEnabled,
    updateKernel,
    updateLayout,
} from "./ocean";
export {
    centeredLabelPreFix,
    type LabelFn,
    lag1AutocorrParityWitness,
    measuredLag1AutocorrX,
    type ParityWitnessReading,
} from "./parity-witness";
export {
    assertAllPowerOfTwo,
    assertCoprimeL,
    CASCADE_CONFIGS,
    directDftFlops,
    G,
    gcd,
    generateH0,
    isPowerOfTwo,
    kIndex,
    philips,
    theoreticalFlops,
    tilePeriod,
    totalTheoreticalFlops,
} from "./spectrum";
