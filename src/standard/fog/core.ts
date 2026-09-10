// fog's extension + diagnostics surface. The march WGSL chunks (so a custom pass — or the fog probe —
// splices the same integration the production `FogSystem` runs), the `Fog` uniform layout + `packFog`, and
// the CPU-side march oracles the GPU readback is diffed against. Every march primitive is one TGSL function
// that resolves to the spliced WGSL and runs on the CPU, so the chunks and the oracle are the same source.
// The extinction half (S1: `fogMarchWgsl` / `fogTransmittance`), the clustered in-scatter half (S2: light
// shafts — `fogInScatterWgsl` / `henyeyGreenstein` / `fogInScatter`), and the sun half (S3: the directional
// shaft — `sunInScatter` / `fogSunInScatter`). The happy path (`Fog`, `FogPlugin`) is on the index barrel.
export { FogSystem } from "./index";
export type { FogScatter, FogSun } from "./march";
export {
    FOG_BYTES,
    FOG_FLOATS,
    FOG_MAX_STEPS,
    FogGpu,
    fogComposite,
    fogDensity,
    fogInScatter,
    fogInScatterWgsl,
    fogMarchWgsl,
    fogStructWgsl,
    fogSunInScatter,
    fogTransmittance,
    heightOpticalDepth,
    henyeyGreenstein,
    inScatterContribution,
    reconstructWorld,
    sunInScatter,
    WORKGROUP,
} from "./march";
export { packFog } from "./pack";
