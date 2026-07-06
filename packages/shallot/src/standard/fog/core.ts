// fog's extension + diagnostics surface. The march WGSL chunks (so a custom pass — or the fog probe —
// splices the same integration the production `FogSystem` runs), the Fog uniform layout + `packFog`, and
// the TS oracles the GPU twins are pinned to (the fog probe-readback assert diffs against them). The
// extinction half (S1: `FOG_MARCH_WGSL` / `fogTransmittance`), the clustered in-scatter half (S2: light
// shafts — `FOG_INSCATTER_WGSL` / `henyeyGreenstein` / `fogInScatter`), and the sun half (S3: the
// directional shaft — `sunInScatter` / `fogSunInScatter`). The happy path (`Fog`, `FogPlugin`) is on the
// index barrel.
export { FogSystem } from "./index";
export type { FogLight, FogScatter, FogSun } from "./march";
export {
    FOG_BYTES,
    FOG_FLOATS,
    FOG_INSCATTER_WGSL,
    FOG_MARCH_WGSL,
    FOG_MAX_STEPS,
    FOG_STRUCT_WGSL,
    fogComposite,
    fogDensity,
    fogInScatter,
    fogSunInScatter,
    fogTransmittance,
    heightOpticalDepth,
    henyeyGreenstein,
    inScatterContribution,
    reconstruct,
    sunInScatter,
    WORKGROUP,
} from "./march";
export { packFog } from "./pack";
