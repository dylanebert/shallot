/**
 * The Box3D revision every oracle, corpus lane and gold is held to, and its source tree. A stage moves the
 * pin here and replaces box3d/ beside it with box3d-oracle's lanes generated at it; pin.test.ts holds
 * every lane to the pin, and diagnostics/box3d-parity/native.ts holds BOX3D to it.
 *
 * Authority: Box3D's CMake default build on x86_64, SSE2 at SIMD width 4, the path the kernel's wasm
 * simd128 follows (diagnostics/box3d-parity/native-evidence.ts). box3d-oracle generates the lanes scalar
 * (BOX3D_DISABLE_SIMD=ON) and the scene oracles build the host default, NEON on arm64. At 47d7f7cc and
 * every re-pin stage's revision through 3362633, the scalar and default builds, arm64 and -arch x86_64,
 * hash native.c's scenes equal, so each measures the authority there; where builds differ, SSE2 decides.
 * Under Rosetta, runtime dispatch finds no AVX2 and keeps width 4; an AVX2 host would dispatch width 8,
 * which no oracle here measures.
 */
export const BOX3D_PIN = {
    revision: "954cf879e717334eb96da9a5c255788fdc9f5f87",
    tree: "f112da807e8a98f64339d1456b04ffcdf65a083f",
} as const;
