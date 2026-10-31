/**
 * The Box3D revision every oracle, corpus lane and gold is held to, and its source tree. A stage moves the
 * pin here and replaces box3d/ beside it with box3d-oracle's lanes generated at it; pin.test.ts holds
 * every lane to the pin, and diagnostics/box3d-parity/native.ts holds BOX3D to it.
 *
 * Authority: Box3D's CMake default build on x86_64, SSE2 at SIMD width 4, the path the kernel's wasm
 * simd128 follows (diagnostics/box3d-parity/native-evidence.ts). box3d-oracle generates the lanes scalar
 * (BOX3D_DISABLE_SIMD=ON) and the scene oracles build the host default, NEON on arm64. At the original pin and
 * every re-pin stage's revision through 3362633, the scalar and default builds, arm64 and -arch x86_64,
 * hash native.c's scenes equal, so each measures the authority there; where builds differ, SSE2 decides.
 * Under Rosetta, runtime dispatch finds no AVX2 and keeps width 4; an AVX2 host would dispatch width 8,
 * which no oracle here measures.
 */
export const BOX3D_PIN = {
    revision: "336263320469e5e9febfe18697ab562488f7e97a",
    tree: "50877041b7e4de95ff0b6c194182aa50d453b736",
} as const;
