//! tumble.js physics kernel: the contact-solve + body-integration hot path, ported from box3d's
//! `contact_solver.c` and compiled to wasm-simd128. The TS side owns the API, broadphase,
//! narrowphase, joints, and orchestration; it hands the kernel SoA f32 columns in this module's
//! linear memory and drives it phase by phase (see `arena`).
//!
//! Native `cargo test` exercises the same logic through the scalar `FloatW` fallback (see `simd`),
//! which is bit-identical to the wasm-simd128 path.

// The FloatW fallback methods that only the tests touch today are the solver's foundation; the
// wide-solver port (stage 3b) consumes them. Remove when it lands.
#![allow(dead_code)]

pub mod body;
pub mod col;
pub mod contact;
pub mod contact_wide;
pub mod distance;
pub mod finalize;
pub mod hull;
pub mod integrate;
pub mod joint;
pub mod joint_abi;
pub mod manifold;
pub mod manifold_abi;
pub mod math;
// The convex-manifold bridge (b3ComputeConvexManifold). Native `cargo test` gold-verifies it; the wasm
// build dispatches it over the geometry + manifold columns through `arena::dispatch_convex` (3c.3).
pub mod narrowphase;
pub mod parfor;
pub mod recycle;
mod simd;
pub mod stages;
// The broad-phase pair query + dynamic-tree rebuild (3d). `tree`/`table` are native-testable (gold
// vectors: `tests/tree_gold.rs`); the wasm arena shim that drives them over the resident region is
// `pairwork`, wasm-only.
pub mod table;
pub mod tree;
pub mod wide;

use simd::FloatW;

/// Scratch region the TS loader views as a `Float32Array` to exercise the shared-memory FFI shape and
/// the wasm-simd128 toolchain. Kept until the wide solver (stage 3b) exercises `FloatW` on a wired
/// path — none of the scalar phases wired so far touch simd128, so `smokeScale` is the only cliff gate.
const SCRATCH_LEN: usize = 1024;
static mut SCRATCH: [f32; SCRATCH_LEN] = [0.0; SCRATCH_LEN];

/// Byte offset of the scratch buffer in linear memory; the TS loader builds its view at this address.
#[export_name = "scratchPtr"]
pub extern "C" fn scratch_ptr() -> *mut f32 {
    &raw mut SCRATCH as *mut f32
}

/// Toolchain smoke: scale the first `len` (multiple of 4) scratch floats by `k`, 4 lanes at a time
/// through `FloatW` (simd128 on wasm). Proves simd128 builds, loads, and runs correctly in the JS
/// host — the "wasm-simd cliff on JSC" the spec flags is a hard gate before the wide solver lands.
#[export_name = "smokeScale"]
pub extern "C" fn smoke_scale(len: usize, k: f32) {
    let kw = FloatW::splat(k);
    let p = &raw mut SCRATCH as *mut f32;
    let mut i = 0;
    while i + 4 <= len {
        unsafe {
            let a = FloatW::set(*p.add(i), *p.add(i + 1), *p.add(i + 2), *p.add(i + 3));
            let r = a.mul(kw).to_array();
            *p.add(i) = r[0];
            *p.add(i + 1) = r[1];
            *p.add(i + 2) = r[2];
            *p.add(i + 3) = r[3];
        }
        i += 4;
    }
}

// The shared-column arena + phase export shims are wasm-only: they hand the phase functions slices
// carved straight out of linear memory, which is meaningful only in the JS host. Native `cargo test`
// exercises the phase modules directly against their gold vectors, so the arena is cfg'd out there.
#[cfg(target_arch = "wasm32")]
mod arena;
#[cfg(target_arch = "wasm32")]
mod bodies;
#[cfg(target_arch = "wasm32")]
mod broad;
#[cfg(target_arch = "wasm32")]
mod fataabb;
#[cfg(target_arch = "wasm32")]
mod geo;
#[cfg(target_arch = "wasm32")]
mod manifolds;
#[cfg(target_arch = "wasm32")]
mod pairwork;
#[cfg(target_arch = "wasm32")]
mod shapes;
// The staged solve's wasm entries (`solveBuild` / `solveMt` / `workerMain`), over the arena columns —
// the shared-memory artifact only (`mt`), since a single-thread consumer has no pool to drive them and
// would carry the stage/block tables for nothing. Native `cargo test` drives the same machinery over
// owned columns (`tests/stages.rs`).
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
mod solve;
