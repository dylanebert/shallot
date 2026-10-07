//! Box3D world and compute kernel, compiled to wasm-simd128. TypeScript binds the public API,
//! bridges ECS authoring and placement, drives the worker pool, and runs user callbacks at the
//! kernel's serial points.
//!
//! Native `cargo test` exercises the same logic through the scalar `FloatW` fallback (see `simd`),
//! which is bit-identical to the wasm-simd128 path.

// The FloatW fallback methods that only the tests touch today are the solver's foundation; the
// wide-solver port consumes them. Remove when it lands.
#![allow(dead_code)]
// These lints are intentionally allowed for the C-reference port: argument-heavy phase entry
// points, explicit index loops, named f32 constants, and method names mirror the upstream seams
// and preserve operation order/bit behavior rather than following idiomatic Rust rewrites.
#![allow(clippy::approx_constant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::too_many_arguments)]

mod aabb;
pub mod body;
pub mod col;
#[cfg(target_arch = "wasm32")]
mod constraint_graph;
pub mod contact;
pub mod contact_wide;
#[cfg(target_arch = "wasm32")]
pub mod continuous;
pub mod continuous_shape;
pub mod distance;
pub mod finalize;
#[cfg(target_arch = "wasm32")]
mod geometry_database;
pub mod height_query;
pub mod hull;
pub mod integrate;
#[cfg(target_arch = "wasm32")]
mod island;
pub mod joint;
pub mod joint_abi;
#[cfg(target_arch = "wasm32")]
mod joints;
pub mod manifold;
pub mod manifold_abi;
pub mod math;
pub mod mesh_contact;
pub mod mesh_query;
pub mod narrowphase;
pub mod parfor;
#[cfg(target_arch = "wasm32")]
mod physics_world;
pub mod query;
pub mod recycle;
mod simd;
#[cfg(target_arch = "wasm32")]
mod solver_set;
pub mod stages;
mod toi;
pub mod triangle_manifold;
// The broad-phase pair query + dynamic-tree rebuild (3d). `tree`/`table` are native-testable (gold
// vectors: `tests/tree_gold.rs`); the wasm arena shim that drives them over the resident region is
// `pairwork`, wasm-only.
pub mod table;
pub mod tree;
pub mod wide;

/// In a checked build (`build-kernel.ts --checked`), a panic hands its message to the host before the
/// abort traps, so a failed bounds or precondition check names its file and line instead of only
/// "unreachable".
#[cfg(all(target_arch = "wasm32", debug_assertions))]
mod checked {
    #[link(wasm_import_module = "env")]
    extern "C" {
        fn kernelPanic(message: *const u8, len: usize);
    }
    #[export_name = "installPanicHook"]
    pub extern "C" fn install_panic_hook() {
        std::panic::set_hook(Box::new(|info| {
            let message = info.to_string();
            // SAFETY: the host reads `len` bytes at `message` before returning, while it is alive.
            unsafe { kernelPanic(message.as_ptr(), message.len()) }
        }));
    }
}

// The shared-column arena + phase export shims are wasm-only: they hand the phase functions slices
// carved straight out of linear memory, which is meaningful only in the JS host. Native `cargo test`
// exercises the phase modules directly against their gold vectors, so the arena is cfg'd out there.
#[cfg(all(target_arch = "wasm32", feature = "count-allocations"))]
mod allocation;
#[cfg(target_arch = "wasm32")]
mod arena;
#[cfg(target_arch = "wasm32")]
mod bodies;
#[cfg(target_arch = "wasm32")]
mod body_mutation;
#[cfg(target_arch = "wasm32")]
mod body_query;
mod body_record;
#[cfg(all(target_arch = "wasm32", feature = "box3d-oracle"))]
mod box3d_oracle;
#[cfg(target_arch = "wasm32")]
mod broad;
#[cfg(target_arch = "wasm32")]
mod compound_query;
#[cfg(target_arch = "wasm32")]
mod contact_layout;
#[cfg(target_arch = "wasm32")]
mod contact_list;
#[cfg(target_arch = "wasm32")]
mod draw;
#[cfg(any(target_arch = "wasm32", test))]
mod events;
#[cfg(target_arch = "wasm32")]
#[cfg(target_arch = "wasm32")]
mod geo;
#[cfg(any(target_arch = "wasm32", test))]
mod hull_database;
#[cfg(target_arch = "wasm32")]
mod joint_creation;
mod joint_draw;
#[cfg(target_arch = "wasm32")]
mod joint_lifecycle;
mod joint_record;
#[cfg(target_arch = "wasm32")]
mod manifolds;
mod mover;
#[cfg(target_arch = "wasm32")]
mod pairwork;
#[cfg(target_arch = "wasm32")]
mod query_abi;
#[cfg(target_arch = "wasm32")]
mod regions;
#[cfg(target_arch = "wasm32")]
mod sensor;
mod shape_geometry;
#[cfg(target_arch = "wasm32")]
mod shape_lifecycle;
#[cfg(target_arch = "wasm32")]
mod shapes;
#[cfg(target_arch = "wasm32")]
mod treework;
#[cfg(target_arch = "wasm32")]
mod world_query;
// Both wasm artifacts run the staged solve over arena columns. Native tests use owned columns.
#[cfg(target_arch = "wasm32")]
mod solve;
