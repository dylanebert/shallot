//! The kernel's integration tests, in one binary: each file is a module, so a kernel edit relinks and
//! launches one executable instead of one per file.

// `joint_draw` names the kernel through `crate::`, so its source compiles here at the crate root
// beside the three modules it reads.
use shallot_physics::{col, joint_abi, math};
#[path = "../../src/joint_draw.rs"]
#[allow(dead_code)]
mod joint_draw;

mod character_gold;
mod contact_gold;
mod contact_wide_gold;
mod continuous_shape;
mod convex_manifold_gold;
mod distance_gold;
mod geometry_read_gold;
mod hull_gold;
mod integrate_gold;
mod joint_gold;
mod joint_layout;
mod joint_motion_gold;
mod joint_reaction;
mod joint_read_gold;
mod manifold_gold;
mod math_gold;
mod math_invariants;
mod mesh_query_gold;
mod nonconvex_gold;
mod query_gold;
mod stages;
mod tree_gold;
