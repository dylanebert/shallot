//! Resident contact and manifold records in Box3D field order. Word views borrow these records;
//! pointers are WASM addresses (native fixtures supply offsets into owned manifold storage).

use crate::col::Col;
use crate::math::{Quat, Transform, Vec3};
use core::mem::{offset_of, size_of};

#[repr(C)]
pub struct ContactEdge {
    pub body_id: i32,
    pub prev_key: i32,
    pub next_key: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union ContactCache {
    pub words: [u32; 4],
    pub sat: crate::manifold::SatCache,
    pub simplex: crate::distance::SimplexCache,
}
#[repr(C)]
pub union ContactCacheRecord {
    pub convex: ContactCache,
    pub mesh: core::mem::ManuallyDrop<crate::mesh_contact::MeshCache>,
}
#[repr(C)]
pub struct ContactRecord {
    pub set_index: i32,
    pub color_index: i32,
    pub local_index: i32,
    pub edges: [ContactEdge; 2],
    pub shape_id_a: i32,
    pub shape_id_b: i32,
    pub child_index: i32,
    pub island_id: i32,
    pub island_index: i32,
    pub contact_id: i32,
    pub body_sim_index_a: i32,
    pub body_sim_index_b: i32,
    pub flags: u32,
    pub manifolds: u32,
    pub manifold_count: i32,
    pub cached_rotation_a: Quat,
    pub cached_rotation_b: Quat,
    pub cached_relative_pose: Transform,
    pub friction: f32,
    pub cache: ContactCacheRecord,
    pub restitution: f32,
    pub rolling_resistance: f32,
    pub tangent_velocity: Vec3,
    pub generation: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ManifoldPointRecord {
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub separation: f32,
    pub base_separation: f32,
    pub normal_impulse: f32,
    pub total_normal_impulse: f32,
    pub normal_velocity: f32,
    pub feature_id: u32,
    pub triangle_index: i32,
    pub persisted: bool,
}
#[repr(C)]
pub struct ManifoldRecord {
    pub points: [ManifoldPointRecord; 4],
    pub normal: Vec3,
    pub twist_impulse: f32,
    pub friction_impulse: Vec3,
    pub rolling_impulse: Vec3,
    pub point_count: i32,
}

pub const DIR_STRIDE: usize = size_of::<ContactRecord>() / 4;
pub const DIR_SET_INDEX: usize = offset_of!(ContactRecord, set_index) / 4;
pub const DIR_COLOR_INDEX: usize = offset_of!(ContactRecord, color_index) / 4;
pub const DIR_LOCAL_INDEX: usize = offset_of!(ContactRecord, local_index) / 4;
pub const DIR_EDGE_A: usize = offset_of!(ContactRecord, edges) / 4;
pub const DIR_EDGE_B: usize = DIR_EDGE_A + size_of::<ContactEdge>() / 4;
pub const DIR_SHAPE_A: usize = offset_of!(ContactRecord, shape_id_a) / 4;
pub const DIR_SHAPE_B: usize = offset_of!(ContactRecord, shape_id_b) / 4;
pub const DIR_CHILD_INDEX: usize = offset_of!(ContactRecord, child_index) / 4;
pub const DIR_ISLAND_ID: usize = offset_of!(ContactRecord, island_id) / 4;
pub const DIR_ISLAND_INDEX: usize = offset_of!(ContactRecord, island_index) / 4;
pub const DIR_CONTACT_ID: usize = offset_of!(ContactRecord, contact_id) / 4;
pub const DIR_GENERATION: usize = offset_of!(ContactRecord, generation) / 4;
pub const DIR_FRICTION: usize = offset_of!(ContactRecord, friction) / 4;
pub const DIR_RESTITUTION: usize = offset_of!(ContactRecord, restitution) / 4;
pub const DIR_ROLLING_RESISTANCE: usize = offset_of!(ContactRecord, rolling_resistance) / 4;
pub const DIR_TANGENT_VELOCITY: usize = offset_of!(ContactRecord, tangent_velocity) / 4;
pub const DIR_FLAGS: usize = offset_of!(ContactRecord, flags) / 4;
pub const DIR_MANIFOLD_COUNT: usize = offset_of!(ContactRecord, manifold_count) / 4;
pub const DIR_MANIFOLD_BASE: usize = offset_of!(ContactRecord, manifolds) / 4;
pub const DIR_INDEX_A: usize = offset_of!(ContactRecord, body_sim_index_a) / 4;
pub const DIR_INDEX_B: usize = offset_of!(ContactRecord, body_sim_index_b) / 4;
pub const DIR_CACHE: usize = offset_of!(ContactRecord, cache) / 4;
pub const DIR_MESH_CACHE: usize = DIR_CACHE;
pub const DIR_CACHED_ROT_A: usize = offset_of!(ContactRecord, cached_rotation_a) / 4;
pub const DIR_CACHED_ROT_B: usize = offset_of!(ContactRecord, cached_rotation_b) / 4;
pub const DIR_CACHED_REL_POSE: usize = offset_of!(ContactRecord, cached_relative_pose) / 4;
pub const MANIFOLD_STRIDE: usize = size_of::<ManifoldRecord>() / 4;
pub const M_NORMAL: usize = offset_of!(ManifoldRecord, normal) / 4;
pub const M_FRICTION: usize = offset_of!(ManifoldRecord, friction_impulse) / 4;
pub const M_TWIST: usize = offset_of!(ManifoldRecord, twist_impulse) / 4;
pub const M_ROLLING: usize = offset_of!(ManifoldRecord, rolling_impulse) / 4;
pub const M_POINT_COUNT: usize = offset_of!(ManifoldRecord, point_count) / 4;
pub const M_POINTS: usize = offset_of!(ManifoldRecord, points) / 4;
pub const POOL_POINT_STRIDE: usize = size_of::<ManifoldPointRecord>() / 4;
pub const P_ANCHOR_A: usize = offset_of!(ManifoldPointRecord, anchor_a) / 4;
pub const P_ANCHOR_B: usize = offset_of!(ManifoldPointRecord, anchor_b) / 4;
pub const P_SEPARATION: usize = offset_of!(ManifoldPointRecord, separation) / 4;
pub const P_BASE_SEPARATION: usize = offset_of!(ManifoldPointRecord, base_separation) / 4;
pub const P_NORMAL_IMPULSE: usize = offset_of!(ManifoldPointRecord, normal_impulse) / 4;
pub const P_TOTAL_NORMAL_IMPULSE: usize = offset_of!(ManifoldPointRecord, total_normal_impulse) / 4;
pub const P_NORMAL_VELOCITY: usize = offset_of!(ManifoldPointRecord, normal_velocity) / 4;
pub const P_FEATURE_ID: usize = offset_of!(ManifoldPointRecord, feature_id) / 4;
pub const P_TRIANGLE_INDEX: usize = offset_of!(ManifoldPointRecord, triangle_index) / 4;
pub const P_PERSISTED: usize = offset_of!(ManifoldPointRecord, persisted) / 4;

/// The directory record the solver gathers for one contact.
pub struct DirEntry {
    pub friction: f32,
    pub restitution: f32,
    pub rolling_resistance: f32,
    pub tangent_velocity: Vec3,
    pub flags: u32,
    pub manifold_count: usize,
    pub manifold_base: usize,
    pub index_a: u32,
    pub index_b: u32,
}

/// Read contact `contact_id`'s directory record (material row + body indices + block descriptor).
#[inline]
pub fn read_dir(dir: Col<u32>, contact_id: usize) -> DirEntry {
    let o = contact_id * DIR_STRIDE;
    DirEntry {
        friction: f32::from_bits(dir.get(o + DIR_FRICTION)),
        restitution: f32::from_bits(dir.get(o + DIR_RESTITUTION)),
        rolling_resistance: f32::from_bits(dir.get(o + DIR_ROLLING_RESISTANCE)),
        tangent_velocity: Vec3::new(
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY)),
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY + 1)),
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY + 2)),
        ),
        flags: dir.get(o + DIR_FLAGS),
        manifold_count: dir.get(o + DIR_MANIFOLD_COUNT) as usize,
        manifold_base: dir.get(o + DIR_MANIFOLD_BASE) as usize,
        index_a: dir.get(o + DIR_INDEX_A),
        index_b: dir.get(o + DIR_INDEX_B),
    }
}

/// Resolve a contact's stable block. The native harness supplies owned storage rather than linear-memory addresses.
#[inline]
pub fn block_col(pool: Col<f32>, base: usize, count: usize) -> Col<f32> {
    #[cfg(target_arch = "wasm32")]
    let ptr = base as *mut f32;
    #[cfg(not(target_arch = "wasm32"))]
    let ptr = unsafe { pool.ptr().add(base * MANIFOLD_STRIDE) };
    #[cfg(target_arch = "wasm32")]
    let _ = pool;
    unsafe { Col::new(ptr, count * MANIFOLD_STRIDE) }
}
