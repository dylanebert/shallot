//! Each World's contact directory and retained manifold pool own their allocations.
use crate::col::Col;
use crate::manifold_abi::{DIR_STRIDE, MANIFOLD_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};
static mut COLUMNS: [Columns<2>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];
static mut MESH_CACHES: [Vec<Columns<1>>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];

#[export_name = "meshCacheCapacity"]
pub extern "C" fn mesh_cache_capacity(world: usize) -> usize {
    assert!(world < MAX_WORLDS);
    unsafe { MESH_CACHES[world].capacity() }
}

#[export_name = "ensureMeshCache"]
pub extern "C" fn ensure_mesh_cache(contact: usize) {
    unsafe {
        let caches = &mut MESH_CACHES[regions::active()];
        caches.resize(caches.len().max(contact + 1), Columns::EMPTY);
        if caches[contact].layout[0] == 16 {
            caches[contact].reserve(0, core::mem::size_of::<crate::mesh_contact::MeshCache>());
            let cache = &mut *(caches[contact].layout[0] as *mut crate::mesh_contact::MeshCache);
            cache.lower = crate::math::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            cache.upper = crate::math::Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX);
        }
    }
}
#[export_name = "freeMeshCache"]
pub extern "C" fn free_mesh_cache(contact: usize) {
    unsafe {
        if let Some(cache) = MESH_CACHES[regions::active()].get_mut(contact) {
            cache.release();
        }
    }
}
pub unsafe fn mesh_cache_ptr(contact: usize) -> *mut crate::mesh_contact::MeshCache {
    MESH_CACHES[regions::active()][contact].layout[0] as *mut crate::mesh_contact::MeshCache
}

static mut CAPS: [[usize; 2]; MAX_WORLDS] = [[0; 2]; MAX_WORLDS];
pub fn dir_col() -> Col<'static, u32> {
    unsafe {
        let id = regions::active();
        Col::new(COLUMNS[id].layout[0] as *mut u32, CAPS[id][0] * DIR_STRIDE)
    }
}
pub fn pool_col() -> Col<'static, f32> {
    unsafe {
        let id = regions::active();
        Col::new(
            COLUMNS[id].layout[1] as *mut f32,
            CAPS[id][1] * MANIFOLD_STRIDE,
        )
    }
}
#[export_name = "copyManifolds"]
pub extern "C" fn copy_manifolds(source: usize, base: usize, count: usize) {
    unsafe {
        core::ptr::copy_nonoverlapping(
            source as *const u32,
            (COLUMNS[regions::active()].layout[1] as *mut u32).add(base * MANIFOLD_STRIDE),
            count * MANIFOLD_STRIDE,
        );
    }
}

#[export_name = "manifoldLayoutPtr"]
pub extern "C" fn manifold_layout_ptr() -> *const u32 {
    unsafe { COLUMNS[regions::active()].layout.as_ptr() }
}
#[export_name = "reserveManifolds"]
pub extern "C" fn reserve_manifolds(contact_cap: usize, manifold_cap: usize) {
    unsafe {
        let id = regions::active();
        for (column, cap, stride) in [
            (0, contact_cap, DIR_STRIDE),
            (1, manifold_cap, MANIFOLD_STRIDE),
        ] {
            let cap = cap.max(CAPS[id][column]);
            COLUMNS[id].reserve(column, cap * stride * 4);
            CAPS[id][column] = cap;
        }
    }
}
pub unsafe fn reset(id: usize) {
    for cache in &mut MESH_CACHES[id] {
        cache.release();
    }
    MESH_CACHES[id] = Vec::new();
    COLUMNS[id].release();
    CAPS[id] = [0; 2];
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    for value in CAPS[id] {
        regions::write_word(out, value);
    }
    COLUMNS[id].snapshot(out);
    regions::write_word(out, MESH_CACHES[id].len());
    for cache in &MESH_CACHES[id] {
        cache.snapshot(out);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    for value in &mut CAPS[id] {
        *value = regions::read_word(input);
    }
    COLUMNS[id].restore(input);
    for cache in &mut MESH_CACHES[id] {
        cache.release();
    }
    let len = regions::read_word(input);
    MESH_CACHES[id].resize(len, Columns::EMPTY);
    for cache in &mut MESH_CACHES[id] {
        cache.restore(input);
    }
}
