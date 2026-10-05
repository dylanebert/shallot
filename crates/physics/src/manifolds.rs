//! World-owned contact directories and Box3D's manifold-count block allocators.
use crate::col::Col;
use crate::manifold_abi::{DIR_MANIFOLD_BASE, DIR_STRIDE, MANIFOLD_STRIDE};
use crate::regions::{self, Buffer, Columns, MAX_WORLDS};

static mut COLUMNS: [Columns<1>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];
static mut CAPS: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut ALLOCATORS: [Vec<BlockAllocator>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
static mut MESH_CACHES: [Vec<Columns<1>>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];

const BLOCK_SIZE: usize = 256;
struct BlockAllocator {
    chunks: Vec<Buffer>,
    element_size: usize,
    free: usize,
    next: usize,
}
impl BlockAllocator {
    unsafe fn new(count: usize) -> Self {
        let mut allocator = Self {
            chunks: Vec::new(),
            element_size: (count * MANIFOLD_STRIDE * 4 + 15) & !15,
            free: 0,
            next: 0,
        };
        // b3CreateBlockAllocator initializes two chunks for each size class.
        allocator.add_chunk();
        allocator.add_chunk();
        allocator
    }
    unsafe fn add_chunk(&mut self) {
        let mut chunk = Buffer::EMPTY;
        chunk.reserve(BLOCK_SIZE * self.element_size);
        self.chunks.push(chunk);
    }
    unsafe fn allocate(&mut self) -> usize {
        if self.free != 0 {
            let address = self.free;
            self.free = *(address as *const u32) as usize;
            return address;
        }
        let index = self.next;
        self.next += 1;
        if index / BLOCK_SIZE == self.chunks.len() {
            self.add_chunk();
        }
        self.chunks[index / BLOCK_SIZE].ptr + (index % BLOCK_SIZE) * self.element_size
    }
    unsafe fn free(&mut self, address: usize) {
        *(address as *mut u32) = self.free as u32;
        self.free = address;
    }
    unsafe fn release(&mut self) {
        for chunk in &mut self.chunks {
            chunk.release();
        }
    }
}

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

pub fn dir_col() -> Col<'static, u32> {
    unsafe {
        let id = regions::active();
        Col::new(COLUMNS[id].layout[0] as *mut u32, CAPS[id] * DIR_STRIDE)
    }
}
// Native fixtures supply an owned backing column; WASM addresses blocks directly.
pub fn pool_col() -> Col<'static, f32> {
    unsafe { Col::new(16 as *mut f32, 0) }
}
unsafe fn allocate(id: usize, contact: usize, count: usize) -> usize {
    free(id, contact);
    if count == 0 {
        return 0;
    }
    let allocators = &mut ALLOCATORS[id];
    while allocators.len() < count {
        allocators.push(BlockAllocator::new(allocators.len() + 1));
    }
    let address = allocators[count - 1].allocate();
    (address as *mut u8).write_bytes(0, count * MANIFOLD_STRIDE * 4);
    let dir = COLUMNS[id].layout[0] as *mut u32;
    *dir.add(contact * DIR_STRIDE + 7) = count as u32;
    *dir.add(contact * DIR_STRIDE + DIR_MANIFOLD_BASE) = address as u32;
    address
}
unsafe fn free(id: usize, contact: usize) {
    let dir = COLUMNS[id].layout[0] as *mut u32;
    let count = *dir.add(contact * DIR_STRIDE + 7) as usize;
    if count != 0 {
        let address = *dir.add(contact * DIR_STRIDE + DIR_MANIFOLD_BASE) as usize;
        ALLOCATORS[id][count - 1].free(address);
        *dir.add(contact * DIR_STRIDE + 7) = 0;
        *dir.add(contact * DIR_STRIDE + DIR_MANIFOLD_BASE) = 0;
    }
}
#[export_name = "allocateManifolds"]
pub extern "C" fn allocate_manifolds(contact: usize, count: usize) -> usize {
    unsafe { allocate(regions::active(), contact, count) }
}
#[export_name = "freeManifolds"]
pub extern "C" fn free_manifolds(contact: usize) {
    unsafe {
        free(regions::active(), contact);
    }
}
#[export_name = "copyManifolds"]
pub extern "C" fn copy_manifolds(source: usize, address: usize, count: usize) {
    unsafe {
        core::ptr::copy_nonoverlapping(
            source as *const u32,
            address as *mut u32,
            count * MANIFOLD_STRIDE,
        );
    }
}
#[export_name = "manifoldLayoutPtr"]
pub extern "C" fn manifold_layout_ptr() -> *const u32 {
    unsafe { COLUMNS[regions::active()].layout.as_ptr() }
}
#[export_name = "reserveManifolds"]
pub extern "C" fn reserve_manifolds(contact_cap: usize, _manifold_cap: usize) {
    unsafe {
        let id = regions::active();
        let cap = contact_cap.max(CAPS[id]);
        COLUMNS[id].reserve(0, cap * DIR_STRIDE * 4);
        CAPS[id] = cap;
    }
}
unsafe fn release_allocators(id: usize) {
    for allocator in &mut ALLOCATORS[id] {
        allocator.release();
    }
    ALLOCATORS[id] = Vec::new();
}
pub unsafe fn reset(id: usize) {
    release_allocators(id);
    for cache in &mut MESH_CACHES[id] {
        cache.release();
    }
    MESH_CACHES[id] = Vec::new();
    COLUMNS[id].release();
    CAPS[id] = 0;
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, CAPS[id]);
    let dir =
        core::slice::from_raw_parts(COLUMNS[id].layout[0] as *const u32, CAPS[id] * DIR_STRIDE);
    // world_snapshot.c serializes contents, never allocator positions or free chunks.
    for (index, &word) in dir.iter().enumerate() {
        regions::write_word(
            out,
            if index % DIR_STRIDE == DIR_MANIFOLD_BASE {
                0
            } else {
                word as usize
            },
        );
    }
    for contact in 0..CAPS[id] {
        let count = dir[contact * DIR_STRIDE + 7] as usize;
        if count != 0 {
            let address = dir[contact * DIR_STRIDE + DIR_MANIFOLD_BASE] as usize;
            out.extend_from_slice(core::slice::from_raw_parts(
                address as *const u8,
                count * MANIFOLD_STRIDE * 4,
            ));
        }
    }
    regions::write_word(out, MESH_CACHES[id].len());
    for cache in &MESH_CACHES[id] {
        cache.snapshot(out);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    CAPS[id] = regions::read_word(input);
    let bytes = CAPS[id] * DIR_STRIDE * 4;
    COLUMNS[id].reserve(0, bytes);
    let (data, rest) = input.split_at(bytes);
    core::ptr::copy_nonoverlapping(data.as_ptr(), COLUMNS[id].layout[0] as *mut u8, bytes);
    *input = rest;
    let dir = COLUMNS[id].layout[0] as *mut u32;
    for contact in 0..CAPS[id] {
        let count = *dir.add(contact * DIR_STRIDE + 7) as usize;
        if count != 0 {
            *dir.add(contact * DIR_STRIDE + 7) = 0;
            let address = allocate(id, contact, count);
            let (data, rest) = input.split_at(count * MANIFOLD_STRIDE * 4);
            core::ptr::copy_nonoverlapping(data.as_ptr(), address as *mut u8, data.len());
            *input = rest;
        }
    }
    let len = regions::read_word(input);
    MESH_CACHES[id].resize(len, Columns::EMPTY);
    for cache in &mut MESH_CACHES[id] {
        cache.restore(input);
    }
}
