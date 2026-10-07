//! World-owned contact directories and Box3D's manifold-count block allocators.
use crate::col::Col;
use crate::manifold_abi::{DIR_MANIFOLD_BASE, DIR_STRIDE, MANIFOLD_STRIDE};
use crate::regions::{self, Buffer, Columns, MAX_WORLDS};

static mut COLUMNS: [Columns<1>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];
static mut CAPS: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut ALLOCATORS: [Vec<BlockAllocator>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
static mut MESH_CACHES: [Vec<Columns<1>>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
static mut NEXT_IDS: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut FREE_IDS: [Vec<usize>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
static LOCKS: [core::sync::atomic::AtomicU32; MAX_WORLDS] =
    [const { core::sync::atomic::AtomicU32::new(0) }; MAX_WORLDS];
fn lock(id: usize) {
    use core::sync::atomic::Ordering;
    while LOCKS[id]
        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}
fn unlock(id: usize) {
    LOCKS[id].store(0, core::sync::atomic::Ordering::Release);
}
#[export_name = "contactPairOrder"]
pub extern "C" fn contact_pair_order(a: usize, b: usize) -> u32 {
    // b3InitializeContactRegisters: primary rows for capsule, compound, height field, hull, mesh, sphere.
    const PRIMARY: [u8; 6] = [0b100001, 0b101001, 0b101001, 0b101001, 0b101001, 0b100000];
    if PRIMARY[a] & (1 << b) != 0 {
        1
    } else if PRIMARY[b] & (1 << a) != 0 {
        2
    } else {
        0
    }
}

#[export_name = "contactCapacity"]
pub extern "C" fn contact_capacity(world: usize) -> usize {
    assert!(world < MAX_WORLDS);
    unsafe { NEXT_IDS[world] }
}
#[export_name = "contactCount"]
pub extern "C" fn contact_count(world: usize) -> usize {
    assert!(world < MAX_WORLDS);
    unsafe { NEXT_IDS[world] - FREE_IDS[world].len() }
}
#[export_name = "allocContact"]
pub extern "C" fn alloc_contact() -> usize {
    alloc_contact_in_world(crate::regions::active())
}

pub extern "C" fn alloc_contact_in_world(world_index: usize) -> usize {
    use crate::manifold_abi::*;
    unsafe {
        let world = world_index;
        let id = if let Some(id) = FREE_IDS[world].pop() {
            id
        } else {
            let id = NEXT_IDS[world];
            NEXT_IDS[world] += 1;
            id
        };
        reserve_directory(world_index, (id + 1).next_power_of_two().max(16));
        let dir = dir_col(world_index);
        let o = id * DIR_STRIDE;
        let generation = dir.get(o + DIR_GENERATION).wrapping_add(1);
        for field in 0..DIR_STRIDE {
            dir.set(o + field, 0);
        }
        for field in [
            9,
            10,
            DIR_SET_INDEX,
            DIR_COLOR_INDEX,
            DIR_LOCAL_INDEX,
            DIR_EDGE_A,
            DIR_EDGE_A + 1,
            DIR_EDGE_A + 2,
            DIR_EDGE_B,
            DIR_EDGE_B + 1,
            DIR_EDGE_B + 2,
            DIR_SHAPE_A,
            DIR_SHAPE_B,
            DIR_ISLAND_ID,
            DIR_ISLAND_INDEX,
        ] {
            dir.set(o + field, u32::MAX);
        }
        dir.set(o + DIR_CONTACT_ID, id as u32);
        dir.set(o + DIR_GENERATION, generation);
        id
    }
}
#[export_name = "freeContact"]
pub extern "C" fn free_contact(contact: usize) {
    free_contact_in_world(crate::regions::active(), contact)
}

pub extern "C" fn free_contact_in_world(world_index: usize, contact: usize) {
    use crate::manifold_abi::*;
    unsafe {
        let world = world_index;
        let dir = dir_col(world_index);
        for field in [
            DIR_CONTACT_ID,
            DIR_SET_INDEX,
            DIR_COLOR_INDEX,
            DIR_LOCAL_INDEX,
        ] {
            dir.set(contact * DIR_STRIDE + field, u32::MAX);
        }
        FREE_IDS[world].push(contact);
    }
}

const BLOCK_SIZE: usize = 256;
struct BlockAllocator {
    chunks: Vec<Buffer>,
    element_size: usize,
    operations: u64,
    free: usize,
    next: usize,
}
impl BlockAllocator {
    unsafe fn new(count: usize) -> Self {
        let mut allocator = Self {
            chunks: Vec::new(),
            element_size: (count * MANIFOLD_STRIDE * 4 + 15) & !15,
            operations: 0,
            free: 0,
            next: 0,
        };
        // b3CreateBlockAllocator initializes two chunks for each size class.
        allocator.add_chunk();
        allocator.add_chunk();
        allocator
    }
    unsafe fn add_chunk(&mut self) {
        self.chunks
            .push(Buffer::allocate(BLOCK_SIZE * self.element_size));
    }
    unsafe fn allocate(&mut self) -> usize {
        self.operations += 1;
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
        self.operations += 1;
        *(address as *mut u32) = self.free as u32;
        self.free = address;
    }
    unsafe fn release(&mut self) {
        for chunk in &mut self.chunks {
            chunk.release();
        }
    }
}

// Read diagnostics only after the worker join; allocation/free counts live with their allocator.
#[export_name = "manifoldAllocatorOperations"]
pub extern "C" fn manifold_allocator_operations(world: usize) -> u64 {
    assert!(world < MAX_WORLDS);
    unsafe {
        ALLOCATORS[world]
            .iter()
            .map(|allocator| allocator.operations)
            .sum()
    }
}

pub fn has_mesh_caches(world_index: usize) -> bool {
    unsafe { !MESH_CACHES[world_index].is_empty() }
}

#[export_name = "meshCacheCapacity"]
pub extern "C" fn mesh_cache_capacity(world: usize) -> usize {
    assert!(world < MAX_WORLDS);
    unsafe { MESH_CACHES[world].capacity() }
}
#[export_name = "ensureMeshCache"]
pub extern "C" fn ensure_mesh_cache(contact: usize) {
    ensure_mesh_cache_in_world(crate::regions::active(), contact)
}

pub extern "C" fn ensure_mesh_cache_in_world(world_index: usize, contact: usize) {
    unsafe {
        let caches = &mut MESH_CACHES[world_index];
        caches.resize(caches.len().max(contact + 1), Columns::EMPTY);
        if caches[contact].layout[0] == 16 {
            caches[contact].reserve(0, core::mem::size_of::<crate::mesh_contact::MeshCache>());
            let cache = &mut *(caches[contact].layout[0] as *mut crate::mesh_contact::MeshCache);
            cache.lower = crate::math::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            cache.upper = crate::math::Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX);
        }
        dir_col(world_index).set(
            contact * DIR_STRIDE + crate::manifold_abi::DIR_MESH_CACHE,
            caches[contact].layout[0],
        );
    }
}
#[export_name = "freeMeshCache"]
pub extern "C" fn free_mesh_cache(contact: usize) {
    free_mesh_cache_in_world(crate::regions::active(), contact)
}

pub extern "C" fn free_mesh_cache_in_world(world_index: usize, contact: usize) {
    unsafe {
        if let Some(cache) = MESH_CACHES[world_index].get_mut(contact) {
            cache.release();
        }
        dir_col(world_index).set(
            contact * DIR_STRIDE + crate::manifold_abi::DIR_MESH_CACHE,
            0,
        );
    }
}
pub unsafe fn mesh_cache_ptr(
    world_index: usize,
    contact: usize,
) -> *mut crate::mesh_contact::MeshCache {
    dir_col(world_index).get(contact * DIR_STRIDE + crate::manifold_abi::DIR_MESH_CACHE)
        as *mut crate::mesh_contact::MeshCache
}

pub fn dir_col(world_index: usize) -> Col<'static, u32> {
    unsafe {
        let id = world_index;
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
    allocate_manifolds_in_world(crate::regions::active(), contact, count)
}

pub extern "C" fn allocate_manifolds_in_world(
    world_index: usize,
    contact: usize,
    count: usize,
) -> usize {
    let id = world_index;
    lock(id);
    let address = unsafe { allocate(id, contact, count) };
    unlock(id);
    if count > 0 {
        unsafe {
            (address as *mut u8).write_bytes(0, count * MANIFOLD_STRIDE * 4);
        }
    }
    address
}
#[export_name = "freeManifolds"]
pub extern "C" fn free_manifolds(contact: usize) {
    free_manifolds_in_world(crate::regions::active(), contact)
}

pub extern "C" fn free_manifolds_in_world(world_index: usize, contact: usize) {
    let id = world_index;
    lock(id);
    unsafe {
        free(id, contact);
    }
    unlock(id);
}
pub fn copy_manifolds(source: usize, address: usize, count: usize) {
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
    manifold_layout_ptr_in_world(crate::regions::active())
}

pub extern "C" fn manifold_layout_ptr_in_world(world_index: usize) -> *const u32 {
    unsafe { COLUMNS[world_index].layout.as_ptr() }
}
#[export_name = "contactRecordCapacity"]
pub extern "C" fn contact_record_capacity(world: usize) -> usize {
    assert!(world < MAX_WORLDS);
    unsafe { CAPS[world] }
}
fn reserve_directory(world_index: usize, contact_cap: usize) {
    unsafe {
        let id = world_index;
        if contact_cap > CAPS[id] {
            COLUMNS[id].reserve(0, contact_cap * DIR_STRIDE * 4);
            CAPS[id] = contact_cap;
        }
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
    NEXT_IDS[id] = 0;
    FREE_IDS[id] = Vec::new();
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, NEXT_IDS[id]);
    regions::write_word(out, FREE_IDS[id].len());
    for &value in &FREE_IDS[id] {
        regions::write_word(out, value);
    }
    regions::write_word(out, CAPS[id]);
    let dir =
        core::slice::from_raw_parts(COLUMNS[id].layout[0] as *const u32, CAPS[id] * DIR_STRIDE);
    // world_snapshot.c serializes contents, never allocator positions or free chunks.
    for (index, &word) in dir.iter().enumerate() {
        regions::write_word(
            out,
            if index % DIR_STRIDE == DIR_MANIFOLD_BASE
                || index % DIR_STRIDE == crate::manifold_abi::DIR_MESH_CACHE
            {
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
    NEXT_IDS[id] = regions::read_word(input);
    let free_count = regions::read_word(input);
    for _ in 0..free_count {
        FREE_IDS[id].push(regions::read_word(input));
    }
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
    for (contact, cache) in MESH_CACHES[id].iter_mut().enumerate() {
        cache.restore(input);
        if contact < CAPS[id] {
            *dir.add(contact * DIR_STRIDE + crate::manifold_abi::DIR_MESH_CACHE) =
                if cache.layout[0] == 16 {
                    0
                } else {
                    cache.layout[0]
                };
        }
    }
}
