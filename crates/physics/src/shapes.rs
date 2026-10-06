//! World-local shape and material columns. A reachable shape is authored before it is queried.
use crate::col::Col;
use crate::regions::{self, Columns, MAX_WORLDS};
pub const SHAPE_STRIDE: usize = 69;
pub const S_ID: usize = 61;
pub const S_PREV: usize = 62;
pub const S_DENSITY: usize = 63;
pub const S_GENERATION: usize = 68;
pub const S_FLAGS: usize = S_GENERATION;
pub const SENSOR_FLAG: u32 = 1 << 16;
pub const HIT_FLAG: u32 = 8 << 16;
pub const ENLARGED_FLAG: u32 = 32 << 16;
pub const SPECULATIVE_FLAG: u32 = 64 << 16;
pub const S_MATERIAL: usize = 52;
pub const S_HIT_EVENTS: usize = S_FLAGS;
pub const S_PROXY_KEY: usize = 50;
pub const S_QUERY_POSE: usize = 18;
pub const S_QUERY_CATEGORY: usize = 25;
pub const S_QUERY_MASK: usize = 27;
pub const S_QUERY_BODY: usize = 29;
pub const S_QUERY_SENSOR: usize = S_FLAGS;
pub const S_QUERY_GROUP: usize = 31;
pub const S_TYPE: usize = 0;
pub const S_NEXT: usize = 1;
pub const S_GEOM: usize = 2;
pub const S_GEO_REFERENCE: usize = 8;
pub const S_MATERIAL_HEAD: usize = 16;
pub const S_MATERIAL_COUNT: usize = 17;
pub const NULL_SHAPE: u32 = u32::MAX;
const FREE_ARRAY: usize = 1;
const N_SHAPE: usize = 2;
pub const MATERIAL_STRIDE: usize = 9;

#[derive(Clone, Copy)]
struct Pool {
    cap: usize,
    next: usize,
    free: usize,
}
impl Pool {
    const EMPTY: Self = Self {
        cap: 0,
        next: 0,
        free: 0,
    };
}
#[derive(Clone, Copy)]
struct Shapes {
    columns: Columns<N_SHAPE>,
    shape: Pool,
}
impl Shapes {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        shape: Pool::EMPTY,
    };
}
static mut WORLDS: [Shapes; MAX_WORLDS] = [Shapes::EMPTY; MAX_WORLDS];
unsafe fn world(id: usize) -> &'static Shapes {
    &WORLDS[id]
}
unsafe fn world_mut(id: usize) -> &'static mut Shapes {
    &mut WORLDS[id]
}
fn base() -> usize {
    unsafe { world(regions::active()).columns.layout[0] as usize }
}
pub fn col() -> Col<'static, u32> {
    unsafe { Col::new(base() as *mut u32, shape_cap() * SHAPE_STRIDE) }
}
pub fn col_f() -> Col<'static, f32> {
    unsafe { Col::new(base() as *mut f32, shape_cap() * SHAPE_STRIDE) }
}
pub fn col_slice() -> &'static [u32] {
    unsafe { core::slice::from_raw_parts(base() as *const u32, shape_cap() * SHAPE_STRIDE) }
}
#[export_name = "shapeSetActiveWorld"]
pub extern "C" fn shape_set_active_world(id: u32) {
    regions::select(id);
}
#[export_name = "shapeLayoutPtr"]
pub extern "C" fn shape_layout_ptr() -> *const u32 {
    unsafe { world(regions::active()).columns.layout.as_ptr() }
}
#[export_name = "shapeCap"]
pub extern "C" fn shape_cap() -> usize {
    unsafe { world(regions::active()).shape.cap }
}
#[export_name = "reserveShapes"]
pub extern "C" fn reserve_shapes(cap: usize) -> u32 {
    unsafe {
        let w = world_mut(regions::active());
        if cap <= w.shape.cap {
            return 0;
        }
        w.columns.reserve(0, cap * SHAPE_STRIDE * 4);
        w.columns.reserve(FREE_ARRAY, cap.max(32) * 4);
        for id in w.shape.cap..cap {
            let p = record(regions::active(), id);
            p.write_bytes(0, SHAPE_STRIDE);
            *p.add(S_ID) = NULL_SHAPE;
        }
        crate::fataabb::reserve_fat_aabb(cap);
        w.shape.cap = cap;
        1
    }
}
unsafe fn record(id: usize, shape: usize) -> *mut u32 {
    (world(id).columns.layout[0] as *mut u32).add(shape * SHAPE_STRIDE)
}
unsafe fn free_materials(id: usize, shape: usize) {
    let p = record(id, shape);
    let count = *p.add(S_MATERIAL_COUNT) as usize;
    if *p.add(S_MATERIAL_HEAD) != 0 {
        std::alloc::dealloc(
            *p.add(S_MATERIAL_HEAD) as *mut u8,
            std::alloc::Layout::from_size_align_unchecked(count * MATERIAL_STRIDE * 4, 4),
        );
    }
    *p.add(S_MATERIAL_HEAD) = 0;
    *p.add(S_MATERIAL_COUNT) = 0;
}
/// Compounds own their material array; other shapes keep a single material inline.
#[export_name = "shapeAllocateMaterials"]
pub unsafe extern "C" fn allocate_materials(id: u32, shape: u32, count: usize) -> usize {
    free_materials(id as usize, shape as usize);
    let compound = *record(id as usize, shape as usize).add(S_TYPE) == crate::finalize::TY_COMPOUND;
    let ptr = if count > 1 || (count > 0 && compound) {
        let layout = std::alloc::Layout::from_size_align_unchecked(count * MATERIAL_STRIDE * 4, 4);
        let p = std::alloc::alloc(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as usize
    } else {
        0
    };
    let p = record(id as usize, shape as usize);
    *p.add(S_MATERIAL_HEAD) = ptr as u32;
    *p.add(S_MATERIAL_COUNT) = count as u32;
    material_ptr(id as usize, shape as usize) as usize
}
#[export_name = "shapeFreeMaterials"]
pub unsafe extern "C" fn release_materials(id: u32, shape: u32) {
    free_materials(id as usize, shape as usize);
}
unsafe fn material_ptr(id: usize, shape: usize) -> *mut u32 {
    let p = record(id, shape);
    if *p.add(S_MATERIAL_HEAD) != 0 {
        *p.add(S_MATERIAL_HEAD) as *mut u32
    } else {
        p.add(S_MATERIAL)
    }
}
#[export_name = "shapeMaterialPtr"]
pub unsafe extern "C" fn shape_material_ptr(id: u32, shape: u32) -> usize {
    material_ptr(id as usize, shape as usize) as usize
}
pub(crate) fn material(shape: usize, index: usize) -> &'static [u32] {
    unsafe {
        let id = regions::active();
        let count = *record(id, shape).add(S_MATERIAL_COUNT) as usize;
        assert!(count > 0);
        core::slice::from_raw_parts(
            material_ptr(id, shape).add(index.min(count - 1) * MATERIAL_STRIDE),
            MATERIAL_STRIDE,
        )
    }
}
unsafe fn shape_lane(id: u32, shape: u32, lane: usize) -> u32 {
    let w = world(id as usize);
    if shape as usize >= w.shape.cap {
        return if lane == S_MATERIAL_HEAD { u32::MAX } else { 0 };
    }
    *((w.columns.layout[0] as *const u32).add(shape as usize * SHAPE_STRIDE + lane))
}
#[export_name = "shapeMaterialSet"]
pub unsafe extern "C" fn material_set(
    id: usize,
    shape: usize,
    index: usize,
    friction: f32,
    restitution: f32,
    rolling: f32,
    x: f32,
    y: f32,
    z: f32,
    low: u32,
    high: u32,
    color: u32,
) {
    let p = material_ptr(id, shape).add(index * MATERIAL_STRIDE);
    let f = p as *mut f32;
    *f = friction;
    *f.add(1) = restitution;
    *f.add(2) = rolling;
    *f.add(3) = x;
    *f.add(4) = y;
    *f.add(5) = z;
    *p.add(6) = low;
    *p.add(7) = high;
    *p.add(8) = color;
}
#[export_name = "shapeMaterialCount"]
pub extern "C" fn shape_material_count(id: u32, shape: u32) -> u32 {
    unsafe { shape_lane(id, shape, S_MATERIAL_COUNT) }
}
#[export_name = "shapeCreate"]
pub extern "C" fn shape_create(
    id: u32,
    body: u32,
    kind: u32,
    density: f32,
    explosion: f32,
    flags: u32,
) -> u32 {
    regions::select(id);
    unsafe {
        let p = world(id as usize).shape;
        if p.free == 0 && p.next == p.cap {
            reserve_shapes((p.cap * 2).max(16));
        }
        let w = world_mut(id as usize);
        let shape = if w.shape.free > 0 {
            w.shape.free -= 1;
            *(w.columns.layout[FREE_ARRAY] as *const u32).add(w.shape.free) as usize
        } else {
            let shape = w.shape.next;
            w.shape.next += 1;
            shape
        };
        let p = record(id as usize, shape);
        let generation = (*p.add(S_GENERATION) as u16).wrapping_add(1);
        p.write_bytes(0, SHAPE_STRIDE);
        *p.add(S_GENERATION) = generation as u32;
        *p.add(S_ID) = shape as u32;
        *p.add(S_TYPE) = kind;
        *p.add(S_QUERY_BODY) = body;
        *((p.add(S_FLAGS) as *mut u8).add(2)) = flags as u8;
        *(p.add(S_DENSITY) as *mut f32) = density;
        *(p.add(64) as *mut f32) = explosion;
        for lane in [S_NEXT, S_PREV, S_PROXY_KEY, 41] {
            *p.add(lane) = NULL_SHAPE;
        }
        let fat = crate::fataabb::col();
        for lane in 0..6 {
            fat.set(shape * 6 + lane, 0.0);
        }
        shape as u32
    }
}
#[export_name = "shapeDestroy"]
pub extern "C" fn shape_destroy(id: u32, shape: u32) {
    unsafe {
        let w = world_mut(id as usize);
        if shape as usize >= w.shape.next {
            return;
        }
        let p = record(id as usize, shape as usize);
        if *p.add(S_ID) == NULL_SHAPE {
            return;
        }
        free_materials(id as usize, shape as usize);
        *p.add(S_ID) = NULL_SHAPE;
        *(w.columns.layout[FREE_ARRAY] as *mut u32).add(w.shape.free) = shape;
        w.shape.free += 1;
    }
}
#[export_name = "shapeResetWorld"]
pub extern "C" fn shape_reset_world(id: u32) {
    unsafe {
        let w = world_mut(id as usize);
        for shape in 0..w.shape.next {
            free_materials(id as usize, shape);
        }
        w.columns.release();
        *w = Shapes::EMPTY;
    }
}
#[export_name = "shapeGeneration"]
pub extern "C" fn shape_generation(id: u32, shape: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if shape as usize >= w.shape.cap {
            return 0;
        }
        *(record(id as usize, shape as usize).add(S_GENERATION) as *const u16) as u32
    }
}
#[export_name = "shapeAlive"]
pub extern "C" fn shape_alive(id: u32, shape: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if shape as usize >= w.shape.cap {
            return 0;
        }
        (*record(id as usize, shape as usize).add(S_ID) != NULL_SHAPE) as u32
    }
}
#[export_name = "shapeCount"]
pub extern "C" fn shape_count(id: u32) -> usize {
    unsafe {
        let pool = world(id as usize).shape;
        pool.next - pool.free
    }
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    for pool in [w.shape] {
        for value in [pool.cap, pool.next, pool.free] {
            regions::write_word(out, value);
        }
    }
    w.columns.snapshot(out);
    for shape in 0..w.shape.next {
        let count = *record(id, shape).add(S_MATERIAL_COUNT) as usize;
        if *record(id, shape).add(S_MATERIAL_HEAD) != 0 {
            out.extend_from_slice(core::slice::from_raw_parts(
                material_ptr(id, shape) as *const u8,
                count * MATERIAL_STRIDE * 4,
            ));
        }
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    let w = &mut WORLDS[id];
    for shape in 0..w.shape.next {
        free_materials(id, shape);
    }
    for pool in [&mut w.shape] {
        pool.cap = regions::read_word(input);
        pool.next = regions::read_word(input);
        pool.free = regions::read_word(input);
    }
    w.columns.restore(input);
    for shape in 0..w.shape.next {
        let p = record(id, shape);
        let count = *p.add(S_MATERIAL_COUNT) as usize;
        if *p.add(S_MATERIAL_HEAD) != 0 {
            // The saved pointer is not owned by the restored world.
            *p.add(S_MATERIAL_HEAD) = 0;
            *p.add(S_MATERIAL_COUNT) = 0;
            let ptr = allocate_materials(id as u32, shape as u32, count);
            let bytes = count * MATERIAL_STRIDE * 4;
            let (data, rest) = input.split_at(bytes);
            core::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, bytes);
            *input = rest;
        } else {
            *p.add(S_MATERIAL_HEAD) = 0;
        }
    }
}
