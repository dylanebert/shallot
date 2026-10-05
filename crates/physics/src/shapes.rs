//! World-local shape and material columns. A reachable shape is authored before it is queried.
use crate::col::Col;
use crate::regions::{self, Columns, MAX_WORLDS};
pub const SHAPE_STRIDE: usize = 51;
pub const S_PROXY_KEY: usize = 50;
pub const S_QUERY_POSE: usize = 18;
pub const S_QUERY_CATEGORY: usize = 25;
pub const S_QUERY_MASK: usize = 27;
pub const S_QUERY_BODY: usize = 29;
pub const S_QUERY_SENSOR: usize = 30;
pub const S_QUERY_GROUP: usize = 31;
pub const S_TYPE: usize = 0;
pub const S_NEXT: usize = 1;
pub const S_GEOM: usize = 2;
pub const S_GEO_REFERENCE: usize = 8;
pub const S_ESCAPED: usize = 15;
pub const S_MATERIAL_HEAD: usize = 16;
pub const S_MATERIAL_COUNT: usize = 17;
pub const NULL_SHAPE: u32 = u32::MAX;
const GENERATION: usize = 1;
const ALIVE: usize = 2;
const NEXT: usize = 3;
const MATERIAL: usize = 4;
const N_SHAPE: usize = 5;
pub const MATERIAL_STRIDE: usize = 12;
const M_NEXT: usize = 9;
const M_GENERATION: usize = 10;
const M_ALIVE: usize = 11;

#[derive(Clone, Copy)]
struct Pool {
    cap: usize,
    next: usize,
    free: i32,
    count: usize,
}
impl Pool {
    const EMPTY: Self = Self {
        cap: 0,
        next: 0,
        free: -1,
        count: 0,
    };
}
#[derive(Clone, Copy)]
struct Shapes {
    columns: Columns<N_SHAPE>,
    shape: Pool,
    material: Pool,
}
impl Shapes {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        shape: Pool::EMPTY,
        material: Pool::EMPTY,
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
        for c in [GENERATION, ALIVE, NEXT] {
            w.columns.reserve(c, cap * 4);
        }
        for id in w.shape.cap..cap {
            *(w.columns.layout[GENERATION] as *mut u32).add(id) = 0;
            *(w.columns.layout[ALIVE] as *mut u32).add(id) = 0;
            *(w.columns.layout[NEXT] as *mut u32).add(id) = u32::MAX;
        }
        w.shape.cap = cap;
        1
    }
}
#[export_name = "reserveMaterials"]
pub extern "C" fn reserve_materials(cap: usize) -> u32 {
    unsafe {
        let w = world_mut(regions::active());
        if cap <= w.material.cap {
            return 0;
        }
        w.columns.reserve(MATERIAL, cap * MATERIAL_STRIDE * 4);
        w.material.cap = cap;
        1
    }
}
#[export_name = "materialLayoutPtr"]
pub extern "C" fn material_layout_ptr() -> *const u32 {
    unsafe { &raw const world(regions::active()).columns.layout[MATERIAL] }
}
#[export_name = "materialCap"]
pub extern "C" fn material_cap() -> usize {
    unsafe { world(regions::active()).material.cap }
}
pub(crate) fn materials() -> Col<'static, u32> {
    unsafe {
        let w = world(regions::active());
        Col::new(
            w.columns.layout[MATERIAL] as *mut u32,
            w.material.cap * MATERIAL_STRIDE,
        )
    }
}

unsafe fn material_ptr(id: usize, material: usize) -> *mut u32 {
    (world(id).columns.layout[MATERIAL] as *mut u32).add(material * MATERIAL_STRIDE)
}
#[export_name = "materialCreate"]
pub extern "C" fn material_create(id: u32) -> u32 {
    regions::select(id);
    unsafe {
        let p = world(id as usize).material;
        if p.free < 0 && p.next == p.cap {
            reserve_materials((p.cap * 2).max(16));
        }
        let w = world_mut(id as usize);
        let material = if w.material.free >= 0 {
            let material = w.material.free as usize;
            w.material.free = *((w.columns.layout[MATERIAL] as *const u32)
                .add(material * MATERIAL_STRIDE + M_NEXT)) as i32;
            material
        } else {
            let material = w.material.next;
            w.material.next += 1;
            material
        };
        let p = (w.columns.layout[MATERIAL] as *mut u32).add(material * MATERIAL_STRIDE);
        *p.add(M_GENERATION) = (*p.add(M_GENERATION)).wrapping_add(1);
        *p.add(M_ALIVE) = 1;
        *p.add(M_NEXT) = u32::MAX;
        w.material.count += 1;
        material as u32
    }
}
#[export_name = "materialDestroy"]
pub extern "C" fn material_destroy(id: u32, material: u32) {
    unsafe {
        let w = world_mut(id as usize);
        if material as usize >= w.material.next {
            return;
        }
        let p = (w.columns.layout[MATERIAL] as *mut u32).add(material as usize * MATERIAL_STRIDE);
        if *p.add(M_ALIVE) == 0 {
            return;
        }
        *p.add(M_ALIVE) = 0;
        *p.add(M_NEXT) = w.material.free as u32;
        w.material.free = material as i32;
        w.material.count -= 1;
    }
}
#[export_name = "materialResetWorld"]
pub extern "C" fn material_reset_world(id: u32) {
    unsafe {
        let w = world_mut(id as usize);
        w.material = Pool::EMPTY;
    }
}
#[export_name = "materialGeneration"]
pub extern "C" fn material_generation(id: u32, material: u32) -> u32 {
    unsafe {
        if material as usize >= world(id as usize).material.cap {
            return 0;
        }
        *material_ptr(id as usize, material as usize).add(M_GENERATION)
    }
}
#[export_name = "materialAlive"]
pub extern "C" fn material_alive(id: u32, material: u32) -> u32 {
    unsafe {
        if material as usize >= world(id as usize).material.cap {
            return 0;
        }
        *material_ptr(id as usize, material as usize).add(M_ALIVE)
    }
}
#[export_name = "materialListCount"]
pub extern "C" fn material_list_count(id: u32, head: u32) -> u32 {
    unsafe {
        let cap = world(id as usize).material.cap;
        let mut material = head;
        let mut count = 0;
        while material != u32::MAX && (material as usize) < cap && count <= cap as u32 {
            let p = material_ptr(id as usize, material as usize);
            if *p.add(M_ALIVE) == 0 {
                break;
            }
            count += 1;
            material = *p.add(M_NEXT);
        }
        count
    }
}
unsafe fn shape_lane(id: u32, shape: u32, lane: usize) -> u32 {
    let w = world(id as usize);
    if shape as usize >= w.shape.cap {
        return if lane == S_MATERIAL_HEAD { u32::MAX } else { 0 };
    }
    *((w.columns.layout[0] as *const u32).add(shape as usize * SHAPE_STRIDE + lane))
}
#[export_name = "shapeMaterialHead"]
pub extern "C" fn shape_material_head(id: u32, shape: u32) -> u32 {
    unsafe { shape_lane(id, shape, S_MATERIAL_HEAD) }
}
#[export_name = "shapeMaterialCount"]
pub extern "C" fn shape_material_count(id: u32, shape: u32) -> u32 {
    unsafe { shape_lane(id, shape, S_MATERIAL_COUNT) }
}
#[export_name = "shapeCreate"]
pub extern "C" fn shape_create(id: u32) -> u32 {
    regions::select(id);
    unsafe {
        let p = world(id as usize).shape;
        if p.free < 0 && p.next == p.cap {
            reserve_shapes((p.cap * 2).max(16));
        }
        let w = world_mut(id as usize);
        let shape = if w.shape.free >= 0 {
            let shape = w.shape.free as usize;
            w.shape.free = *(w.columns.layout[NEXT] as *const u32).add(shape) as i32;
            shape
        } else {
            let shape = w.shape.next;
            w.shape.next += 1;
            shape
        };
        let generation = (w.columns.layout[GENERATION] as *mut u32).add(shape);
        *generation = (*generation).wrapping_add(1);
        *(w.columns.layout[ALIVE] as *mut u32).add(shape) = 1;
        *(w.columns.layout[NEXT] as *mut u32).add(shape) = u32::MAX;
        w.shape.count += 1;
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
        let alive = (w.columns.layout[ALIVE] as *mut u32).add(shape as usize);
        if *alive == 0 {
            return;
        }
        *alive = 0;
        *(w.columns.layout[NEXT] as *mut u32).add(shape as usize) = w.shape.free as u32;
        w.shape.free = shape as i32;
        w.shape.count -= 1;
    }
}
#[export_name = "shapeResetWorld"]
pub extern "C" fn shape_reset_world(id: u32) {
    unsafe {
        let w = world_mut(id as usize);
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
        *(w.columns.layout[GENERATION] as *const u32).add(shape as usize)
    }
}
#[export_name = "shapeAlive"]
pub extern "C" fn shape_alive(id: u32, shape: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if shape as usize >= w.shape.cap {
            return 0;
        }
        *(w.columns.layout[ALIVE] as *const u32).add(shape as usize)
    }
}
#[export_name = "shapeCount"]
pub extern "C" fn shape_count(id: u32) -> usize {
    unsafe { world(id as usize).shape.count }
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    for pool in [w.shape, w.material] {
        for value in [pool.cap, pool.next, pool.free as usize, pool.count] {
            regions::write_word(out, value);
        }
    }
    w.columns.snapshot(out);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    let w = &mut WORLDS[id];
    for pool in [&mut w.shape, &mut w.material] {
        pool.cap = regions::read_word(input);
        pool.next = regions::read_word(input);
        pool.free = regions::read_word(input) as i32;
        pool.count = regions::read_word(input);
    }
    w.columns.restore(input);
}
