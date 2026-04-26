use core::arch::wasm32::*;
use wasm_bindgen::prelude::*;

const INITIAL_CAPACITY: usize = 1024;
const NO_PARENT: u32 = u32::MAX;

use core::cell::UnsafeCell;

struct DataCell(UnsafeCell<Option<TransformData>>);
unsafe impl Sync for DataCell {}
static DATA: DataCell = DataCell(UnsafeCell::new(None));

struct TransformData {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    quat_x: Vec<f32>,
    quat_y: Vec<f32>,
    quat_z: Vec<f32>,
    quat_w: Vec<f32>,
    scale_x: Vec<f32>,
    scale_y: Vec<f32>,
    scale_z: Vec<f32>,
    matrices: Vec<f32>,
    indices: Vec<u32>,
    parents: Vec<u32>,
    capacity: usize,
}

impl TransformData {
    fn new(cap: usize) -> Self {
        let mut data = Self {
            pos_x: vec![0.0; cap],
            pos_y: vec![0.0; cap],
            pos_z: vec![0.0; cap],
            quat_x: vec![0.0; cap],
            quat_y: vec![0.0; cap],
            quat_z: vec![0.0; cap],
            quat_w: vec![1.0; cap],
            scale_x: vec![1.0; cap],
            scale_y: vec![1.0; cap],
            scale_z: vec![1.0; cap],
            matrices: vec![0.0; cap * 16],
            indices: vec![0; cap],
            parents: vec![NO_PARENT; cap],
            capacity: cap,
        };

        for i in 0..cap {
            let o = i * 16;
            data.matrices[o] = 1.0;
            data.matrices[o + 5] = 1.0;
            data.matrices[o + 10] = 1.0;
            data.matrices[o + 15] = 1.0;
        }

        data
    }

    fn grow(&mut self, new_cap: usize) {
        let old = self.capacity;
        if new_cap <= old {
            return;
        }

        self.pos_x.resize(new_cap, 0.0);
        self.pos_y.resize(new_cap, 0.0);
        self.pos_z.resize(new_cap, 0.0);
        self.quat_x.resize(new_cap, 0.0);
        self.quat_y.resize(new_cap, 0.0);
        self.quat_z.resize(new_cap, 0.0);
        self.quat_w.resize(new_cap, 1.0);
        self.scale_x.resize(new_cap, 1.0);
        self.scale_y.resize(new_cap, 1.0);
        self.scale_z.resize(new_cap, 1.0);
        self.indices.resize(new_cap, 0);
        self.parents.resize(new_cap, NO_PARENT);

        self.matrices.resize(new_cap * 16, 0.0);
        for i in old..new_cap {
            let o = i * 16;
            self.matrices[o] = 1.0;
            self.matrices[o + 5] = 1.0;
            self.matrices[o + 10] = 1.0;
            self.matrices[o + 15] = 1.0;
        }

        self.capacity = new_cap;
    }
}

#[inline(always)]
fn data() -> &'static mut TransformData {
    unsafe { (*DATA.0.get()).as_mut().unwrap_unchecked() }
}

#[wasm_bindgen]
pub fn init_data() {
    unsafe {
        let slot = &mut *DATA.0.get();
        if slot.is_none() {
            *slot = Some(TransformData::new(INITIAL_CAPACITY));
        }
    }
}

#[wasm_bindgen]
pub fn ensure_capacity(n: usize) {
    let d = data();
    if n <= d.capacity {
        return;
    }
    let mut next = d.capacity;
    while next < n {
        next *= 2;
    }
    d.grow(next);
}

#[wasm_bindgen]
pub fn get_capacity() -> usize {
    data().capacity
}

#[wasm_bindgen]
pub fn get_pos_x_ptr() -> *const f32 {
    data().pos_x.as_ptr()
}
#[wasm_bindgen]
pub fn get_pos_y_ptr() -> *const f32 {
    data().pos_y.as_ptr()
}
#[wasm_bindgen]
pub fn get_pos_z_ptr() -> *const f32 {
    data().pos_z.as_ptr()
}
#[wasm_bindgen]
pub fn get_quat_x_ptr() -> *const f32 {
    data().quat_x.as_ptr()
}
#[wasm_bindgen]
pub fn get_quat_y_ptr() -> *const f32 {
    data().quat_y.as_ptr()
}
#[wasm_bindgen]
pub fn get_quat_z_ptr() -> *const f32 {
    data().quat_z.as_ptr()
}
#[wasm_bindgen]
pub fn get_quat_w_ptr() -> *const f32 {
    data().quat_w.as_ptr()
}
#[wasm_bindgen]
pub fn get_scale_x_ptr() -> *const f32 {
    data().scale_x.as_ptr()
}
#[wasm_bindgen]
pub fn get_scale_y_ptr() -> *const f32 {
    data().scale_y.as_ptr()
}
#[wasm_bindgen]
pub fn get_scale_z_ptr() -> *const f32 {
    data().scale_z.as_ptr()
}
#[wasm_bindgen]
pub fn get_matrices_ptr() -> *const f32 {
    data().matrices.as_ptr()
}
#[wasm_bindgen]
pub fn get_indices_ptr() -> *const u32 {
    data().indices.as_ptr()
}
#[wasm_bindgen]
pub fn get_parents_ptr() -> *const u32 {
    data().parents.as_ptr()
}
#[wasm_bindgen]
pub fn get_max_entities() -> usize {
    data().capacity
}
#[wasm_bindgen]
pub fn get_no_parent() -> u32 {
    NO_PARENT
}

#[inline(always)]
fn quat_to_rotation(qx: f32, qy: f32, qz: f32, qw: f32) -> [f32; 9] {
    let x2 = qx + qx;
    let y2 = qy + qy;
    let z2 = qz + qz;
    let xx = qx * x2;
    let xy = qx * y2;
    let xz = qx * z2;
    let yy = qy * y2;
    let yz = qy * z2;
    let zz = qz * z2;
    let wx = qw * x2;
    let wy = qw * y2;
    let wz = qw * z2;

    [
        1.0 - (yy + zz),
        xy + wz,
        xz - wy,
        xy - wz,
        1.0 - (xx + zz),
        yz + wx,
        xz + wy,
        yz - wx,
        1.0 - (xx + yy),
    ]
}

#[inline(always)]
fn compute_matrix(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    sx: f32,
    sy: f32,
    sz: f32,
    px: f32,
    py: f32,
    pz: f32,
    out: &mut [f32; 16],
) {
    let r = quat_to_rotation(qx, qy, qz, qw);
    unsafe {
        let col0 = f32x4(r[0] * sx, r[1] * sx, r[2] * sx, 0.0);
        let col1 = f32x4(r[3] * sy, r[4] * sy, r[5] * sy, 0.0);
        let col2 = f32x4(r[6] * sz, r[7] * sz, r[8] * sz, 0.0);
        let col3 = f32x4(px, py, pz, 1.0);
        v128_store(out.as_mut_ptr() as *mut v128, col0);
        v128_store(out.as_mut_ptr().add(4) as *mut v128, col1);
        v128_store(out.as_mut_ptr().add(8) as *mut v128, col2);
        v128_store(out.as_mut_ptr().add(12) as *mut v128, col3);
    }
}

#[inline(always)]
fn mat4_multiply(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    unsafe {
        let a0 = v128_load(a.as_ptr() as *const v128);
        let a1 = v128_load(a.as_ptr().add(4) as *const v128);
        let a2 = v128_load(a.as_ptr().add(8) as *const v128);
        let a3 = v128_load(a.as_ptr().add(12) as *const v128);

        for col in 0..4 {
            let base = col * 4;
            let b0 = f32x4_splat(b[base]);
            let b1 = f32x4_splat(b[base + 1]);
            let b2 = f32x4_splat(b[base + 2]);
            let b3 = f32x4_splat(b[base + 3]);

            let r = f32x4_add(
                f32x4_add(f32x4_mul(a0, b0), f32x4_mul(a1, b1)),
                f32x4_add(f32x4_mul(a2, b2), f32x4_mul(a3, b3)),
            );
            v128_store(out.as_mut_ptr().add(base) as *mut v128, r);
        }
    }
}

#[inline(always)]
fn copy16(dst: &mut [f32], dst_offset: usize, src: &[f32; 16]) {
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(dst_offset);
        v128_store(dst_ptr as *mut v128, v128_load(src.as_ptr() as *const v128));
        v128_store(
            dst_ptr.add(4) as *mut v128,
            v128_load(src.as_ptr().add(4) as *const v128),
        );
        v128_store(
            dst_ptr.add(8) as *mut v128,
            v128_load(src.as_ptr().add(8) as *const v128),
        );
        v128_store(
            dst_ptr.add(12) as *mut v128,
            v128_load(src.as_ptr().add(12) as *const v128),
        );
    }
}

#[inline(always)]
fn load16(src: &[f32], src_offset: usize, dst: &mut [f32; 16]) {
    unsafe {
        let src_ptr = src.as_ptr().add(src_offset);
        v128_store(
            dst.as_mut_ptr() as *mut v128,
            v128_load(src_ptr as *const v128),
        );
        v128_store(
            dst.as_mut_ptr().add(4) as *mut v128,
            v128_load(src_ptr.add(4) as *const v128),
        );
        v128_store(
            dst.as_mut_ptr().add(8) as *mut v128,
            v128_load(src_ptr.add(8) as *const v128),
        );
        v128_store(
            dst.as_mut_ptr().add(12) as *mut v128,
            v128_load(src_ptr.add(12) as *const v128),
        );
    }
}

#[wasm_bindgen]
pub fn compute_transforms(count: usize) {
    let data = data();
    let mut local = [0.0f32; 16];
    let mut parent_mat = [0.0f32; 16];
    let mut result = [0.0f32; 16];

    for i in 0..count {
        let eid = data.indices[i] as usize;
        let parent = data.parents[i];

        compute_matrix(
            data.quat_x[eid],
            data.quat_y[eid],
            data.quat_z[eid],
            data.quat_w[eid],
            data.scale_x[eid],
            data.scale_y[eid],
            data.scale_z[eid],
            data.pos_x[eid],
            data.pos_y[eid],
            data.pos_z[eid],
            &mut local,
        );

        let o = eid * 16;
        if parent == NO_PARENT {
            copy16(&mut data.matrices, o, &local);
        } else {
            let po = (parent as usize) * 16;
            load16(&data.matrices, po, &mut parent_mat);
            mat4_multiply(&parent_mat, &local, &mut result);
            copy16(&mut data.matrices, o, &result);
        }
    }
}
