//! Joint-sim arrays owned by each world, in graph-color and solver-set order.
use crate::col::Col;
use crate::joint_abi::{JOINT_STRIDE, J_JOINT_ID, NULL_INDEX};
use crate::regions::{self, Columns, MAX_WORLDS};

#[derive(Clone, Copy)]
pub(crate) struct JointArray {
    pub(crate) records: Columns<1>,
    pub(crate) count: usize,
}
impl JointArray {
    pub(crate) const EMPTY: Self = Self {
        records: Columns::EMPTY,
        count: 0,
    };
    unsafe fn ptr(&self, index: usize) -> *mut u32 {
        (self.records.layout[0] as *mut u32).add(index * JOINT_STRIDE)
    }
    unsafe fn append(&mut self) -> usize {
        let index = self.count;
        self.records.reserve(0, (index + 1) * JOINT_STRIDE * 4);
        self.ptr(index).write_bytes(0, JOINT_STRIDE);
        self.count += 1;
        index
    }
    unsafe fn remove(&mut self, index: usize) -> u32 {
        assert!(index < self.count);
        self.count -= 1;
        if index == self.count {
            return NULL_INDEX;
        }
        core::ptr::copy_nonoverlapping(self.ptr(self.count), self.ptr(index), JOINT_STRIDE);
        *self.ptr(index).add(J_JOINT_ID)
    }
}
static mut ARRAYS: [[JointArray; crate::constraint_graph::COLORS]; MAX_WORLDS] =
    [[JointArray::EMPTY; crate::constraint_graph::COLORS]; MAX_WORLDS];
unsafe fn array(key: usize) -> &'static mut JointArray {
    if key >= crate::constraint_graph::COLORS {
        return crate::solver_set::joint_array(key - crate::constraint_graph::COLORS);
    }
    let arrays = &mut ARRAYS[regions::active()];
    &mut arrays[key]
}
#[export_name = "jointArrayRelease"]
pub extern "C" fn release(key: usize) {
    unsafe {
        let a = array(key);
        a.records.release();
        a.count = 0;
    }
}
#[export_name = "jointArrayCount"]
pub extern "C" fn count(key: usize) -> usize {
    unsafe { array(key).count }
}
#[export_name = "jointArrayPtr"]
pub extern "C" fn pointer(key: usize) -> usize {
    unsafe { array(key).records.layout[0] as usize }
}
#[export_name = "jointArrayAppend"]
pub extern "C" fn append(key: usize) -> usize {
    unsafe { array(key).append() }
}
#[export_name = "jointArrayRemove"]
pub extern "C" fn remove(key: usize, index: usize) -> u32 {
    unsafe { array(key).remove(index) }
}
#[export_name = "jointArrayMove"]
pub extern "C" fn move_record(source: usize, index: usize, target: usize) -> u32 {
    assert_ne!(source, target);
    unsafe {
        assert!(index < array(source).count);
        let destination = array(target).append();
        core::ptr::copy_nonoverlapping(
            array(source).ptr(index),
            array(target).ptr(destination),
            JOINT_STRIDE,
        );
        array(source).remove(index)
    }
}
#[export_name = "jointReadFloat"]
pub extern "C" fn read_float(key: usize, index: usize, field: usize) -> f32 {
    unsafe {
        assert!(index < array(key).count && field < JOINT_STRIDE);
        f32::from_bits(*array(key).ptr(index).add(field))
    }
}
#[export_name = "jointWriteFloat"]
pub extern "C" fn write_float(key: usize, index: usize, field: usize, value: f32) {
    unsafe {
        assert!(index < array(key).count && field < JOINT_STRIDE);
        *array(key).ptr(index).add(field) = value.to_bits();
    }
}
#[export_name = "jointReadWord"]
pub extern "C" fn read_word(key: usize, index: usize, field: usize) -> u32 {
    unsafe {
        assert!(index < array(key).count && field < JOINT_STRIDE);
        *array(key).ptr(index).add(field)
    }
}
#[export_name = "jointWriteWord"]
pub extern "C" fn write_word(key: usize, index: usize, field: usize, value: u32) {
    unsafe {
        assert!(index < array(key).count && field < JOINT_STRIDE);
        *array(key).ptr(index).add(field) = value;
    }
}
pub unsafe fn column(key: usize) -> Col<'static, f32> {
    let a = array(key);
    Col::new(a.ptr(0) as *mut f32, a.count * JOINT_STRIDE)
}
pub unsafe fn reset(id: usize) {
    for a in &mut ARRAYS[id] {
        a.records.release();
    }
    ARRAYS[id] = [JointArray::EMPTY; crate::constraint_graph::COLORS];
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, ARRAYS[id].len());
    for a in &ARRAYS[id] {
        regions::write_word(out, a.count);
        a.records.snapshot(out);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let count = regions::read_word(input);
    assert_eq!(count, crate::constraint_graph::COLORS);
    for a in &mut ARRAYS[id] {
        a.count = regions::read_word(input);
        a.records.restore(input);
    }
}
