//! Joint-sim arrays owned by each world, in graph-color and solver-set order.
use crate::col::Col;
use crate::joint_abi::{JOINT_STRIDE, J_JOINT_ID, NULL_INDEX};
use crate::regions::{self, Columns};

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
unsafe fn array(world_index: usize, key: usize) -> &'static mut JointArray {
    if key >= crate::constraint_graph::COLORS {
        return crate::solver_set::joint_array(world_index, key - crate::constraint_graph::COLORS);
    }
    crate::constraint_graph::joint_array(world_index, key)
}
#[export_name = "jointArrayRelease"]
pub extern "C" fn release(key: usize) {
    release_in_world(crate::regions::active(), key)
}

pub extern "C" fn release_in_world(world_index: usize, key: usize) {
    unsafe {
        let a = array(world_index, key);
        a.records.release();
        a.count = 0;
    }
}
#[export_name = "jointArrayCount"]
pub extern "C" fn count(key: usize) -> usize {
    count_in_world(crate::regions::active(), key)
}

pub extern "C" fn count_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).count }
}
#[export_name = "jointArrayPtr"]
pub extern "C" fn pointer(key: usize) -> usize {
    pointer_in_world(crate::regions::active(), key)
}

pub extern "C" fn pointer_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).records.layout[0] as usize }
}
#[export_name = "jointArrayAppend"]
pub extern "C" fn append(key: usize) -> usize {
    append_in_world(crate::regions::active(), key)
}

pub extern "C" fn append_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).append() }
}
#[export_name = "jointArrayRemove"]
pub extern "C" fn remove(key: usize, index: usize) -> u32 {
    remove_in_world(crate::regions::active(), key, index)
}

pub extern "C" fn remove_in_world(world_index: usize, key: usize, index: usize) -> u32 {
    unsafe {
        let moved = array(world_index, key).remove(index);
        if moved != NULL_INDEX {
            crate::joint_record::set_location(world_index, moved as usize, key, index);
        }
        moved
    }
}
#[export_name = "jointArrayMove"]
pub extern "C" fn move_record(source: usize, index: usize, target: usize) -> u32 {
    move_record_in_world(crate::regions::active(), source, index, target)
}

pub extern "C" fn move_record_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
) -> u32 {
    assert_ne!(source, target);
    unsafe {
        assert!(index < array(world_index, source).count);
        let destination = array(world_index, target).append();
        core::ptr::copy_nonoverlapping(
            array(world_index, source).ptr(index),
            array(world_index, target).ptr(destination),
            JOINT_STRIDE,
        );
        let id = *array(world_index, target).ptr(destination).add(J_JOINT_ID);
        crate::joint_record::set_location(world_index, id as usize, target, destination);
        remove_in_world(world_index, source, index)
    }
}
#[export_name = "jointReadFloat"]
pub extern "C" fn read_float(key: usize, index: usize, field: usize) -> f32 {
    read_float_in_world(crate::regions::active(), key, index, field)
}

pub extern "C" fn read_float_in_world(
    world_index: usize,
    key: usize,
    index: usize,
    field: usize,
) -> f32 {
    unsafe {
        assert!(index < array(world_index, key).count && field < JOINT_STRIDE);
        f32::from_bits(*array(world_index, key).ptr(index).add(field))
    }
}
#[export_name = "jointWriteFloat"]
pub extern "C" fn write_float(key: usize, index: usize, field: usize, value: f32) {
    write_float_in_world(crate::regions::active(), key, index, field, value)
}

pub extern "C" fn write_float_in_world(
    world_index: usize,
    key: usize,
    index: usize,
    field: usize,
    value: f32,
) {
    unsafe {
        assert!(index < array(world_index, key).count && field < JOINT_STRIDE);
        *array(world_index, key).ptr(index).add(field) = value.to_bits();
    }
}
#[export_name = "jointReadWord"]
pub extern "C" fn read_word(key: usize, index: usize, field: usize) -> u32 {
    read_word_in_world(crate::regions::active(), key, index, field)
}

pub extern "C" fn read_word_in_world(
    world_index: usize,
    key: usize,
    index: usize,
    field: usize,
) -> u32 {
    unsafe {
        assert!(index < array(world_index, key).count);
        if field & crate::joint_abi::BOOL_FIELD != 0 {
            crate::joint_abi::read_flags(column(world_index, key), index, field)
        } else {
            assert!(field < JOINT_STRIDE);
            *array(world_index, key).ptr(index).add(field)
        }
    }
}
#[export_name = "jointWriteWord"]
pub extern "C" fn write_word(key: usize, index: usize, field: usize, value: u32) {
    write_word_in_world(crate::regions::active(), key, index, field, value)
}

pub extern "C" fn write_word_in_world(
    world_index: usize,
    key: usize,
    index: usize,
    field: usize,
    value: u32,
) {
    unsafe {
        assert!(index < array(world_index, key).count);
        if field & crate::joint_abi::BOOL_FIELD != 0 {
            crate::joint_abi::write_flags(column(world_index, key), index, field, value);
        } else {
            assert!(field < JOINT_STRIDE);
            *array(world_index, key).ptr(index).add(field) = value;
        }
    }
}
pub unsafe fn column(world_index: usize, key: usize) -> Col<'static, f32> {
    let a = array(world_index, key);
    Col::new(a.ptr(0) as *mut f32, a.count * JOINT_STRIDE)
}
pub unsafe fn reset(id: usize) {
    for color in 0..crate::constraint_graph::COLORS {
        let a = crate::constraint_graph::joint_array(id, color);
        a.records.release();
        *a = JointArray::EMPTY;
    }
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, crate::constraint_graph::COLORS);
    for color in 0..crate::constraint_graph::COLORS {
        let a = crate::constraint_graph::joint_array(id, color);
        regions::write_word(out, a.count);
        a.records.snapshot(out);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let count = regions::read_word(input);
    assert_eq!(count, crate::constraint_graph::COLORS);
    for color in 0..crate::constraint_graph::COLORS {
        let a = crate::constraint_graph::joint_array(id, color);
        a.count = regions::read_word(input);
        a.records.restore(input);
    }
}
