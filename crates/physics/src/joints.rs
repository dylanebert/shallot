//! Joint-sim arrays owned by each world, in graph-color and solver-set order.
use crate::col::Col;
use crate::joint_abi::{JOINT_STRIDE, NULL_INDEX};
use crate::joint_sim::JointSim;
use crate::regions;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JointArray {
    data: *mut JointSim,
    pub(crate) count: usize,
    capacity: usize,
}
impl JointArray {
    pub(crate) const EMPTY: Self = Self {
        data: 16 as *mut JointSim,
        count: 0,
        capacity: 0,
    };
    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(
            (capacity * core::mem::size_of::<JointSim>()).next_multiple_of(16),
            16,
        )
        .unwrap()
    }
    pub(crate) unsafe fn reserve(&mut self, capacity: usize) {
        if capacity <= self.capacity {
            return;
        }
        i32::try_from(capacity).expect("joint array capacity exceeds Box3D int range");
        let layout = Self::layout(capacity);
        let data = alloc(layout).cast::<JointSim>();
        if data.is_null() {
            handle_alloc_error(layout);
        }
        if self.capacity != 0 {
            core::ptr::copy_nonoverlapping(self.data, data, self.capacity);
            dealloc(self.data.cast(), Self::layout(self.capacity));
        }
        self.data = data;
        self.capacity = capacity;
        regions::invalidate_views();
    }
    unsafe fn ptr(&self, index: usize) -> *mut u32 {
        self.data.add(index).cast()
    }
    unsafe fn emplace(&mut self) -> usize {
        let index = self.count;
        if index == self.capacity {
            self.reserve(if self.capacity == 0 {
                16
            } else {
                self.capacity.checked_mul(2).unwrap()
            });
        }
        self.count += 1;
        index
    }
    unsafe fn append(&mut self) -> usize {
        let index = self.emplace();
        self.data.add(index).write_bytes(0, 1);
        index
    }
    unsafe fn remove(&mut self, index: usize) -> u32 {
        assert!(index < self.count);
        self.count -= 1;
        if index == self.count {
            return NULL_INDEX;
        }
        core::ptr::copy_nonoverlapping(self.data.add(self.count), self.data.add(index), 1);
        (*self.data.add(index)).joint_id as u32
    }
    pub(crate) unsafe fn release(&mut self) {
        if self.capacity != 0 {
            dealloc(self.data.cast(), Self::layout(self.capacity));
            regions::invalidate_views();
        }
        *self = Self::EMPTY;
    }
    pub(crate) unsafe fn snapshot(&self, out: &mut Vec<u8>) {
        regions::write_word(out, self.count);
        out.extend_from_slice(core::slice::from_raw_parts(
            self.data.cast::<u8>(),
            self.count * core::mem::size_of::<JointSim>(),
        ));
    }
    pub(crate) unsafe fn restore(&mut self, input: &mut &[u8]) {
        self.count = regions::read_word(input);
        self.reserve(self.count);
        assert!(self.count <= self.capacity);
        let (bytes, rest) = input.split_at(self.count * core::mem::size_of::<JointSim>());
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.cast(), bytes.len());
        *input = rest;
    }
}
unsafe fn array(world_index: usize, key: usize) -> &'static mut JointArray {
    if key >= crate::constraint_graph::COLORS {
        return crate::solver_set::joint_array(world_index, key - crate::constraint_graph::COLORS);
    }
    crate::constraint_graph::joint_array(world_index, key)
}
#[export_name = "jointArrayCount"]
pub extern "C" fn count(key: usize) -> usize {
    count_in_world(crate::regions::active(), key)
}

pub(crate) fn count_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).count }
}
#[export_name = "jointArrayPtr"]
pub extern "C" fn pointer(key: usize) -> usize {
    pointer_in_world(crate::regions::active(), key)
}

pub(crate) fn pointer_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).data as usize }
}
pub(crate) fn append_in_world(world_index: usize, key: usize) -> usize {
    unsafe { array(world_index, key).append() }
}
pub(crate) fn remove_in_world(world_index: usize, key: usize, index: usize) -> u32 {
    unsafe {
        let moved = array(world_index, key).remove(index);
        if moved != NULL_INDEX {
            crate::joint_record::set_location(world_index, moved as usize, key, index);
        }
        moved
    }
}
pub(crate) fn move_record_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
) -> u32 {
    unsafe {
        copy_record_in_world(world_index, source, index, target);
        remove_in_world(world_index, source, index)
    }
}

pub(crate) unsafe fn copy_record_in_world(
    world: usize,
    source: usize,
    index: usize,
    target: usize,
) {
    assert_ne!(source, target);
    assert!(index < array(world, source).count);
    let destination = if target < crate::constraint_graph::COLORS {
        array(world, target).append()
    } else {
        array(world, target).emplace()
    };
    core::ptr::copy_nonoverlapping(
        array(world, source).data.add(index),
        array(world, target).data.add(destination),
        1,
    );
    let id = (*array(world, target).data.add(destination)).joint_id;
    crate::joint_record::set_location(world, id as usize, target, destination);
}
pub(crate) fn read_float_in_world(
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
pub(crate) fn write_float_in_world(
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

pub(crate) fn read_word_in_world(
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
pub(crate) fn write_word_in_world(
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
        a.release();
    }
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, crate::constraint_graph::COLORS);
    for color in 0..crate::constraint_graph::COLORS {
        let a = crate::constraint_graph::joint_array(id, color);
        a.snapshot(out);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let count = regions::read_word(input);
    assert_eq!(count, crate::constraint_graph::COLORS);
    for color in 0..crate::constraint_graph::COLORS {
        let a = crate::constraint_graph::joint_array(id, color);
        a.restore(input);
    }
}
