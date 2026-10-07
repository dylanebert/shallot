use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};

unsafe fn allocate(size: usize) -> *mut u8 {
    if size == 0 {
        return 16 as *mut u8;
    }
    let layout = Layout::from_size_align(size, 16).unwrap();
    let data = alloc(layout);
    if data.is_null() {
        handle_alloc_error(layout);
    }
    data
}

unsafe fn release(data: *mut u8, size: usize) {
    if size != 0 {
        dealloc(data, Layout::from_size_align(size, 16).unwrap());
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StackEntry {
    data: *mut u8,
    name: *const u8,
    size: usize,
    used_malloc: bool,
}

#[repr(C)]
pub(crate) struct Stack {
    memory: *mut u8,
    capacity: usize,
    index: usize,
    allocation: usize,
    max_allocation: usize,
    entries: [StackEntry; 32],
    entry_count: usize,
}

impl Stack {
    pub(crate) const EMPTY: Self = Self {
        memory: 16 as *mut u8,
        capacity: 0,
        index: 0,
        entries: [StackEntry {
            data: 16 as *mut u8,
            name: std::ptr::null(),
            size: 0,
            used_malloc: false,
        }; 32],
        entry_count: 0,
        allocation: 0,
        max_allocation: 0,
    };

    pub(crate) fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    pub(crate) unsafe fn new(capacity: usize) -> Self {
        Self {
            memory: allocate(capacity),
            capacity,
            ..Self::EMPTY
        }
    }

    pub(crate) unsafe fn alloc(&mut self, size: usize) -> *mut u8 {
        assert!(self.entry_count < self.entries.len());
        let size = size.next_multiple_of(16);
        let used_malloc = self.index + size > self.capacity;
        let data = if used_malloc {
            allocate(size)
        } else {
            let data = self.memory.add(self.index);
            self.index += size;
            data
        };
        self.allocation += size;
        self.max_allocation = self.max_allocation.max(self.allocation);
        self.entries[self.entry_count] = StackEntry {
            data,
            name: c"step columns".as_ptr().cast(),
            size,
            used_malloc,
        };
        self.entry_count += 1;
        data
    }

    pub(crate) unsafe fn free(&mut self, data: *mut u8) {
        assert!(self.entry_count > 0);
        let entry = self.entries[self.entry_count - 1];
        assert_eq!(data, entry.data);
        if entry.used_malloc {
            release(data, entry.size);
        } else {
            self.index -= entry.size;
        }
        self.allocation -= entry.size;
        self.entry_count -= 1;
    }

    pub(crate) unsafe fn grow(&mut self) {
        assert_eq!(self.allocation, 0);
        if self.max_allocation > self.capacity {
            release(self.memory, self.capacity);
            self.capacity = self.max_allocation + self.max_allocation / 2;
            self.memory = allocate(self.capacity);
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        assert_eq!(self.entry_count, 0);
        unsafe { release(self.memory, self.capacity) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_overflow_preserves_index_and_grows_only_when_empty() {
        unsafe {
            let mut stack = Stack::new(32);
            let first = stack.alloc(17);
            let overflow = stack.alloc(1);
            assert_eq!(stack.index, 32);
            assert_eq!(stack.allocation, 48);
            assert_eq!(stack.max_allocation, 48);
            stack.free(overflow);
            stack.free(first);
            stack.grow();
            assert_eq!(stack.capacity, 72);
            let next = stack.alloc(48);
            assert_eq!(next, stack.memory);
            stack.free(next);
            stack.grow();
            assert_eq!(stack.capacity, 72);
        }
    }

    #[test]
    #[should_panic]
    fn stack_requires_lifo_free() {
        unsafe {
            let mut stack = Stack::new(32);
            let first = stack.alloc(16);
            let second = stack.alloc(16);
            // Avoid a second panic in Drop while testing the rejected operation.
            let mut stack = std::mem::ManuallyDrop::new(stack);
            assert_ne!(first, second);
            stack.free(first);
        }
    }
}
