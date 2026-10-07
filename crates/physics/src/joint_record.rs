/// Joint identity and topology, separate from its solver-set simulation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JointEdge {
    pub body_id: i32,
    pub prev_key: i32,
    pub next_key: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct JointRecord {
    pub set_index: i32,
    pub color_index: i32,
    pub local_index: i32,
    pub edges: [JointEdge; 2],
    pub joint_id: i32,
    pub island_id: i32,
    pub island_index: i32,
    pub draw_scale: f32,
    pub joint_type: i32,
    pub generation: u16,
    pub collide_connected: bool,
    pub(crate) padding: u8,
}

impl JointRecord {
    pub const EMPTY: Self = Self {
        set_index: -1,
        color_index: -1,
        local_index: -1,
        edges: [JointEdge {
            body_id: -1,
            prev_key: -1,
            next_key: -1,
        }; 2],
        joint_id: -1,
        island_id: -1,
        island_index: -1,
        draw_scale: 0.0,
        joint_type: 0,
        generation: 0,
        collide_connected: false,
        padding: 0,
    };
}

#[cfg(target_arch = "wasm32")]
pub use runtime::*;

#[cfg(target_arch = "wasm32")]
mod runtime {
    use super::JointRecord;
    use crate::regions::{self, Columns, MAX_WORLDS};

    struct Records {
        records: Columns<1>,
        next: usize,
        free: Vec<u32>,
    }
    impl Records {
        const EMPTY: Self = Self {
            records: Columns::EMPTY,
            next: 0,
            free: Vec::new(),
        };
    }
    static mut WORLDS: [Records; MAX_WORLDS] = [const { Records::EMPTY }; MAX_WORLDS];

    pub unsafe fn record(world_index: usize, id: usize) -> &'static JointRecord {
        &*(WORLDS[world_index].records.layout[0] as usize as *const JointRecord).add(id)
    }
    pub unsafe fn record_mut(world_index: usize, id: usize) -> &'static mut JointRecord {
        &mut *(WORLDS[world_index].records.layout[0] as usize as *mut JointRecord).add(id)
    }

    pub unsafe fn alloc(world_index: usize) -> u32 {
        let w = &mut WORLDS[world_index];
        let id = if let Some(id) = w.free.pop() {
            id
        } else {
            let id = w.next;
            w.next += 1;
            w.records
                .reserve(0, w.next * core::mem::size_of::<JointRecord>());
            w.free.reserve(w.next.saturating_sub(w.free.len()));
            (w.records.layout[0] as usize as *mut JointRecord)
                .add(id)
                .write(JointRecord::EMPTY);
            id as u32
        };
        let r = record_mut(world_index, id as usize);
        let generation = r.generation.wrapping_add(1);
        *r = JointRecord::EMPTY;
        r.generation = generation;
        r.joint_id = id as i32;
        id
    }

    pub unsafe fn link_bodies(world_index: usize, id: usize, a: usize, b: usize) {
        for (side, body_id) in [a, b].into_iter().enumerate() {
            let body = crate::bodies::record_mut(world_index, body_id);
            let head = body.head_joint_key;
            let key = ((id as i32) << 1) | side as i32;
            record_mut(world_index, id).edges[side] = super::JointEdge {
                body_id: body_id as i32,
                prev_key: -1,
                next_key: head,
            };
            if head != -1 {
                record_mut(world_index, (head >> 1) as usize).edges[(head & 1) as usize].prev_key =
                    key;
            }
            body.head_joint_key = key;
            body.joint_count += 1;
        }
    }

    pub unsafe fn unlink_bodies(world_index: usize, id: usize) {
        let edges = record(world_index, id).edges;
        for (side, edge) in edges.into_iter().enumerate() {
            if edge.prev_key != -1 {
                record_mut(world_index, (edge.prev_key >> 1) as usize).edges
                    [(edge.prev_key & 1) as usize]
                    .next_key = edge.next_key;
            }
            if edge.next_key != -1 {
                record_mut(world_index, (edge.next_key >> 1) as usize).edges
                    [(edge.next_key & 1) as usize]
                    .prev_key = edge.prev_key;
            }
            let body = crate::bodies::record_mut(world_index, edge.body_id as usize);
            if body.head_joint_key == (((id as i32) << 1) | side as i32) {
                body.head_joint_key = edge.next_key;
            }
            body.joint_count -= 1;
        }
    }

    pub unsafe fn set_location(world_index: usize, id: usize, key: usize, index: usize) {
        let r = record_mut(world_index, id);
        if key < crate::constraint_graph::COLORS {
            r.set_index = 2;
            r.color_index = key as i32;
        } else {
            r.set_index = (key - crate::constraint_graph::COLORS) as i32;
            r.color_index = -1;
        }
        r.local_index = index as i32;
    }

    pub unsafe fn free(world_index: usize, id: u32) {
        let r = record_mut(world_index, id as usize);
        r.set_index = -1;
        r.color_index = -1;
        r.local_index = -1;
        r.joint_id = -1;
        WORLDS[world_index].free.push(id);
    }

    #[export_name = "jointSimPtr"]
    pub unsafe extern "C" fn sim_pointer(id: usize) -> usize {
        unsafe { sim_pointer_in_world(crate::regions::active(), id) }
    }

    pub unsafe extern "C" fn sim_pointer_in_world(world_index: usize, id: usize) -> usize {
        let r = record(world_index, id);
        assert!(r.set_index >= 0 && r.local_index >= 0);
        let key = if r.set_index == 2 {
            r.color_index as usize
        } else {
            crate::constraint_graph::COLORS + r.set_index as usize
        };
        assert!((r.local_index as usize) < crate::joints::count_in_world(world_index, key));
        crate::joints::pointer_in_world(world_index, key)
            + r.local_index as usize * crate::joint_abi::JOINT_STRIDE * 4
    }

    #[export_name = "jointRecordPtr"]
    pub unsafe extern "C" fn pointer() -> usize {
        unsafe { pointer_in_world(crate::regions::active()) }
    }

    pub unsafe extern "C" fn pointer_in_world(world_index: usize) -> usize {
        WORLDS[world_index].records.layout[0] as usize
    }

    #[export_name = "jointRecordCount"]
    pub unsafe extern "C" fn count() -> usize {
        unsafe { count_in_world(crate::regions::active()) }
    }

    pub unsafe extern "C" fn count_in_world(world_index: usize) -> usize {
        let w = &WORLDS[world_index];
        w.next - w.free.len()
    }

    #[export_name = "jointRecordCapacity"]
    pub unsafe extern "C" fn capacity() -> usize {
        unsafe { capacity_in_world(crate::regions::active()) }
    }

    pub unsafe extern "C" fn capacity_in_world(world_index: usize) -> usize {
        WORLDS[world_index].next
    }

    pub unsafe fn reset(world: usize) {
        WORLDS[world].records.release();
        WORLDS[world] = Records::EMPTY;
    }
    pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
        let w = &WORLDS[world];
        regions::write_word(out, w.next);
        regions::write_word(out, w.free.len());
        for &id in &w.free {
            regions::write_word(out, id as usize);
        }
        w.records.snapshot(out);
    }
    pub unsafe fn restore(world: usize, input: &mut &[u8]) {
        reset(world);
        let w = &mut WORLDS[world];
        w.next = regions::read_word(input);
        let count = regions::read_word(input);
        w.free.reserve(w.next);
        for _ in 0..count {
            w.free.push(regions::read_word(input) as u32);
        }
        w.records.restore(input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn joint_record_layout_matches_plain_word_bindings() {
        assert_eq!(size_of::<JointEdge>(), 12);
        assert_eq!(size_of::<JointRecord>(), 60);
        assert_eq!(offset_of!(JointRecord, edges), 12);
        assert_eq!(offset_of!(JointRecord, joint_id), 36);
        assert_eq!(offset_of!(JointRecord, island_id), 40);
        assert_eq!(offset_of!(JointRecord, island_index), 44);
        assert_eq!(offset_of!(JointRecord, draw_scale), 48);
        assert_eq!(offset_of!(JointRecord, joint_type), 52);
        assert_eq!(offset_of!(JointRecord, generation), 56);
        assert_eq!(offset_of!(JointRecord, collide_connected), 58);
    }
}
