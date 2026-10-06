// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT
// Persistent connectivity follows Box3D's island.c and island.h.
use crate::manifold_abi::{DIR_EDGE_A, DIR_EDGE_B, DIR_ISLAND_ID, DIR_ISLAND_INDEX, DIR_STRIDE};
use crate::{manifolds, regions, solver_set};
#[derive(Clone, Copy)]
pub struct Link {
    pub id: i32,
    pub body_a: i32,
    pub body_b: i32,
}
pub struct Island {
    pub set_index: i32,
    pub local_index: i32,
    pub island_id: i32,
    pub constraint_remove_count: i32,
    pub bodies: Vec<i32>,
    pub contacts: Vec<Link>,
    pub joints: Vec<Link>,
}
impl Default for Island {
    fn default() -> Self {
        Self {
            set_index: -1,
            local_index: -1,
            island_id: -1,
            constraint_remove_count: 0,
            bodies: Vec::new(),
            contacts: Vec::new(),
            joints: Vec::new(),
        }
    }
}
struct Islands {
    records: Vec<Island>,
    free: Vec<usize>,
    fixes: Vec<i32>,
}
static mut WORLDS: [Islands; regions::MAX_WORLDS] = [const {
    Islands {
        records: Vec::new(),
        free: Vec::new(),
        fixes: Vec::new(),
    }
}; regions::MAX_WORLDS];
unsafe fn world() -> &'static mut Islands {
    &mut WORLDS[regions::active()]
}
unsafe fn record(id: usize) -> &'static mut Island {
    &mut world().records[id]
}
unsafe fn fix(kind: i32, id: i32, island: i32, index: i32) {
    world().fixes.extend_from_slice(&[kind, id, island, index]);
}
unsafe fn contact_fix(id: i32, island: i32, index: i32) {
    let d = manifolds::dir_col();
    let base = id as usize * DIR_STRIDE;
    d.set(base + DIR_ISLAND_ID, island as u32);
    d.set(base + DIR_ISLAND_INDEX, index as u32);
}
#[export_name = "islandFixCount"]
pub unsafe extern "C" fn fix_count() -> usize {
    world().fixes.len()
}
#[export_name = "islandFixData"]
pub unsafe extern "C" fn fix_data() -> *const i32 {
    world().fixes.as_ptr()
}
#[export_name = "islandFixClear"]
pub unsafe extern "C" fn fix_clear() {
    world().fixes.clear();
}
#[export_name = "islandCount"]
pub unsafe extern "C" fn count() -> usize {
    world().records.len() - world().free.len()
}
#[export_name = "islandCreate"]
pub unsafe extern "C" fn create(set: usize) -> usize {
    let w = world();
    let id = if let Some(id) = w.free.pop() {
        id
    } else {
        w.records.push(Island::default());
        w.records.len() - 1
    };
    let index = solver_set::array_push(set, 1, id as i32);
    w.records[id] = Island {
        set_index: set as i32,
        local_index: index as i32,
        island_id: id as i32,
        ..Island::default()
    };
    id
}
#[export_name = "islandDestroy"]
pub unsafe extern "C" fn destroy(id: usize) {
    let s = record(id);
    let set = s.set_index as usize;
    let index = s.local_index as usize;
    let last = solver_set::array_count(set, 1) - 1;
    let moved = solver_set::array_get(set, 1, last);
    solver_set::array_write(set, 1, index, moved);
    record(moved as usize).local_index = index as i32;
    solver_set::array_pop(set, 1);
    world().records[id] = Island::default();
    world().free.push(id);
}
#[export_name = "islandField"]
pub unsafe extern "C" fn field(id: usize, field: usize) -> i32 {
    let s = record(id);
    match field {
        0 => s.set_index,
        1 => s.local_index,
        2 => s.island_id,
        3 => s.constraint_remove_count,
        _ => unreachable!(),
    }
}
#[export_name = "islandSetField"]
pub unsafe extern "C" fn set_field(id: usize, field: usize, value: i32) {
    let s = record(id);
    match field {
        0 => s.set_index = value,
        1 => s.local_index = value,
        3 => s.constraint_remove_count = value,
        _ => unreachable!(),
    }
}
#[export_name = "islandArrayCount"]
pub unsafe extern "C" fn array_count(id: usize, kind: usize) -> usize {
    let s = record(id);
    match kind {
        0 => s.bodies.len(),
        1 => s.contacts.len(),
        2 => s.joints.len(),
        _ => unreachable!(),
    }
}
#[export_name = "islandArrayGet"]
pub unsafe extern "C" fn array_get(id: usize, kind: usize, index: usize, lane: usize) -> i32 {
    let s = record(id);
    if kind == 0 {
        return s.bodies[index];
    }
    let l = if kind == 1 {
        s.contacts[index]
    } else {
        s.joints[index]
    };
    match lane {
        0 => l.id,
        1 => l.body_a,
        2 => l.body_b,
        _ => unreachable!(),
    }
}
#[export_name = "islandAddBody"]
pub unsafe extern "C" fn add_body(id: usize, body: i32) {
    let s = record(id);
    fix(0, body, id as i32, s.bodies.len() as i32);
    s.bodies.push(body);
}
#[export_name = "islandRemoveBody"]
pub unsafe extern "C" fn remove_body(id: usize, index: usize) {
    let s = record(id);
    let removed = s.bodies.swap_remove(index);
    if index < s.bodies.len() {
        fix(0, s.bodies[index], id as i32, index as i32);
    }
    fix(0, removed, -1, -1);
}
unsafe fn merge(a: i32, b: i32) -> usize {
    if a == b || b == -1 {
        return a as usize;
    }
    if a == -1 {
        return b as usize;
    }
    let (big, small) = if record(a as usize).bodies.len() >= record(b as usize).bodies.len() {
        (a as usize, b as usize)
    } else {
        (b as usize, a as usize)
    };
    let bodies = std::mem::take(&mut record(small).bodies);
    record(big).bodies.reserve(bodies.len());
    for id in bodies {
        add_body(big, id);
    }
    let contacts = std::mem::take(&mut record(small).contacts);
    record(big).contacts.reserve(contacts.len());
    for l in contacts {
        add_contact(big, l);
    }
    let joints = std::mem::take(&mut record(small).joints);
    record(big).joints.reserve(joints.len());
    for l in joints {
        add_joint(big, l);
    }
    record(big).constraint_remove_count += record(small).constraint_remove_count;
    destroy(small);
    big
}
unsafe fn add_contact(id: usize, l: Link) {
    let s = record(id);
    contact_fix(l.id, id as i32, s.contacts.len() as i32);
    s.contacts.push(l);
}
unsafe fn add_joint(id: usize, l: Link) {
    let s = record(id);
    fix(1, l.id, id as i32, s.joints.len() as i32);
    s.joints.push(l);
}
#[export_name = "islandLinkContact"]
pub unsafe extern "C" fn link_contact(contact: i32, a: i32, b: i32) {
    let id = merge(a, b);
    let d = manifolds::dir_col();
    let base = contact as usize * DIR_STRIDE;
    add_contact(
        id,
        Link {
            id: contact,
            body_a: d.get(base + DIR_EDGE_A) as i32,
            body_b: d.get(base + DIR_EDGE_B) as i32,
        },
    );
}
#[export_name = "islandUnlinkContact"]
pub unsafe extern "C" fn unlink_contact(contact: i32) {
    let d = manifolds::dir_col();
    let base = contact as usize * DIR_STRIDE;
    let id = d.get(base + DIR_ISLAND_ID) as usize;
    let index = d.get(base + DIR_ISLAND_INDEX) as usize;
    let s = record(id);
    s.contacts.swap_remove(index);
    if index < s.contacts.len() {
        contact_fix(s.contacts[index].id, id as i32, index as i32);
    }
    contact_fix(contact, -1, -1);
    s.constraint_remove_count += 1;
}
#[export_name = "islandLinkJoint"]
pub unsafe extern "C" fn link_joint(joint: i32, body_a: i32, body_b: i32, a: i32, b: i32) {
    let id = merge(a, b);
    add_joint(
        id,
        Link {
            id: joint,
            body_a,
            body_b,
        },
    );
}
#[export_name = "islandUnlinkJoint"]
pub unsafe extern "C" fn unlink_joint(joint: i32, id: i32, index: usize) {
    if id == -1 {
        return;
    }
    let s = record(id as usize);
    s.joints.swap_remove(index);
    if index < s.joints.len() {
        fix(1, s.joints[index].id, id, index as i32);
    }
    fix(1, joint, -1, -1);
    s.constraint_remove_count += 1;
}
fn find_parent(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        let grand = parents[parents[node]];
        parents[node] = grand;
        node = grand;
    }
    node
}
fn union(
    parents: &mut [usize],
    ranks: &mut [usize],
    a: usize,
    b: usize,
    contacts: &mut [usize],
    joints: &mut [usize],
) {
    let a = find_parent(parents, a);
    let b = find_parent(parents, b);
    if a == b {
        return;
    }
    if ranks[a] < ranks[b] {
        parents[a] = b;
        contacts[b] += contacts[a];
        joints[b] += joints[a];
    } else {
        parents[b] = a;
        contacts[a] += contacts[b];
        joints[a] += joints[b];
        if ranks[a] == ranks[b] {
            ranks[a] += 1;
        }
    }
}
#[export_name = "islandSplitIndices"]
pub unsafe extern "C" fn split_indices(count: usize) -> usize {
    crate::arena::reserve_scratch(count * 4)
}
#[export_name = "islandSplit"]
pub unsafe extern "C" fn split(base: usize, body_indices: *const i32, body_count: usize) {
    // Body records remain in TypeScript until stage 6; only their island indices cross this seam.
    let indices = std::slice::from_raw_parts(body_indices, body_count);
    let n = record(base).bodies.len();
    let mut parents: Vec<usize> = (0..n).collect();
    let mut ranks = vec![0; n];
    let mut contact_counts = vec![0; n];
    let mut joint_counts = vec![0; n];
    for kind in 0..2 {
        let links = if kind == 0 {
            &record(base).contacts
        } else {
            &record(base).joints
        };
        for l in links {
            let a = indices[l.body_a as usize];
            let b = indices[l.body_b as usize];
            if a != -1 && b != -1 {
                union(
                    &mut parents,
                    &mut ranks,
                    a as usize,
                    b as usize,
                    &mut contact_counts,
                    &mut joint_counts,
                );
            }
            let root = find_parent(&mut parents, if a != -1 { a } else { b } as usize);
            if kind == 0 {
                contact_counts[root] += 1;
            } else {
                joint_counts[root] += 1;
            }
        }
    }
    drop(ranks);
    let mut components = 0;
    for i in 0..n {
        parents[i] = find_parent(&mut parents, i);
        if parents[i] == i {
            components += 1;
        }
    }
    if components == 1 {
        record(base).constraint_remove_count = 0;
        return;
    }
    let bodies = std::mem::take(&mut record(base).bodies);
    let contacts = std::mem::take(&mut record(base).contacts);
    let joints = std::mem::take(&mut record(base).joints);
    let mut root_map = vec![usize::MAX; n];
    let mut body_counts = vec![0; components];
    let mut component_contacts = vec![0; components];
    let mut component_joints = vec![0; components];
    let mut island_count = 0;
    for i in 0..n {
        let root = parents[i];
        if root_map[root] == usize::MAX {
            root_map[root] = island_count;
            component_contacts[island_count] = contact_counts[root];
            component_joints[island_count] = joint_counts[root];
            island_count += 1;
        }
        body_counts[root_map[root]] += 1;
    }
    let mut ids = Vec::with_capacity(island_count);
    for i in 0..island_count {
        let id = create(2);
        ids.push(id);
        let s = record(id);
        s.bodies.reserve(body_counts[i]);
        s.contacts.reserve(component_contacts[i]);
        s.joints.reserve(component_joints[i]);
    }
    for (i, &body) in bodies.iter().enumerate() {
        add_body(ids[root_map[parents[i]]], body);
    }
    for l in contacts {
        let a = indices[l.body_a as usize];
        let b = indices[l.body_b as usize];
        let index = if a != -1 { a } else { b } as usize;
        add_contact(ids[root_map[parents[index]]], l);
    }
    for l in joints {
        let a = indices[l.body_a as usize];
        let b = indices[l.body_b as usize];
        let index = if a != -1 { a } else { b } as usize;
        add_joint(ids[root_map[parents[index]]], l);
    }
    destroy(base);
}
pub unsafe fn reset(id: usize) {
    WORLDS[id] = Islands {
        records: Vec::new(),
        free: Vec::new(),
        fixes: Vec::new(),
    };
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    regions::write_word(out, w.records.len());
    for s in &w.records {
        for v in [
            s.set_index,
            s.local_index,
            s.island_id,
            s.constraint_remove_count,
        ] {
            regions::write_word(out, v as usize);
        }
        regions::write_word(out, s.bodies.len());
        for &v in &s.bodies {
            regions::write_word(out, v as usize);
        }
        for links in [&s.contacts, &s.joints] {
            regions::write_word(out, links.len());
            for l in links {
                for v in [l.id, l.body_a, l.body_b] {
                    regions::write_word(out, v as usize);
                }
            }
        }
    }
    regions::write_word(out, w.free.len());
    for &v in &w.free {
        regions::write_word(out, v);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let w = &mut WORLDS[id];
    let n = regions::read_word(input);
    for _ in 0..n {
        let mut s = Island::default();
        s.set_index = regions::read_word(input) as i32;
        s.local_index = regions::read_word(input) as i32;
        s.island_id = regions::read_word(input) as i32;
        s.constraint_remove_count = regions::read_word(input) as i32;
        let n = regions::read_word(input);
        for _ in 0..n {
            s.bodies.push(regions::read_word(input) as i32);
        }
        for links in [&mut s.contacts, &mut s.joints] {
            let n = regions::read_word(input);
            for _ in 0..n {
                links.push(Link {
                    id: regions::read_word(input) as i32,
                    body_a: regions::read_word(input) as i32,
                    body_b: regions::read_word(input) as i32,
                });
            }
        }
        w.records.push(s);
    }
    let n = regions::read_word(input);
    for _ in 0..n {
        w.free.push(regions::read_word(input));
    }
}
