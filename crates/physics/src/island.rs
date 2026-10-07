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
    split_island_id: i32,
    records: Vec<Island>,
    free: Vec<usize>,
}
static mut WORLDS: [Islands; regions::MAX_WORLDS] = [const {
    Islands {
        split_island_id: -1,
        records: Vec::new(),
        free: Vec::new(),
    }
}; regions::MAX_WORLDS];
unsafe fn world(world_index: usize) -> &'static mut Islands {
    &mut WORLDS[world_index]
}
unsafe fn record(world_index: usize, id: usize) -> &'static mut Island {
    &mut world(world_index).records[id]
}
#[export_name = "islandSplitCandidate"]
pub unsafe extern "C" fn split_candidate() -> i32 {
    unsafe { split_candidate_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn split_candidate_in_world(world_index: usize) -> i32 {
    world(world_index).split_island_id
}
#[export_name = "islandSetSplitCandidate"]
pub unsafe extern "C" fn set_split_candidate(id: i32) {
    unsafe { set_split_candidate_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn set_split_candidate_in_world(world_index: usize, id: i32) {
    world(world_index).split_island_id = id;
}
#[export_name = "islandCanSleep"]
pub unsafe extern "C" fn can_sleep(id: usize) -> bool {
    unsafe { can_sleep_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn can_sleep_in_world(world_index: usize, id: usize) -> bool {
    record(world_index, id)
        .bodies
        .iter()
        .all(|&body| crate::bodies::record(world_index, body as usize).sleep_time >= 0.5)
}

unsafe fn fix(world_index: usize, kind: i32, id: i32, island: i32, index: i32) {
    if kind == 0 {
        let body = crate::bodies::record_mut(world_index, id as usize);
        body.island_id = island;
        body.island_index = index;
    } else {
        let joint = crate::joint_record::record_mut(world_index, id as usize);
        joint.island_id = island;
        joint.island_index = index;
    }
}
unsafe fn contact_fix(world_index: usize, id: i32, island: i32, index: i32) {
    let d = manifolds::dir_col(world_index);
    let base = id as usize * DIR_STRIDE;
    d.set(base + DIR_ISLAND_ID, island as u32);
    d.set(base + DIR_ISLAND_INDEX, index as u32);
}
#[export_name = "islandCount"]
pub unsafe extern "C" fn count() -> usize {
    unsafe { count_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn count_in_world(world_index: usize) -> usize {
    world(world_index).records.len() - world(world_index).free.len()
}
#[export_name = "islandCreate"]
pub unsafe extern "C" fn create(set: usize) -> usize {
    unsafe { create_in_world(crate::regions::active(), set) }
}

pub unsafe extern "C" fn create_in_world(world_index: usize, set: usize) -> usize {
    let w = world(world_index);
    let id = if let Some(id) = w.free.pop() {
        id
    } else {
        w.records.push(Island::default());
        w.records.len() - 1
    };
    let index = solver_set::array_push_in_world(world_index, set, 1, id as i32);
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
    unsafe { destroy_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn destroy_in_world(world_index: usize, id: usize) {
    if world(world_index).split_island_id == id as i32 {
        world(world_index).split_island_id = -1;
    }
    let s = record(world_index, id);
    let set = s.set_index as usize;
    let index = s.local_index as usize;
    let last = solver_set::array_count_in_world(world_index, set, 1) - 1;
    let moved = solver_set::array_get_in_world(world_index, set, 1, last);
    solver_set::array_write_in_world(world_index, set, 1, index, moved);
    record(world_index, moved as usize).local_index = index as i32;
    solver_set::array_pop_in_world(world_index, set, 1);
    world(world_index).records[id] = Island::default();
    world(world_index).free.push(id);
}
#[export_name = "islandField"]
pub unsafe extern "C" fn field(id: usize, field: usize) -> i32 {
    unsafe { field_in_world(crate::regions::active(), id, field) }
}

pub unsafe extern "C" fn field_in_world(world_index: usize, id: usize, field: usize) -> i32 {
    let s = record(world_index, id);
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
    unsafe { set_field_in_world(crate::regions::active(), id, field, value) }
}

pub unsafe extern "C" fn set_field_in_world(
    world_index: usize,
    id: usize,
    field: usize,
    value: i32,
) {
    let s = record(world_index, id);
    match field {
        0 => s.set_index = value,
        1 => s.local_index = value,
        3 => s.constraint_remove_count = value,
        _ => unreachable!(),
    }
}
#[export_name = "islandArrayCount"]
pub unsafe extern "C" fn array_count(id: usize, kind: usize) -> usize {
    unsafe { array_count_in_world(crate::regions::active(), id, kind) }
}

pub unsafe extern "C" fn array_count_in_world(world_index: usize, id: usize, kind: usize) -> usize {
    let s = record(world_index, id);
    match kind {
        0 => s.bodies.len(),
        1 => s.contacts.len(),
        2 => s.joints.len(),
        _ => unreachable!(),
    }
}
#[export_name = "islandArrayGet"]
pub unsafe extern "C" fn array_get(id: usize, kind: usize, index: usize, lane: usize) -> i32 {
    unsafe { array_get_in_world(crate::regions::active(), id, kind, index, lane) }
}

pub unsafe extern "C" fn array_get_in_world(
    world_index: usize,
    id: usize,
    kind: usize,
    index: usize,
    lane: usize,
) -> i32 {
    let s = record(world_index, id);
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
    unsafe { add_body_in_world(crate::regions::active(), id, body) }
}

pub unsafe extern "C" fn add_body_in_world(world_index: usize, id: usize, body: i32) {
    let s = record(world_index, id);
    fix(world_index, 0, body, id as i32, s.bodies.len() as i32);
    s.bodies.push(body);
}
#[export_name = "islandRemoveBody"]
pub unsafe extern "C" fn remove_body(id: usize, index: usize) {
    unsafe { remove_body_in_world(crate::regions::active(), id, index) }
}

pub unsafe extern "C" fn remove_body_in_world(world_index: usize, id: usize, index: usize) {
    let s = record(world_index, id);
    let removed = s.bodies.swap_remove(index);
    if index < s.bodies.len() {
        fix(world_index, 0, s.bodies[index], id as i32, index as i32);
    }
    fix(world_index, 0, removed, -1, -1);
}
unsafe fn merge(world_index: usize, a: i32, b: i32) -> usize {
    if a == b || b == -1 {
        return a as usize;
    }
    if a == -1 {
        return b as usize;
    }
    let (big, small) = if record(world_index, a as usize).bodies.len()
        >= record(world_index, b as usize).bodies.len()
    {
        (a as usize, b as usize)
    } else {
        (b as usize, a as usize)
    };
    let bodies = std::mem::take(&mut record(world_index, small).bodies);
    record(world_index, big).bodies.reserve(bodies.len());
    for id in bodies {
        add_body_in_world(world_index, big, id);
    }
    let contacts = std::mem::take(&mut record(world_index, small).contacts);
    record(world_index, big).contacts.reserve(contacts.len());
    for l in contacts {
        add_contact(world_index, big, l);
    }
    let joints = std::mem::take(&mut record(world_index, small).joints);
    record(world_index, big).joints.reserve(joints.len());
    for l in joints {
        add_joint(world_index, big, l);
    }
    record(world_index, big).constraint_remove_count +=
        record(world_index, small).constraint_remove_count;
    destroy_in_world(world_index, small);
    big
}
unsafe fn add_contact(world_index: usize, id: usize, l: Link) {
    let s = record(world_index, id);
    contact_fix(world_index, l.id, id as i32, s.contacts.len() as i32);
    s.contacts.push(l);
}
unsafe fn add_joint(world_index: usize, id: usize, l: Link) {
    let s = record(world_index, id);
    fix(world_index, 1, l.id, id as i32, s.joints.len() as i32);
    s.joints.push(l);
}
#[export_name = "islandLinkContact"]
pub unsafe extern "C" fn link_contact(contact: i32, a: i32, b: i32) {
    unsafe { link_contact_in_world(crate::regions::active(), contact, a, b) }
}

pub unsafe extern "C" fn link_contact_in_world(world_index: usize, contact: i32, a: i32, b: i32) {
    let id = merge(world_index, a, b);
    let d = manifolds::dir_col(world_index);
    let base = contact as usize * DIR_STRIDE;
    add_contact(
        world_index,
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
    unsafe { unlink_contact_in_world(crate::regions::active(), contact) }
}

pub unsafe extern "C" fn unlink_contact_in_world(world_index: usize, contact: i32) {
    let d = manifolds::dir_col(world_index);
    let base = contact as usize * DIR_STRIDE;
    let id = d.get(base + DIR_ISLAND_ID) as usize;
    let index = d.get(base + DIR_ISLAND_INDEX) as usize;
    let s = record(world_index, id);
    s.contacts.swap_remove(index);
    if index < s.contacts.len() {
        contact_fix(world_index, s.contacts[index].id, id as i32, index as i32);
    }
    contact_fix(world_index, contact, -1, -1);
    s.constraint_remove_count += 1;
}
#[export_name = "islandLinkJoint"]
pub unsafe extern "C" fn link_joint(joint: i32, body_a: i32, body_b: i32, a: i32, b: i32) {
    unsafe { link_joint_in_world(crate::regions::active(), joint, body_a, body_b, a, b) }
}

pub unsafe extern "C" fn link_joint_in_world(
    world_index: usize,
    joint: i32,
    body_a: i32,
    body_b: i32,
    a: i32,
    b: i32,
) {
    let id = merge(world_index, a, b);
    add_joint(
        world_index,
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
    unsafe { unlink_joint_in_world(crate::regions::active(), joint, id, index) }
}

pub unsafe extern "C" fn unlink_joint_in_world(
    world_index: usize,
    joint: i32,
    id: i32,
    index: usize,
) {
    if id == -1 {
        return;
    }
    let s = record(world_index, id as usize);
    s.joints.swap_remove(index);
    if index < s.joints.len() {
        fix(world_index, 1, s.joints[index].id, id, index as i32);
    }
    fix(world_index, 1, joint, -1, -1);
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
struct SplitScratch {
    parents: Vec<usize>,
    ranks: Vec<usize>,
    contact_counts: Vec<usize>,
    joint_counts: Vec<usize>,
    root_map: Vec<usize>,
    body_counts: Vec<usize>,
    component_contacts: Vec<usize>,
    component_joints: Vec<usize>,
    ids: Vec<usize>,
}
impl SplitScratch {
    const EMPTY: Self = Self {
        parents: Vec::new(),
        ranks: Vec::new(),
        contact_counts: Vec::new(),
        joint_counts: Vec::new(),
        root_map: Vec::new(),
        body_counts: Vec::new(),
        component_contacts: Vec::new(),
        component_joints: Vec::new(),
        ids: Vec::new(),
    };
}
// Like Box3D's task contexts/arena, scratch is retained by the worker, not allocated by a split.
static mut SPLIT_SCRATCH: [SplitScratch; crate::solve::MAX_THREADS] =
    [const { SplitScratch::EMPTY }; crate::solve::MAX_THREADS];
pub unsafe fn prepare_split(world_index: usize, base: usize, worker: usize) {
    let s = &mut SPLIT_SCRATCH[worker];
    let n = record(world_index, base).bodies.len();
    for v in [
        &mut s.parents,
        &mut s.ranks,
        &mut s.contact_counts,
        &mut s.joint_counts,
        &mut s.root_map,
        &mut s.body_counts,
        &mut s.component_contacts,
        &mut s.component_joints,
    ] {
        v.resize(n, 0);
    }
    s.ids.clear();
    s.ids.reserve(n);
}
#[export_name = "islandSplit"]
pub unsafe extern "C" fn split(base: usize) {
    unsafe { split_in_world(crate::regions::active(), base) }
}

pub unsafe extern "C" fn split_in_world(world_index: usize, base: usize) {
    prepare_split(world_index, base, 0);
    split_task(world_index, base, 0);
}
pub unsafe fn split_task(world_index: usize, base: usize, worker: usize) {
    let SplitScratch {
        parents,
        ranks,
        contact_counts,
        joint_counts,
        root_map,
        body_counts,
        component_contacts,
        component_joints,
        ids,
    } = &mut SPLIT_SCRATCH[worker];
    let n = record(world_index, base).bodies.len();
    for (i, p) in parents.iter_mut().enumerate() {
        *p = i;
    }
    ranks.fill(0);
    contact_counts.fill(0);
    joint_counts.fill(0);
    for kind in 0..2 {
        let links = if kind == 0 {
            &record(world_index, base).contacts
        } else {
            &record(world_index, base).joints
        };
        for l in links {
            let a = crate::bodies::record(world_index, l.body_a as usize).island_index;
            let b = crate::bodies::record(world_index, l.body_b as usize).island_index;
            if a != -1 && b != -1 {
                union(
                    parents,
                    ranks,
                    a as usize,
                    b as usize,
                    contact_counts,
                    joint_counts,
                );
            }
            let root = find_parent(parents, if a != -1 { a } else { b } as usize);
            if kind == 0 {
                contact_counts[root] += 1;
            } else {
                joint_counts[root] += 1;
            }
        }
    }
    let mut components = 0;
    for i in 0..n {
        parents[i] = find_parent(parents, i);
        if parents[i] == i {
            components += 1;
        }
    }
    if components == 1 {
        record(world_index, base).constraint_remove_count = 0;
        return;
    }
    root_map.fill(usize::MAX);
    body_counts.fill(0);
    component_contacts.fill(0);
    component_joints.fill(0);
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
    for i in 0..island_count {
        let id = create_in_world(world_index, 2);
        ids.push(id);
        let s = record(world_index, id);
        s.bodies.reserve(body_counts[i]);
        s.contacts.reserve(component_contacts[i]);
        s.joints.reserve(component_joints[i]);
    }
    for i in 0..n {
        let body = record(world_index, base).bodies[i];
        add_body_in_world(world_index, ids[root_map[parents[i]]], body);
    }
    for i in 0..record(world_index, base).contacts.len() {
        let l = record(world_index, base).contacts[i];
        let a = crate::bodies::record(world_index, l.body_a as usize).island_id;
        let b = crate::bodies::record(world_index, l.body_b as usize).island_id;
        add_contact(world_index, if a != -1 { a } else { b } as usize, l);
    }
    for i in 0..record(world_index, base).joints.len() {
        let l = record(world_index, base).joints[i];
        let a = crate::bodies::record(world_index, l.body_a as usize).island_id;
        let b = crate::bodies::record(world_index, l.body_b as usize).island_id;
        add_joint(world_index, if a != -1 { a } else { b } as usize, l);
    }
    destroy_in_world(world_index, base);
}
pub unsafe fn reset(id: usize) {
    WORLDS[id] = Islands {
        split_island_id: -1,
        records: Vec::new(),
        free: Vec::new(),
    };
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    regions::write_word(out, w.split_island_id as usize);
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
    w.split_island_id = regions::read_word(input) as i32;
    let n = regions::read_word(input);
    for _ in 0..n {
        let mut s = Island {
            set_index: regions::read_word(input) as i32,
            local_index: regions::read_word(input) as i32,
            island_id: regions::read_word(input) as i32,
            constraint_remove_count: regions::read_word(input) as i32,
            ..Island::default()
        };
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
