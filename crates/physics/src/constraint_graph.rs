//! Box3D constraint_graph.c: persistent color occupancy and contact lists.
use crate::regions::MAX_WORLDS;
use crate::{bodies, joints, manifold_abi::*, manifolds, regions};
pub(crate) const COLORS: usize = 24;
const OVERFLOW: usize = COLORS - 1;
const DYNAMIC: usize = COLORS - 4;
#[repr(C)]
#[derive(Clone, Copy)]
struct ContactSpec {
    contact_id: u32,
    manifold_start: u16,
    manifold_count: u16,
}
#[derive(Default)]
struct GraphColor {
    body_set: Vec<u64>,
    convex_contacts: Vec<u32>,
    contacts: Vec<ContactSpec>,
}
static mut GRAPHS: [Vec<GraphColor>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
unsafe fn colors(world_index: usize) -> &'static mut Vec<GraphColor> {
    let g = &mut GRAPHS[world_index];
    g.resize_with(COLORS, GraphColor::default);
    g
}
fn bit(c: &GraphColor, id: usize) -> bool {
    c.body_set
        .get(id / 64)
        .is_some_and(|w| w & (1 << (id % 64)) != 0)
}
fn set(c: &mut GraphColor, id: usize) {
    c.body_set.resize(c.body_set.len().max(id / 64 + 1), 0);
    c.body_set[id / 64] |= 1 << (id % 64);
}
#[export_name = "graphCreate"]
pub extern "C" fn create(capacity: usize) {
    create_in_world(crate::regions::active(), capacity)
}

pub extern "C" fn create_in_world(world_index: usize, capacity: usize) {
    unsafe {
        for c in colors(world_index).iter_mut().take(OVERFLOW) {
            c.body_set.resize(capacity.max(8).div_ceil(64), 0);
        }
    }
}
pub extern "C" fn create_joint(world_index: usize, a: usize, b: usize) -> usize {
    let color = joint_color(world_index, a, b);
    joints::append_in_world(world_index, color);
    color
}
pub fn add_joint(world_index: usize, source: usize, index: usize, a: usize, b: usize) {
    let color = joint_color(world_index, a, b);
    joints::move_record_in_world(world_index, source, index, color);
}
#[export_name = "graphBodyBit"]
pub extern "C" fn body_bit(color: usize, id: usize) -> bool {
    body_bit_in_world(crate::regions::active(), color, id)
}

pub extern "C" fn body_bit_in_world(world_index: usize, color: usize, id: usize) -> bool {
    unsafe { bit(&colors(world_index)[color], id) }
}
#[export_name = "graphAssignColor"]
pub extern "C" fn assign(a: usize, b: usize, ta: u32, tb: u32) -> usize {
    assign_in_world(crate::regions::active(), a, b, ta, tb)
}

pub extern "C" fn assign_in_world(
    world_index: usize,
    a: usize,
    b: usize,
    ta: u32,
    tb: u32,
) -> usize {
    unsafe {
        let g = colors(world_index);
        if ta == 2 && tb == 2 {
            for (i, c) in g.iter_mut().enumerate().take(DYNAMIC) {
                if !bit(c, a) && !bit(c, b) {
                    set(c, a);
                    set(c, b);
                    return i;
                }
            }
        } else if ta == 2 || tb == 2 {
            let id = if ta == 2 { a } else { b };
            for i in (1..OVERFLOW).rev() {
                if !bit(&g[i], id) {
                    set(&mut g[i], id);
                    return i;
                }
            }
        }
        OVERFLOW
    }
}
#[export_name = "graphClearBodies"]
pub extern "C" fn clear(color: usize, a: usize, b: usize) {
    clear_in_world(crate::regions::active(), color, a, b)
}

pub extern "C" fn clear_in_world(world_index: usize, color: usize, a: usize, b: usize) {
    if color == OVERFLOW {
        return;
    }
    unsafe {
        let c = &mut colors(world_index)[color];
        for id in [a, b] {
            if let Some(w) = c.body_set.get_mut(id / 64) {
                *w &= !(1 << (id % 64));
            }
        }
    }
}
#[export_name = "graphContactCount"]
pub extern "C" fn count(color: usize, scalar: bool) -> usize {
    count_in_world(crate::regions::active(), color, scalar)
}

pub extern "C" fn count_in_world(world_index: usize, color: usize, scalar: bool) -> usize {
    unsafe {
        let c = &colors(world_index)[color];
        if scalar {
            c.contacts.len()
        } else {
            c.convex_contacts.len()
        }
    }
}
#[export_name = "graphContactPtr"]
pub extern "C" fn pointer(color: usize, scalar: bool) -> usize {
    pointer_in_world(crate::regions::active(), color, scalar)
}

pub extern "C" fn pointer_in_world(world_index: usize, color: usize, scalar: bool) -> usize {
    unsafe {
        let c = &colors(world_index)[color];
        if scalar {
            c.contacts.as_ptr() as usize
        } else {
            c.convex_contacts.as_ptr() as usize
        }
    }
}
#[export_name = "graphAddContact"]
pub extern "C" fn add_contact(id: usize, index_a: u32, index_b: u32) {
    add_contact_in_world(crate::regions::active(), id, index_a, index_b)
}

pub extern "C" fn add_contact_in_world(world_index: usize, id: usize, index_a: u32, index_b: u32) {
    unsafe {
        let d = manifolds::dir_col(world_index);
        let o = id * DIR_STRIDE;
        let a = d.get(o + DIR_EDGE_A) as usize;
        let b = d.get(o + DIR_EDGE_B) as usize;
        let ta = bodies::get_type(world_index, a);
        let tb = bodies::get_type(world_index, b);
        let color = assign_in_world(world_index, a, b, ta, tb);
        let scalar = d.get(o + 6) & 0x00400000 != 0 || color == OVERFLOW;
        d.set(o + DIR_COLOR_INDEX, color as u32);
        d.set(
            o + DIR_LOCAL_INDEX,
            count_in_world(world_index, color, scalar) as u32,
        );
        d.set(o + 9, if ta == 0 { u32::MAX } else { index_a });
        d.set(o + 10, if tb == 0 { u32::MAX } else { index_b });
        let c = &mut colors(world_index)[color];
        if scalar {
            c.contacts.push(ContactSpec {
                contact_id: id as u32,
                manifold_start: 0,
                manifold_count: d.get(o + 7) as u16,
            });
        } else {
            c.convex_contacts.push(id as u32);
        }
    }
}
#[export_name = "graphRemoveContact"]
pub extern "C" fn remove_contact(a: usize, b: usize, color: usize, index: usize, mesh: bool) {
    remove_contact_in_world(crate::regions::active(), a, b, color, index, mesh)
}

pub extern "C" fn remove_contact_in_world(
    world_index: usize,
    a: usize,
    b: usize,
    color: usize,
    index: usize,
    mesh: bool,
) {
    clear_in_world(world_index, color, a, b);
    unsafe {
        let c = &mut colors(world_index)[color];
        let moved = if mesh || color == OVERFLOW {
            c.contacts.swap_remove(index);
            c.contacts.get(index).map(|s| s.contact_id)
        } else {
            c.convex_contacts.swap_remove(index);
            c.convex_contacts.get(index).copied()
        };
        if let Some(id) = moved {
            manifolds::dir_col(world_index)
                .set(id as usize * DIR_STRIDE + DIR_LOCAL_INDEX, index as u32);
        }
    }
}
fn joint_color(world_index: usize, a: usize, b: usize) -> usize {
    unsafe {
        assign_in_world(
            world_index,
            a,
            b,
            bodies::get_type(world_index, a),
            bodies::get_type(world_index, b),
        )
    }
}
pub extern "C" fn remove_joint(
    world_index: usize,
    a: usize,
    b: usize,
    color: usize,
    index: usize,
) -> u32 {
    clear_in_world(world_index, color, a, b);
    joints::remove_in_world(world_index, color, index)
}
const LAYOUT_STRIDE: usize = 5;
const LAYOUT_HEADER: usize = 10;
static mut SOLVE_LAYOUT: [[u32; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS] =
    [[0; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS];
unsafe fn extent(world_index: usize, id: u32) -> (u32, u32) {
    let d = manifolds::dir_col(world_index);
    let o = id as usize * DIR_STRIDE;
    let n = d.get(o + 7);
    let p = d.get(o + DIR_MANIFOLD_BASE) as *const u32;
    let mut points = 0;
    for i in 0..n as usize {
        points += *p.add(i * MANIFOLD_STRIDE + M_POINT_COUNT);
    }
    (n, points)
}
#[export_name = "graphComputeLayout"]
pub extern "C" fn compute_layout() -> usize {
    compute_layout_in_world(crate::regions::active())
}

pub extern "C" fn compute_layout_in_world(world_index: usize) -> usize {
    unsafe {
        let result = &mut SOLVE_LAYOUT[world_index];
        let g = colors(world_index);
        let (mut contacts, mut manifolds, mut points, mut wide, mut active) = (0, 0, 0, 0, 0);
        for (color, c) in g.iter().enumerate().take(OVERFLOW) {
            if c.convex_contacts.len()
                + c.contacts.len()
                + joints::count_in_world(world_index, color)
                == 0
            {
                continue;
            }
            let o = LAYOUT_HEADER + active * LAYOUT_STRIDE;
            let n = c.convex_contacts.len() as u32;
            let nw = n.div_ceil(4);
            result[o..o + LAYOUT_STRIDE].copy_from_slice(&[color as u32, wide, nw, 0, 0]);
            for &id in &c.convex_contacts {
                let (m, p) = extent(world_index, id);
                manifolds += m;
                points += p;
            }
            contacts += n;
            wide += nw;
            active += 1;
        }
        let mesh_start = contacts;
        for a in 0..active {
            let o = LAYOUT_HEADER + a * LAYOUT_STRIDE;
            let c = &g[result[o] as usize];
            result[o + 3] = contacts;
            result[o + 4] = c.contacts.len() as u32;
            for s in &c.contacts {
                let (m, p) = extent(world_index, s.contact_id);
                manifolds += m;
                points += p;
            }
            contacts += c.contacts.len() as u32;
        }
        let overflow_start = contacts;
        for s in &g[OVERFLOW].contacts {
            let (m, p) = extent(world_index, s.contact_id);
            manifolds += m;
            points += p;
        }
        let overflow_count = g[OVERFLOW].contacts.len() as u32;
        contacts += overflow_count;
        result[..LAYOUT_HEADER].copy_from_slice(&[
            contacts,
            manifolds,
            points,
            wide,
            mesh_start,
            overflow_start - mesh_start,
            wide,
            overflow_start,
            overflow_count,
            active as u32,
        ]);
        result.as_ptr() as usize
    }
}
#[export_name = "graphWriteSlots"]
pub extern "C" fn write_slots() {
    write_slots_in_world(crate::regions::active())
}

pub extern "C" fn write_slots_in_world(world_index: usize) {
    unsafe {
        let slot = crate::arena::scalar_columns(world_index).slot;
        let (_, _, meta) = crate::arena::wide_columns();
        let dir = manifolds::dir_col(world_index);
        let g = colors(world_index);
        let (mut gm, mut gp, mut cursor, mut wide) = (0, 0, 0, 0);
        for c in g.iter().take(OVERFLOW) {
            let n = c.convex_contacts.len();
            for (j, &id) in c.convex_contacts.iter().enumerate() {
                dir.set(id as usize * DIR_STRIDE + 11, 0);
                meta.set(
                    (wide + j / 4) * crate::contact_wide::WIDE_META_STRIDE + j % 4,
                    id,
                );
                let (m, p) = extent(world_index, id);
                gm += m;
                gp += p;
                cursor += 1;
            }
            let nw = n.div_ceil(4);
            for r in 0..nw {
                meta.set(
                    (wide + r) * crate::contact_wide::WIDE_META_STRIDE + 4,
                    (n - r * 4).min(4) as u32,
                );
            }
            wide += nw;
        }
        for c in g.iter() {
            for s in &c.contacts {
                dir.set(s.contact_id as usize * DIR_STRIDE + 11, 0);
                let o = cursor * SLOT_STRIDE;
                slot.set(o, s.contact_id);
                slot.set(o + 1, gm);
                slot.set(o + 2, gp);
                let (m, p) = extent(world_index, s.contact_id);
                gm += m;
                gp += p;
                cursor += 1;
            }
        }
    }
}
pub unsafe fn reset(id: usize) {
    GRAPHS[id] = Vec::new();
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, GRAPHS[id].len());
    for c in &GRAPHS[id] {
        regions::write_word(out, c.body_set.len());
        for w in &c.body_set {
            out.extend_from_slice(&w.to_le_bytes());
        }
        regions::write_word(out, c.convex_contacts.len());
        for w in &c.convex_contacts {
            out.extend_from_slice(&w.to_le_bytes());
        }
        regions::write_word(out, c.contacts.len());
        for s in &c.contacts {
            out.extend_from_slice(&s.contact_id.to_le_bytes());
            out.extend_from_slice(&s.manifold_start.to_le_bytes());
            out.extend_from_slice(&s.manifold_count.to_le_bytes());
        }
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let n = regions::read_word(input);
    for _ in 0..n {
        let mut c = GraphColor::default();
        let n = regions::read_word(input);
        for _ in 0..n {
            c.body_set
                .push(u64::from_le_bytes(input[..8].try_into().unwrap()));
            *input = &input[8..];
        }
        let n = regions::read_word(input);
        for _ in 0..n {
            c.convex_contacts
                .push(u32::from_le_bytes(input[..4].try_into().unwrap()));
            *input = &input[4..];
        }
        let n = regions::read_word(input);
        for _ in 0..n {
            c.contacts.push(ContactSpec {
                contact_id: u32::from_le_bytes(input[..4].try_into().unwrap()),
                manifold_start: u16::from_le_bytes(input[4..6].try_into().unwrap()),
                manifold_count: u16::from_le_bytes(input[6..8].try_into().unwrap()),
            });
            *input = &input[8..];
        }
        GRAPHS[id].push(c);
    }
}
