//! Box3D constraint_graph.c: persistent color occupancy and contact lists.
use crate::regions::MAX_WORLDS;
use crate::{bodies, joints, manifold_abi::*, manifolds, regions};
const COLORS: usize = 24;
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
unsafe fn colors() -> &'static mut Vec<GraphColor> {
    let g = &mut GRAPHS[regions::active()];
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
    unsafe {
        for c in colors().iter_mut().take(OVERFLOW) {
            c.body_set.resize(capacity.max(8).div_ceil(64), 0);
        }
    }
}
#[export_name = "graphCreateJoint"]
pub extern "C" fn create_joint(a: usize, b: usize) -> usize {
    let color = joint_color(a, b);
    joints::append(color);
    color
}
// The joint's identity stays in the host until body/joint records move; the result carries its fix-up.
static mut JOINT_RESULT: [u32; 3] = [0; 3];
#[export_name = "graphAddJoint"]
pub extern "C" fn add_joint(source: usize, index: usize, a: usize, b: usize) -> usize {
    let color = joint_color(a, b);
    let destination = joints::count(color);
    let moved = joints::move_record(source, index, color);
    unsafe {
        JOINT_RESULT = [color as u32, destination as u32, moved];
        core::ptr::addr_of!(JOINT_RESULT) as usize
    }
}
#[export_name = "graphBodyBit"]
pub extern "C" fn body_bit(color: usize, id: usize) -> bool {
    unsafe { bit(&colors()[color], id) }
}
#[export_name = "graphAssignColor"]
pub extern "C" fn assign(a: usize, b: usize, ta: u32, tb: u32) -> usize {
    unsafe {
        let g = colors();
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
    if color == OVERFLOW {
        return;
    }
    unsafe {
        let c = &mut colors()[color];
        for id in [a, b] {
            if let Some(w) = c.body_set.get_mut(id / 64) {
                *w &= !(1 << (id % 64));
            }
        }
    }
}
#[export_name = "graphContactCount"]
pub extern "C" fn count(color: usize, scalar: bool) -> usize {
    unsafe {
        let c = &colors()[color];
        if scalar {
            c.contacts.len()
        } else {
            c.convex_contacts.len()
        }
    }
}
#[export_name = "graphContactPtr"]
pub extern "C" fn pointer(color: usize, scalar: bool) -> usize {
    unsafe {
        let c = &colors()[color];
        if scalar {
            c.contacts.as_ptr() as usize
        } else {
            c.convex_contacts.as_ptr() as usize
        }
    }
}
#[export_name = "graphAddContact"]
pub extern "C" fn add_contact(id: usize, index_a: u32, index_b: u32) {
    unsafe {
        let d = manifolds::dir_col();
        let o = id * DIR_STRIDE;
        let a = d.get(o + DIR_EDGE_A) as usize;
        let b = d.get(o + DIR_EDGE_B) as usize;
        let ta = bodies::get_type(regions::active(), a);
        let tb = bodies::get_type(regions::active(), b);
        let color = assign(a, b, ta, tb);
        let scalar = d.get(o + 6) & 0x00400000 != 0 || color == OVERFLOW;
        d.set(o + DIR_COLOR_INDEX, color as u32);
        d.set(o + DIR_LOCAL_INDEX, count(color, scalar) as u32);
        d.set(o + 9, if ta == 0 { u32::MAX } else { index_a });
        d.set(o + 10, if tb == 0 { u32::MAX } else { index_b });
        let c = &mut colors()[color];
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
    clear(color, a, b);
    unsafe {
        let c = &mut colors()[color];
        let moved = if mesh || color == OVERFLOW {
            c.contacts.swap_remove(index);
            c.contacts.get(index).map(|s| s.contact_id)
        } else {
            c.convex_contacts.swap_remove(index);
            c.convex_contacts.get(index).copied()
        };
        if let Some(id) = moved {
            manifolds::dir_col().set(id as usize * DIR_STRIDE + DIR_LOCAL_INDEX, index as u32);
        }
    }
}
fn joint_color(a: usize, b: usize) -> usize {
    unsafe {
        assign(
            a,
            b,
            bodies::get_type(regions::active(), a),
            bodies::get_type(regions::active(), b),
        )
    }
}
#[export_name = "graphRemoveJoint"]
pub extern "C" fn remove_joint(a: usize, b: usize, color: usize, index: usize) -> u32 {
    clear(color, a, b);
    joints::remove(color, index)
}
const LAYOUT_STRIDE: usize = 5;
const LAYOUT_HEADER: usize = 10;
static mut SOLVE_LAYOUT: [[u32; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS] =
    [[0; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS];
unsafe fn extent(id: u32) -> (u32, u32) {
    let d = manifolds::dir_col();
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
    unsafe {
        let result = &mut SOLVE_LAYOUT[regions::active()];
        let g = colors();
        let (mut contacts, mut manifolds, mut points, mut wide, mut active) = (0, 0, 0, 0, 0);
        for (color, c) in g.iter().enumerate().take(OVERFLOW) {
            if c.convex_contacts.len() + c.contacts.len() + joints::count(color) == 0 {
                continue;
            }
            let o = LAYOUT_HEADER + active * LAYOUT_STRIDE;
            let n = c.convex_contacts.len() as u32;
            let nw = n.div_ceil(4);
            result[o..o + LAYOUT_STRIDE].copy_from_slice(&[color as u32, wide, nw, 0, 0]);
            for &id in &c.convex_contacts {
                let (m, p) = extent(id);
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
                let (m, p) = extent(s.contact_id);
                manifolds += m;
                points += p;
            }
            contacts += c.contacts.len() as u32;
        }
        let overflow_start = contacts;
        for s in &g[OVERFLOW].contacts {
            let (m, p) = extent(s.contact_id);
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
    unsafe {
        let slot = crate::arena::scalar_columns().slot;
        let (_, _, meta) = crate::arena::wide_columns();
        let dir = manifolds::dir_col();
        let g = colors();
        let (mut gm, mut gp, mut cursor, mut wide) = (0, 0, 0, 0);
        for c in g.iter().take(OVERFLOW) {
            let n = c.convex_contacts.len();
            for (j, &id) in c.convex_contacts.iter().enumerate() {
                dir.set(id as usize * DIR_STRIDE + 11, 0);
                meta.set(
                    (wide + j / 4) * crate::contact_wide::WIDE_META_STRIDE + j % 4,
                    id,
                );
                let (m, p) = extent(id);
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
                let (m, p) = extent(s.contact_id);
                gm += m;
                gp += p;
                cursor += 1;
            }
        }
    }
}
static mut WAKE: [Vec<u32>; MAX_WORLDS] = [const { Vec::new() }; MAX_WORLDS];
#[export_name = "graphWakeBuffer"]
pub extern "C" fn wake_buffer(contacts: usize, joints: usize) -> usize {
    unsafe {
        let buffer = &mut WAKE[regions::active()];
        buffer.resize(3 * (contacts + joints), 0);
        buffer.as_mut_ptr() as usize
    }
}
#[export_name = "graphWake"]
pub extern "C" fn wake(source: usize, contacts: usize, count: usize) {
    unsafe {
        let buffer = &mut WAKE[regions::active()];
        for i in 0..contacts {
            let o = 3 * i;
            let id = buffer[o] as usize;
            add_contact(id, buffer[o + 1], buffer[o + 2]);
            manifolds::dir_col().set(id * DIR_STRIDE + DIR_SET_INDEX, 2);
        }
        for i in 0..count {
            let o = 3 * (contacts + i);
            let index = i.min(count - 1 - i);
            let color = joint_color(buffer[o + 1] as usize, buffer[o + 2] as usize);
            let destination = joints::count(color);
            joints::move_record(source, index, color);
            buffer[o + 1] = color as u32;
            buffer[o + 2] = destination as u32;
        }
    }
}
pub unsafe fn reset(id: usize) {
    GRAPHS[id] = Vec::new();
    WAKE[id] = Vec::new();
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
