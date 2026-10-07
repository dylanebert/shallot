//! Box3D constraint_graph.c: persistent color occupancy and contact lists.
use crate::regions::MAX_WORLDS;
use crate::{bodies, joints, manifold_abi::*, manifolds, regions};
pub(crate) const COLORS: usize = 24;
const OVERFLOW: usize = COLORS - 1;
const DYNAMIC: usize = COLORS - 4;
use crate::contact_spans::{ContactPrepareSpan, ContactSpec, WidePrepareSpan};
const EMPTY_CONTACT_SPAN: ContactPrepareSpan = ContactPrepareSpan {
    start: i32::MAX,
    count: 0,
    contacts: core::ptr::null(),
};
const EMPTY_WIDE_SPAN: WidePrepareSpan = WidePrepareSpan {
    start: i32::MAX,
    count: 0,
    contacts: core::ptr::null(),
};
static mut CONTACT_SPANS: [[ContactPrepareSpan; COLORS + 1]; MAX_WORLDS] =
    [[EMPTY_CONTACT_SPAN; COLORS + 1]; MAX_WORLDS];
static mut WIDE_SPANS: [[WidePrepareSpan; COLORS + 1]; MAX_WORLDS] =
    [[EMPTY_WIDE_SPAN; COLORS + 1]; MAX_WORLDS];
static mut OVERFLOW_SPANS: [[ContactPrepareSpan; 2]; MAX_WORLDS] =
    [[EMPTY_CONTACT_SPAN; 2]; MAX_WORLDS];
static mut OVERFLOW_MANIFOLDS: [usize; MAX_WORLDS] = [0; MAX_WORLDS];

pub(crate) unsafe fn overflow_spans(world: usize) -> crate::col::Col<'static, ContactPrepareSpan> {
    crate::col::Col::new(OVERFLOW_SPANS[world].as_mut_ptr(), 2)
}
pub(crate) unsafe fn overflow_manifold_count(world: usize) -> usize {
    OVERFLOW_MANIFOLDS[world]
}
pub(crate) unsafe fn overflow_contact_count(world: usize) -> usize {
    GRAPHS[world][OVERFLOW].contacts.len()
}

// The graph's backing arrays stay put during collide; each task owns one contact spec.
pub(crate) unsafe fn update_manifold_count(world: usize, color: usize, index: usize, count: u16) {
    let contacts = (&*GRAPHS[world].as_ptr().add(color))
        .contacts
        .as_ptr()
        .cast_mut();
    core::ptr::addr_of_mut!((*contacts.add(index)).manifold_count).write(count);
}

pub(crate) unsafe fn prepare_spans(
    world: usize,
) -> (
    crate::col::Col<'static, ContactPrepareSpan>,
    crate::col::Col<'static, WidePrepareSpan>,
) {
    (
        crate::col::Col::new(CONTACT_SPANS[world].as_mut_ptr(), COLORS + 1),
        crate::col::Col::new(WIDE_SPANS[world].as_mut_ptr(), COLORS + 1),
    )
}

#[repr(C)]
struct GraphColor {
    body_set: Vec<u64>,
    joint_sims: joints::JointArray,
    convex_contacts: Vec<u32>,
    contacts: Vec<ContactSpec>,
}
impl GraphColor {
    const EMPTY: Self = Self {
        body_set: Vec::new(),
        joint_sims: joints::JointArray::EMPTY,
        convex_contacts: Vec::new(),
        contacts: Vec::new(),
    };
}
static mut GRAPHS: [[GraphColor; COLORS]; MAX_WORLDS] =
    [const { [const { GraphColor::EMPTY }; COLORS] }; MAX_WORLDS];
unsafe fn colors(world_index: usize) -> &'static mut [GraphColor; COLORS] {
    &mut GRAPHS[world_index]
}
pub(crate) unsafe fn joint_array(world: usize, color: usize) -> &'static mut joints::JointArray {
    &mut GRAPHS[world][color].joint_sims
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
    unsafe { joints::copy_record_in_world(world_index, source, index, color) };
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
        let scalar = d.get(o + DIR_FLAGS) & 0x00400000 != 0 || color == OVERFLOW;
        d.set(o + DIR_COLOR_INDEX, color as u32);
        d.set(
            o + DIR_LOCAL_INDEX,
            count_in_world(world_index, color, scalar) as u32,
        );
        d.set(o + DIR_INDEX_A, if ta == 0 { u32::MAX } else { index_a });
        d.set(o + DIR_INDEX_B, if tb == 0 { u32::MAX } else { index_b });
        let c = &mut colors(world_index)[color];
        if scalar {
            c.contacts.push(ContactSpec {
                contact_id: id as i32,
                manifold_start: 0,
                manifold_count: d.get(o + DIR_MANIFOLD_COUNT) as u16,
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
            c.contacts.get(index).map(|s| s.contact_id as u32)
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
pub(crate) unsafe fn solver_colors(
    world: usize,
    spans: &mut [crate::stages::ColorSpan; COLORS],
    keys: &mut [usize; COLORS],
) -> (usize, usize, usize, usize) {
    let (mut active, mut wide, mut contacts, mut joints) = (0, 0, 0, 0);
    for (key, color) in GRAPHS[world].iter().enumerate().take(OVERFLOW) {
        let joint_count = color.joint_sims.count;
        let wide_count = color.convex_contacts.len().div_ceil(4);
        let contact_count = color.contacts.len();
        if wide_count + contact_count + joint_count == 0 {
            continue;
        }
        keys[active] = key;
        spans[active] = crate::stages::ColorSpan {
            color: active as u8,
            wide_start: wide,
            wide_count,
            mesh_start: contacts,
            mesh_count: contact_count,
            joint_start: 0,
            joint_count,
        };
        wide += wide_count;
        contacts += contact_count;
        joints += joint_count;
        active += 1;
    }
    (active, wide, contacts, joints)
}

const LAYOUT_STRIDE: usize = 5;
const LAYOUT_HEADER: usize = 10;
static mut SOLVE_LAYOUT: [[u32; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS] =
    [[0; LAYOUT_HEADER + OVERFLOW * LAYOUT_STRIDE]; MAX_WORLDS];
#[export_name = "graphComputeLayout"]
pub extern "C" fn compute_layout() -> usize {
    compute_layout_in_world(crate::regions::active())
}

pub extern "C" fn compute_layout_in_world(world_index: usize) -> usize {
    unsafe {
        let result = &mut SOLVE_LAYOUT[world_index];
        let g = colors(world_index);
        CONTACT_SPANS[world_index].fill(EMPTY_CONTACT_SPAN);
        WIDE_SPANS[world_index].fill(EMPTY_WIDE_SPAN);
        let (mut contacts, mut manifolds, mut wide, mut active) = (0, 0, 0, 0);
        for (color, c) in g.iter_mut().enumerate().take(OVERFLOW) {
            let n = c.convex_contacts.len() as i32;
            let count = c.contacts.len() as i32;
            if n + count + joints::count_in_world(world_index, color) as i32 == 0 {
                continue;
            }
            let nw = if n == 0 { 0 } else { (n - 1) / 4 + 1 };
            let o = LAYOUT_HEADER + active * LAYOUT_STRIDE;
            result[o..o + LAYOUT_STRIDE].copy_from_slice(&[
                color as u32,
                wide as u32,
                nw as u32,
                contacts as u32,
                count as u32,
            ]);
            WIDE_SPANS[world_index][active] = WidePrepareSpan {
                start: wide,
                count: n,
                contacts: c.convex_contacts.as_ptr(),
            };
            CONTACT_SPANS[world_index][active] = ContactPrepareSpan {
                start: contacts,
                count,
                contacts: c.contacts.as_ptr(),
            };
            // The scalar spec already carries its manifold count from collide.
            for spec in &mut c.contacts {
                spec.manifold_start = manifolds;
                manifolds += spec.manifold_count as i32;
            }
            contacts += count;
            wide += nw;
            active += 1;
        }
        CONTACT_SPANS[world_index][active] = ContactPrepareSpan {
            start: contacts,
            ..EMPTY_CONTACT_SPAN
        };
        WIDE_SPANS[world_index][active] = WidePrepareSpan {
            start: wide,
            ..EMPTY_WIDE_SPAN
        };
        let overflow = &mut g[OVERFLOW];
        let mut overflow_manifolds = 0;
        for spec in &mut overflow.contacts {
            spec.manifold_start = overflow_manifolds;
            overflow_manifolds += spec.manifold_count as i32;
        }
        let overflow_count = overflow.contacts.len() as i32;
        OVERFLOW_MANIFOLDS[world_index] = overflow_manifolds as usize;
        OVERFLOW_SPANS[world_index] = [
            ContactPrepareSpan {
                start: 0,
                count: overflow_count,
                contacts: overflow.contacts.as_ptr(),
            },
            ContactPrepareSpan {
                start: overflow_count,
                ..EMPTY_CONTACT_SPAN
            },
        ];
        result[..LAYOUT_HEADER].copy_from_slice(&[
            (contacts + overflow_count) as u32,
            (manifolds + overflow_manifolds) as u32,
            0,
            wide as u32,
            0,
            contacts as u32,
            wide as u32,
            0,
            overflow_count as u32,
            active as u32,
        ]);
        result.as_ptr() as usize
    }
}

pub(crate) unsafe fn initialize_constraints(world: usize) {
    let (wide, _, _) = crate::arena::wide_columns(world);
    for span in WIDE_SPANS[world]
        .iter()
        .take(SOLVE_LAYOUT[world][9] as usize)
    {
        let count = span.count as usize;
        if count % crate::contact_wide::LANES != 0 {
            let tail = span.start as usize + count.div_ceil(crate::contact_wide::LANES) - 1;
            core::ptr::write_bytes(
                wide.ptr().add(tail * crate::contact_wide::WIDE_STRIDE),
                0,
                crate::contact_wide::WIDE_STRIDE,
            );
        }
    }
}
pub unsafe fn reset(id: usize) {
    for color in &mut GRAPHS[id] {
        color.joint_sims.release();
    }
    GRAPHS[id] = [const { GraphColor::EMPTY }; COLORS];
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
    // Joint snapshots are decoded first; restoring contact arrays must not discard their storage.
    for color in &mut GRAPHS[id] {
        color.body_set = Vec::new();
        color.convex_contacts = Vec::new();
        color.contacts = Vec::new();
    }
    let n = regions::read_word(input);
    assert_eq!(n, COLORS);
    for color in 0..n {
        let mut c = GraphColor::EMPTY;
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
                contact_id: i32::from_le_bytes(input[..4].try_into().unwrap()),
                manifold_start: i32::from_le_bytes(input[4..8].try_into().unwrap()),
                manifold_count: u16::from_le_bytes(input[8..10].try_into().unwrap()),
            });
            *input = &input[10..];
        }
        c.joint_sims = GRAPHS[id][color].joint_sims;
        GRAPHS[id][color] = c;
    }
}
