//! Box3D physics_world.c: serial contact transitions and step context.
use crate::{
    bodies, body, constraint_graph, contact_list, events, island, manifold_abi::*, manifolds,
    regions, solver_set,
};

#[export_name = "applyContactTransitions"]
pub unsafe extern "C" fn apply_contact_transitions() {
    let words = manifolds::contact_capacity(regions::active()).div_ceil(32);
    let bits = crate::arena::contact_state_ptr() as *const u32;
    for word in 0..words {
        let mut mask = *bits.add(word);
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            apply_touch(word * 32 + bit);
        }
    }
}
unsafe fn apply_touch(id: usize) {
    let d = manifolds::dir_col();
    let o = id * DIR_STRIDE;
    let flags = d.get(o + 6);
    let world = regions::active();
    if flags & 0x0002_0000 != 0 {
        contact_list::remove(id);
        crate::table::remove_pair(
            d.get(o + DIR_SHAPE_A),
            d.get(o + DIR_SHAPE_B),
            d.get(o + DIR_CHILD_INDEX),
        );
        manifolds::free_manifolds(id);
        manifolds::free_mesh_cache(id);
        if flags & 5 == 5 {
            events::contact_touch(world, id, false);
        }
        crate::body_record::runtime::destroy_contact(world, id);
        return;
    }
    let started = flags & 0x0004_0000 != 0;
    let stopped = flags & 0x0008_0000 != 0;
    if !started && !stopped {
        return;
    }
    if flags & 4 != 0 {
        events::contact_touch(world, id, started);
    }
    let a = d.get(o + DIR_EDGE_A) as usize;
    let b = d.get(o + DIR_EDGE_B) as usize;
    if started {
        d.set(o + 6, (flags & !0x0004_0000) | 1);
        let sa = bodies::record(world, a).set_index;
        let sb = bodies::record(world, b).set_index;
        if sa == 2 && sb >= 3 {
            solver_set::wake(sb as usize);
        } else if sb == 2 && sa >= 3 {
            solver_set::wake(sa as usize);
        }
        island::link_contact(
            id as i32,
            bodies::record(world, a).island_id,
            bodies::record(world, b).island_id,
        );
        let old = d.get(o + DIR_LOCAL_INDEX) as usize;
        constraint_graph::add_contact(
            id,
            bodies::record(world, a).local_index as u32,
            bodies::record(world, b).local_index as u32,
        );
        if solver_set::array_remove(2, 0, old) != -1 {
            let moved = solver_set::array_get(2, 0, old) as usize;
            d.set(moved * DIR_STRIDE + DIR_LOCAL_INDEX, old as u32);
        }
    } else {
        d.set(o + 6, flags & !(0x0008_0000 | 1));
        let color = d.get(o + DIR_COLOR_INDEX) as usize;
        let local = d.get(o + DIR_LOCAL_INDEX) as usize;
        island::unlink_contact(id as i32);
        d.set(o + DIR_COLOR_INDEX, u32::MAX);
        d.set(
            o + DIR_LOCAL_INDEX,
            solver_set::array_push(2, 0, id as i32) as u32,
        );
        constraint_graph::remove_contact(a, b, color, local, flags & 0x0040_0000 != 0);
    }
}

#[export_name = "stepFinalize"]
pub unsafe extern "C" fn finalize(count: usize, dt: f32, enable_sleep: bool) -> bool {
    crate::continuous::consume(regions::active(), count, false);
    let sim2 = bodies::sim2_base() as *const u32;
    let mut bullets = false;
    for i in 0..count {
        let flags = *sim2.add(i * body::SIM2_STRIDE + body::S2_FLAGS);
        bullets |= flags & (body::flags::IS_FAST | crate::continuous::IS_BULLET)
            == (body::flags::IS_FAST | crate::continuous::IS_BULLET);
    }
    let split = crate::body_record::runtime::finish(count, dt, enable_sleep);
    if enable_sleep && split != -1 {
        island::set_split_candidate(split);
    }
    crate::treework::enlarge_pass(count, 0);
    bullets
}

#[export_name = "stepSolveBuild"]
pub unsafe extern "C" fn solve_build(
    threads: usize,
    substeps: usize,
    gx: f32,
    gy: f32,
    gz: f32,
    max_speed: f32,
    contact_speed: f32,
    warm: bool,
    restitution: f32,
    hit: f32,
    continuous: bool,
    sleep: bool,
) {
    let count = solver_set::body_count(2);
    let layout = constraint_graph::compute_layout() as *const u32;
    crate::arena::reserve(
        count,
        *layout as usize,
        *layout.add(1) as usize,
        *layout.add(2) as usize,
        *layout.add(3) as usize,
        *layout.add(9) as usize,
    );
    constraint_graph::write_slots();
    let (spans, colors) = crate::arena::color_span_column();
    let mut total = 0;
    for i in 0..colors {
        let o = 10 + i * 5;
        for j in 0..4 {
            spans.set(i * 6 + j, *layout.add(o + 1 + j));
        }
        let color = *layout.add(o) as usize;
        let n = crate::joints::count(color);
        spans.set(i * 6 + 4, color as u32);
        spans.set(i * 6 + 5, n as u32);
        total += n;
    }
    crate::continuous::roots(
        *crate::broad::tree_state(0) as i32,
        *crate::broad::tree_state(1) as i32,
        *crate::broad::tree_state(2) as i32,
        sleep,
    );
    crate::solve::solve_build(
        threads,
        substeps,
        *layout.add(6) as usize,
        *layout.add(4) as usize,
        *layout.add(5) as usize,
        *layout.add(7) as usize,
        *layout.add(8) as usize,
        total,
        crate::joints::count(23),
        gx,
        gy,
        gz,
        CONTEXT[2],
        CONTEXT[3],
        CONTEXT[0],
        CONTEXT[1],
        max_speed,
        contact_speed,
        CONTEXT[4],
        CONTEXT[5],
        CONTEXT[6],
        CONTEXT[7],
        CONTEXT[8],
        CONTEXT[9],
        if warm { 1.0 } else { 0.0 },
        restitution,
        hit,
        continuous as u32,
    );
}

static mut CONTEXT: [f32; 10] = [0.0; 10];
#[export_name = "stepContext"]
pub unsafe extern "C" fn context(dt: f32, substeps: usize, hertz: f32, damping: f32) -> *const f32 {
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };
    let h = if dt > 0.0 {
        dt / substeps.max(1) as f32
    } else {
        0.0
    };
    let inv_h = substeps.max(1) as f32 * inv_dt;
    let hz = hertz.min(0.125 * inv_h);
    let cs = softness(hz, damping, h);
    let ss = softness(2.0 * hz, 0.5 * damping, h);
    CONTEXT = [
        dt,
        inv_dt,
        h,
        inv_h,
        cs.bias_rate,
        cs.mass_scale,
        cs.impulse_scale,
        ss.bias_rate,
        ss.mass_scale,
        ss.impulse_scale,
    ];
    (&raw const CONTEXT).cast()
}
fn softness(hertz: f32, damping: f32, h: f32) -> crate::contact::Softness {
    if hertz == 0.0 {
        return crate::contact::Softness {
            bias_rate: 0.0,
            mass_scale: 0.0,
            impulse_scale: 0.0,
        };
    }
    let omega = (2.0 * core::f32::consts::PI) * hertz;
    let a1 = 2.0 * damping + h * omega;
    let a2 = (h * omega) * a1;
    let a3 = 1.0 / (1.0 + a2);
    crate::contact::Softness {
        bias_rate: omega / a1,
        mass_scale: a2 * a3,
        impulse_scale: a3,
    }
}
