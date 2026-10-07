//! Box3D physics_world.c: serial contact transitions and step context.
use crate::{
    bodies, body, constraint_graph, contact_list, events, island, manifold_abi::*, manifolds,
    regions, solver_set,
};

#[export_name = "worldDestroyKernel"]
pub unsafe extern "C" fn destroy_world(world: usize) {
    regions::select(world as u32);
    for id in 0..crate::shapes::shape_cap() {
        if crate::shapes::shape_alive(world as u32, id as u32) != 0 {
            crate::shape_lifecycle::release_geometry(world, id);
        }
    }
    assert_eq!(crate::hull_database::count(world), 0);
    for id in 0..solver_set::count() {
        if solver_set::index(id) != -1 {
            solver_set::destroy(id);
        }
    }
    bodies::body_reset_world(world as u32);
    crate::shapes::shape_reset_world(world as u32);
    regions::reset(world as u32);
}
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
#[export_name = "contactDestroyWorld"]
pub unsafe extern "C" fn destroy_contact_world(world: usize, id: usize, wake: bool) {
    regions::select(world as u32);
    destroy_contact(id, wake);
}
#[export_name = "contactLinkWorld"]
pub unsafe extern "C" fn link_contact(world: usize, id: usize) {
    regions::select(world as u32);
    let d = manifolds::dir_col();
    let a = d.get(id * DIR_STRIDE + DIR_EDGE_A) as usize;
    let b = d.get(id * DIR_STRIDE + DIR_EDGE_B) as usize;
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
}
pub unsafe fn destroy_contact(id: usize, wake: bool) {
    let world = regions::active();
    let d = manifolds::dir_col();
    let o = id * DIR_STRIDE;
    let flags = d.get(o + 6);
    let a = d.get(o + DIR_EDGE_A) as usize;
    let b = d.get(o + DIR_EDGE_B) as usize;
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
    if wake && flags & 1 != 0 {
        solver_set::wake(bodies::record(world, a).set_index as usize);
        solver_set::wake(bodies::record(world, b).set_index as usize);
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
    let sim2 = bodies::sim2_base() as *const u32;
    let mut bullets = false;
    for i in 0..count {
        let flags = *sim2.add(i * body::SIM2_STRIDE + body::S2_FLAGS);
        bullets |= flags & (body::flags::IS_FAST | crate::continuous::IS_BULLET)
            == (body::flags::IS_FAST | crate::continuous::IS_BULLET);
    }
    crate::body_record::runtime::finish(count, dt, enable_sleep);
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

#[link(wasm_import_module = "env")]
extern "C" {
    fn now(output: *mut f64);
}
pub(crate) fn ticks() -> f64 {
    let mut value = 0.0;
    unsafe { now(&mut value) };
    value
}
// types.h b3Profile, in its public ABI field order.
#[repr(C)]
#[derive(Clone, Copy)]
struct Profile {
    step: f32,
    pairs: f32,
    collide: f32,
    solve: f32,
    solver_setup: f32,
    constraints: f32,
    prepare_constraints: f32,
    integrate_velocities: f32,
    warm_start: f32,
    solve_impulses: f32,
    integrate_positions: f32,
    relax_impulses: f32,
    apply_restitution: f32,
    store_impulses: f32,
    split_islands: f32,
    transforms: f32,
    sensor_hits: f32,
    joint_events: f32,
    hit_events: f32,
    refit: f32,
    bullets: f32,
    sleep_islands: f32,
    sensors: f32,
}
impl Profile {
    const ZERO: Self = Self {
        step: 0.0,
        pairs: 0.0,
        collide: 0.0,
        solve: 0.0,
        solver_setup: 0.0,
        constraints: 0.0,
        prepare_constraints: 0.0,
        integrate_velocities: 0.0,
        warm_start: 0.0,
        solve_impulses: 0.0,
        integrate_positions: 0.0,
        relax_impulses: 0.0,
        apply_restitution: 0.0,
        store_impulses: 0.0,
        split_islands: 0.0,
        transforms: 0.0,
        sensor_hits: 0.0,
        joint_events: 0.0,
        hit_events: 0.0,
        refit: 0.0,
        bullets: 0.0,
        sleep_islands: 0.0,
        sensors: 0.0,
    };
}
const _: () = assert!(core::mem::size_of::<Profile>() == 23 * 4);
static mut PROFILE: [Profile; regions::MAX_WORLDS] = [Profile::ZERO; regions::MAX_WORLDS];
pub(crate) unsafe fn accumulate(field: usize, start: f64) {
    let p = (&raw mut PROFILE)
        .cast::<Profile>()
        .add(regions::active())
        .cast::<f32>()
        .add(field);
    *p += (ticks() - start) as f32;
}
#[export_name = "stepProfilePtr"]
pub unsafe extern "C" fn profile_ptr(world: usize) -> *const f32 {
    (&raw const PROFILE).cast::<Profile>().add(world).cast()
}
struct Driver {
    phase: u32,
    threads: usize,
    substeps: usize,
    gravity: [f32; 3],
    max_speed: f32,
    contact_speed: f32,
    restitution: f32,
    hit: f32,
    recycle: f32,
    warm: bool,
    continuous: bool,
    sleep: bool,
    default_mix: bool,
    count: usize,
    step_start: f64,
    phase_start: f64,
    solve_start: f64,
}
static mut PAIRS_ONLY: bool = false;
#[export_name = "pairsBegin"]
pub unsafe extern "C" fn pairs_begin(world: usize, threads: usize) {
    bodies::body_set_active_world(world as u32);
    crate::shapes::shape_set_active_world(world as u32);
    DRIVER.threads = threads;
    DRIVER.phase = 1;
    PAIRS_ONLY = true;
}
static mut DRIVER: Driver = Driver {
    phase: 0,
    threads: 1,
    substeps: 1,
    gravity: [0.0; 3],
    max_speed: 0.0,
    contact_speed: 0.0,
    restitution: 0.0,
    hit: 0.0,
    recycle: 0.0,
    warm: false,
    continuous: false,
    sleep: false,
    default_mix: true,
    count: 0,
    step_start: 0.0,
    phase_start: 0.0,
    solve_start: 0.0,
};
#[export_name = "stepBegin"]
pub unsafe extern "C" fn begin(
    world: usize,
    dt: f32,
    substeps: i32,
    threads: usize,
    gx: f32,
    gy: f32,
    gz: f32,
    hertz: f32,
    damping: f32,
    max_speed: f32,
    contact_speed: f32,
    restitution: f32,
    hit: f32,
    recycle: f32,
    warm: bool,
    continuous: bool,
    sleep: bool,
    default_mix: bool,
) {
    regions::select(world as u32);
    bodies::body_set_active_world(world as u32);
    crate::shapes::shape_set_active_world(world as u32);
    PROFILE[world] = Profile::ZERO;
    SYNC_COUNT = 0;
    PAIRS_ONLY = false;
    let start = ticks();
    let substeps = substeps.max(1) as usize;
    context(dt, substeps, hertz, damping);
    bodies::reserve_bodies(
        bodies::body_length(world as u32)
            .max(16)
            .next_power_of_two(),
    );
    events::begin_step(world);
    DRIVER = Driver {
        phase: 1,
        threads,
        substeps: substeps.max(1),
        gravity: [gx, gy, gz],
        max_speed,
        contact_speed,
        restitution,
        hit,
        recycle,
        warm,
        continuous,
        sleep,
        default_mix,
        count: 0,
        step_start: start,
        phase_start: ticks(),
        solve_start: 0.0,
    };
}
unsafe fn parallel(kind: u32, count: usize, a: f32) -> bool {
    let fork = crate::solve::par_build(kind, count, DRIVER.threads, a) != 0;
    if !fork {
        crate::solve::run_mt();
    }
    fork
}
#[export_name = "contactCreateWorld"]
pub unsafe extern "C" fn create_contact(world: usize, a: usize, b: usize, child: i32) {
    regions::select(world as u32);
    let r = crate::shapes::col();
    let compound = if r.get(a * crate::shapes::SHAPE_STRIDE) == 1 {
        Some(a)
    } else if r.get(b * crate::shapes::SHAPE_STRIDE) == 1 {
        Some(b)
    } else {
        None
    };
    let mesh = compound
        .is_some_and(|id| crate::geo::shape_compound_child_type(world, id, child as usize) == 4);
    let id = crate::body_record::runtime::create_contact(
        world,
        a,
        b,
        child,
        if mesh { 0x0040_0000 } else { 0 },
    );
    if id != usize::MAX {
        crate::table::add_pair(a as u32, b as u32, child as u32);
        contact_list::update(id);
    }
}
unsafe fn create_pairs() {
    let world = regions::active();
    let heads = crate::pairwork::pairs_cand_end_ptr();
    let pairs = crate::pairwork::pairs_cand_ptr();
    for i in 0..crate::broad::move_count() {
        let mut entry = *heads.add(i);
        while entry != u32::MAX {
            let p = pairs.add(entry as usize * 4);
            let child = *p;
            let a = *p.add(1) as usize;
            let b = *p.add(2) as usize;
            entry = *p.add(3);
            create_contact(world, a, b, child as i32);
        }
    }
    crate::broad::clear_moves();
}
unsafe fn sleep_islands() {
    for index in (0..solver_set::array_count(2, 1)).rev() {
        let id = solver_set::array_get(2, 1, index) as usize;
        if island::can_sleep(id) {
            try_sleep_island(id);
        }
    }
}
#[export_name = "solverSetTrySleepIsland"]
pub unsafe extern "C" fn try_sleep_island(id: usize) {
    let world = regions::active();
    if island::field(id, 3) > 0 && island::array_count(id, 0) > 1 {
        return;
    }
    let index = island::field(id, 1) as usize;
    let target = solver_set::create();
    for i in 0..island::array_count(id, 0) {
        let body = island::array_get(id, 0, i, 0) as usize;
        crate::body_record::runtime::transfer(world, body, target, false);
        let mut key = bodies::record(world, body).head_contact_key;
        let d = manifolds::dir_col();
        while key != -1 {
            let o = (key >> 1) as usize * DIR_STRIDE;
            let side = (key & 1) as usize;
            key = d.get(o + DIR_EDGE_A + 2 + 3 * side) as i32;
            if d.get(o + DIR_SET_INDEX) == 1 || d.get(o + DIR_COLOR_INDEX) != u32::MAX {
                continue;
            }
            let other = d.get(o + DIR_EDGE_A + 3 * (side ^ 1)) as usize;
            if bodies::record(world, other).set_index != 2 {
                solver_set::move_contact(2, d.get(o + DIR_LOCAL_INDEX) as usize, 1);
            }
        }
    }
    for i in 0..island::array_count(id, 1) {
        solver_set::sleep_contact(island::array_get(id, 1, i, 0) as usize, target);
    }
    for i in 0..island::array_count(id, 2) {
        crate::joint_lifecycle::transfer(island::array_get(id, 2, i, 0) as usize, target);
    }
    let result = solver_set::move_island(2, index, target) as *const u32;
    if *result.add(1) != u32::MAX {
        island::set_field(*result.add(1) as usize, 1, index as i32);
    }
    island::set_field(id, 0, target as i32);
    island::set_field(id, 1, *result as i32);
    for i in 0..island::array_count(id, 0) {
        let body = island::array_get(id, 0, i, 0) as usize;
        let d = manifolds::dir_col();
        let mut key = bodies::record(world, body).head_contact_key;
        while key != -1 {
            let contact = (key >> 1) as usize;
            let side = (key & 1) as usize;
            key = d.get(contact * DIR_STRIDE + DIR_EDGE_A + 2 + 3 * side) as i32;
            contact_list::update(contact);
        }
    }
    if island::split_candidate() == id as i32 {
        island::set_split_candidate(-1);
    }
}
// DONE=0, parallel task=1, custom material callbacks=2. All serial work continues in this call.
#[export_name = "stepAdvance"]
pub unsafe extern "C" fn advance() -> u32 {
    loop {
        let world = regions::active();
        match DRIVER.phase {
            0 => return 0,
            1 => {
                if crate::broad::move_count() == 0 {
                    DRIVER.phase = 3;
                    continue;
                }
                if crate::broad::set_cap() == 0 {
                    crate::table::create_set(16);
                }
                crate::pairwork::reserve_pairs();
                DRIVER.phase = 2;
                if parallel(
                    4,
                    crate::broad::move_count(),
                    crate::broad::set_cap() as f32,
                ) {
                    return 1;
                }
            }
            2 => {
                if crate::pairwork::pairs_overflow() != 0 {
                    DRIVER.phase = 1;
                    continue;
                }
                crate::pairwork::rebuild_trees();
                create_pairs();
                DRIVER.phase = 3;
            }
            3 => {
                if PAIRS_ONLY {
                    DRIVER.phase = 0;
                    PAIRS_ONLY = false;
                    return 0;
                }
                accumulate(1, DRIVER.phase_start);
                DRIVER.phase_start = ticks();
                let count = contact_list::count();
                DRIVER.phase = 4;
                if count != 0 {
                    crate::arena::reserve_collide(
                        count,
                        DRIVER.threads,
                        DRIVER.default_mix as u32,
                        DRIVER.recycle,
                    );
                    contact_list::copy(crate::arena::collide_list_ptr() as *mut u32);
                    if parallel(2, count, 0.0) {
                        return 1;
                    }
                }
            }
            4 => {
                DRIVER.phase = 5;
                if !DRIVER.default_mix && contact_list::count() != 0 {
                    return 2;
                }
            }
            5 => {
                if contact_list::count() != 0 {
                    apply_contact_transitions();
                }
                accumulate(2, DRIVER.phase_start);
                DRIVER.solve_start = ticks();
                DRIVER.count = solver_set::body_count(2);
                DRIVER.phase = 6;
                if CONTEXT[0] <= 0.0 {
                    DRIVER.phase = 10;
                    continue;
                }
                if DRIVER.count == 0 {
                    events::update_begin_impulses(world);
                    DRIVER.phase = 10;
                    continue;
                }
                let start = ticks();
                solve_build(
                    DRIVER.threads,
                    DRIVER.substeps,
                    DRIVER.gravity[0],
                    DRIVER.gravity[1],
                    DRIVER.gravity[2],
                    DRIVER.max_speed,
                    DRIVER.contact_speed,
                    DRIVER.warm,
                    DRIVER.restitution,
                    DRIVER.hit,
                    DRIVER.continuous,
                    DRIVER.sleep,
                );
                accumulate(4, start);
                if DRIVER.threads > 1 {
                    return 1;
                }
                crate::solve::run_mt();
            }
            6 => {
                island::set_split_candidate(-1);
                PROFILE[world].constraints =
                    (ticks() - DRIVER.solve_start) as f32 - PROFILE[world].solver_setup;
                DRIVER.phase_start = ticks();
                DRIVER.phase = 12;
                if parallel(7, DRIVER.count, 0.0) {
                    return 1;
                }
            }
            12 => {
                let bullets = finalize(DRIVER.count, CONTEXT[0], DRIVER.sleep);
                accumulate(15, DRIVER.phase_start);
                events::update_begin_impulses(world);
                let start = ticks();
                crate::joint_lifecycle::collect_events();
                accumulate(17, start);
                let start = ticks();
                events::build_hits(world, DRIVER.hit);
                accumulate(18, start);
                let start = ticks();
                crate::treework::enlarge_pass(DRIVER.count, 0);
                accumulate(19, start);
                DRIVER.phase = 8;
                DRIVER.phase_start = ticks();
                if bullets && parallel(3, DRIVER.count, 0.0) {
                    return 1;
                }
                if !bullets {
                    DRIVER.phase = 9;
                }
            }
            8 => {
                crate::treework::enlarge_pass(DRIVER.count, 1);
                accumulate(20, DRIVER.phase_start);
                DRIVER.phase = 9;
            }
            9 => {
                let start = ticks();
                crate::continuous::consume(world, DRIVER.count, false);
                crate::continuous::consume(world, DRIVER.count, true);
                accumulate(16, start);
                SYNC_COUNT = bodies::body_sync_moved(events::count(world, 6));
                if DRIVER.sleep {
                    let start = ticks();
                    island::set_split_candidate(crate::body_record::runtime::gather_split(
                        DRIVER.count,
                    ));
                    sleep_islands();
                    accumulate(21, start);
                }
                DRIVER.phase = 10;
            }
            10 => {
                if CONTEXT[0] > 0.0 {
                    accumulate(3, DRIVER.solve_start);
                }
                DRIVER.phase_start = ticks();
                let count = crate::sensor::prepare();
                DRIVER.phase = 11;
                if count != 0 && parallel(6, count, 0.0) {
                    return 1;
                }
            }
            11 => {
                crate::sensor::publish(world);
                accumulate(22, DRIVER.phase_start);
                events::end_step(world);
                accumulate(0, DRIVER.step_start);
                DRIVER.phase = 0;
            }
            _ => unreachable!(),
        }
    }
}

static mut SYNC_COUNT: usize = 0;
#[export_name = "bodySyncCount"]
pub unsafe extern "C" fn sync_count() -> usize {
    SYNC_COUNT
}
#[export_name = "stepInvDt"]
pub unsafe extern "C" fn inv_dt() -> f32 {
    CONTEXT[1]
}
#[export_name = "stepInvH"]
pub unsafe extern "C" fn inv_h() -> f32 {
    CONTEXT[3]
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
