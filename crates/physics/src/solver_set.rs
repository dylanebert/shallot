//! solver_set.c's set array and id pool. Body columns retain the solver's column layout.
use crate::body::{FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};
const AWAKE: usize = 2;
const STRIDES: [usize; 6] = [STATE_STRIDE, SIM_STRIDE, FIN_STRIDE, 1, 1, SIM2_STRIDE];
struct SolverSet {
    columns: Columns<6>,
    body_count: usize,
    indices: [Vec<i32>; 2],
    index: i32,
}
impl SolverSet {
    fn empty() -> Self {
        Self {
            columns: Columns::EMPTY,
            body_count: 0,
            indices: [Vec::new(), Vec::new()],
            index: -1,
        }
    }
}
struct Sets {
    sets: Vec<SolverSet>,
    free: Vec<usize>,
}
static mut WORLDS: [Sets; MAX_WORLDS] = [const {
    Sets {
        sets: Vec::new(),
        free: Vec::new(),
    }
}; MAX_WORLDS];
unsafe fn world() -> &'static mut Sets {
    &mut WORLDS[regions::active()]
}
unsafe fn set(id: usize) -> &'static mut SolverSet {
    &mut world().sets[id]
}
#[export_name = "solverSetCreate"]
pub unsafe extern "C" fn create() -> usize {
    let w = world();
    let id = if let Some(id) = w.free.pop() {
        id
    } else {
        w.sets.push(SolverSet::empty());
        w.sets.len() - 1
    };
    w.sets[id].index = id as i32;
    id
}
#[export_name = "solverSetCount"]
pub unsafe extern "C" fn count() -> usize {
    world().sets.len()
}
#[export_name = "solverSetIndex"]
pub unsafe extern "C" fn index(id: usize) -> i32 {
    set(id).index
}
#[export_name = "solverSetDestroy"]
pub unsafe extern "C" fn destroy(id: usize) {
    let s = set(id);
    s.columns.release();
    *s = SolverSet::empty();
    world().free.push(id);
}
#[export_name = "solverSetBodyCount"]
pub unsafe extern "C" fn body_count(id: usize) -> usize {
    set(id).body_count
}
#[export_name = "solverSetBodyAppend"]
pub unsafe extern "C" fn body_append(id: usize) -> usize {
    let s = set(id);
    let i = s.body_count;
    s.body_count += 1;
    if id != AWAKE {
        for c in [1, 2, 5] {
            s.columns.reserve(c, s.body_count * STRIDES[c] * 4);
        }
    }
    i
}
#[export_name = "solverSetBodyPop"]
pub unsafe extern "C" fn body_pop(id: usize) {
    set(id).body_count -= 1;
}
#[export_name = "solverSetLayout"]
pub unsafe extern "C" fn layout(id: usize) -> *const u32 {
    set(id).columns.layout.as_ptr()
}
#[export_name = "solverSetArrayCount"]
pub unsafe extern "C" fn array_count(id: usize, kind: usize) -> usize {
    set(id).indices[kind].len()
}
#[export_name = "solverSetArrayGet"]
pub unsafe extern "C" fn array_get(id: usize, kind: usize, i: usize) -> i32 {
    set(id).indices[kind][i]
}
#[export_name = "solverSetArrayPush"]
pub unsafe extern "C" fn array_push(id: usize, kind: usize, value: i32) -> usize {
    let v = &mut set(id).indices[kind];
    let i = v.len();
    v.push(value);
    i
}
#[export_name = "solverSetArrayRemove"]
pub unsafe extern "C" fn array_remove(id: usize, kind: usize, i: usize) -> i32 {
    let v = &mut set(id).indices[kind];
    let last = v.len() - 1;
    v.swap_remove(i);
    if i == last {
        -1
    } else {
        last as i32
    }
}
#[export_name = "solverSetArrayWrite"]
pub unsafe extern "C" fn array_write(id: usize, kind: usize, i: usize, value: i32) {
    set(id).indices[kind][i] = value;
}
#[export_name = "solverSetArrayPop"]
pub unsafe extern "C" fn array_pop(id: usize, kind: usize) {
    set(id).indices[kind].pop();
}
pub unsafe fn reset(id: usize) {
    for s in &mut WORLDS[id].sets {
        s.columns.release();
    }
    WORLDS[id] = Sets {
        sets: Vec::new(),
        free: Vec::new(),
    };
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    regions::write_word(out, w.sets.len());
    for s in &w.sets {
        regions::write_word(out, s.index as usize);
        regions::write_word(out, s.body_count);
        s.columns.snapshot(out);
        for v in &s.indices {
            regions::write_word(out, v.len());
            for &x in v {
                regions::write_word(out, x as usize);
            }
        }
    }
    regions::write_word(out, w.free.len());
    for &x in &w.free {
        regions::write_word(out, x);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let w = &mut WORLDS[id];
    let count = regions::read_word(input);
    for _ in 0..count {
        let mut s = SolverSet::empty();
        s.index = regions::read_word(input) as i32;
        s.body_count = regions::read_word(input);
        s.columns.restore(input);
        for v in &mut s.indices {
            let n = regions::read_word(input);
            for _ in 0..n {
                v.push(regions::read_word(input) as i32);
            }
        }
        w.sets.push(s);
    }
    let n = regions::read_word(input);
    for _ in 0..n {
        w.free.push(regions::read_word(input));
    }
}
