//! Box3D sensor.c: visitor arrays, overlap task and ordered event publication.
use crate::{bodies, regions, shapes, world_query};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Visitor {
    pub shape_id: u32,
    pub generation: u16,
    padding: u16,
}
#[derive(Default)]
struct Sensor {
    hits: Vec<Visitor>,
    overlaps1: Vec<Visitor>,
    overlaps2: Vec<Visitor>,
    shape_id: usize,
}
struct Sensors {
    sensors: Vec<Sensor>,
    event_bits: Vec<u64>,
}
impl Sensors {
    const EMPTY: Self = Self {
        sensors: Vec::new(),
        event_bits: Vec::new(),
    };
}
static mut WORLDS: [Sensors; regions::MAX_WORLDS] = [const { Sensors::EMPTY }; regions::MAX_WORLDS];
unsafe fn state(world: usize) -> &'static mut Sensors {
    &mut (*(&raw mut WORLDS))[world]
}
pub unsafe fn reset(world: usize) {
    *state(world) = Sensors::EMPTY;
}
pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
    let w = state(world);
    regions::write_word(out, w.sensors.len());
    for s in &w.sensors {
        regions::write_word(out, s.shape_id);
        for array in [&s.hits, &s.overlaps1, &s.overlaps2] {
            regions::write_word(out, array.len());
            for v in array {
                regions::write_word(out, v.shape_id as usize);
                regions::write_word(out, v.generation as usize);
            }
        }
    }
}
pub unsafe fn restore(world: usize, input: &mut &[u8]) {
    reset(world);
    let count = regions::read_word(input);
    for _ in 0..count {
        let mut s = Sensor {
            shape_id: regions::read_word(input),
            ..Sensor::default()
        };
        for array in [&mut s.hits, &mut s.overlaps1, &mut s.overlaps2] {
            let count = regions::read_word(input);
            for _ in 0..count {
                array.push(Visitor {
                    shape_id: regions::read_word(input) as u32,
                    generation: regions::read_word(input) as u16,
                    padding: 0,
                });
            }
        }
        state(world).sensors.push(s);
    }
}
unsafe fn visitor(world: usize, id: usize) -> Visitor {
    Visitor {
        shape_id: id as u32,
        generation: shapes::shape_generation(world as u32, id as u32) as u16,
        padding: 0,
    }
}
#[export_name = "sensorCreate"]
pub unsafe extern "C" fn create(world: usize, id: usize) {
    let w = state(world);
    let index = w.sensors.len();
    w.sensors.push(Sensor {
        shape_id: id,
        ..Sensor::default()
    });
    crate::shape_lifecycle::attach_sensor(world, id, index as i32);
}
pub unsafe fn record_hit(world: usize, sensor: usize, other: usize) {
    shapes::shape_set_active_world(world as u32);
    let index = shapes::col().get(sensor * shapes::SHAPE_STRIDE + 4) as usize;
    state(world).sensors[index].hits.push(visitor(world, other));
}
pub unsafe fn prepare() -> usize {
    let w = state(regions::active());
    w.event_bits.resize(w.sensors.len().div_ceil(64), 0);
    w.event_bits.fill(0);
    w.sensors.len()
}
pub unsafe fn task(start: usize, end: usize) {
    let world = regions::active();
    let sensors = (*(&raw const WORLDS))[world].sensors.as_ptr().cast_mut();
    let bits = (*(&raw const WORLDS))[world].event_bits.as_ptr().cast_mut();
    for index in start..end {
        let s = &mut *sensors.add(index);
        core::mem::swap(&mut s.overlaps1, &mut s.overlaps2);
        s.overlaps2.clear();
        s.overlaps2.extend_from_slice(&s.hits);
        s.hits.clear();
        let r = shapes::col();
        let n = s.shape_id * shapes::SHAPE_STRIDE;
        let body = r.get(n + shapes::S_QUERY_BODY) as usize;
        if bodies::record(world, body).set_index == 1
            || r.get(n + shapes::S_FLAGS) & shapes::SENSOR_FLAG == 0
        {
            if !s.overlaps1.is_empty() {
                core::sync::atomic::AtomicU64::from_ptr(bits.add(index / 64))
                    .fetch_or(1 << (index % 64), core::sync::atomic::Ordering::Relaxed);
            }
            continue;
        }
        let mut header = [0; 20];
        for i in 0..3 {
            header[2 * i] = *crate::broad::tree_state(i);
            header[2 * i + 1] = crate::broad::tree_cap(i) as u32;
        }
        for j in 0..6 {
            header[13 + j] = r.get(n + 10 + j);
        }
        world_query::sensor_task(s.shape_id, &header, |id| {
            s.overlaps2.push(visitor(world, id))
        });
        s.overlaps2.sort_unstable_by_key(|v| v.shape_id);
        s.overlaps2.dedup_by_key(|v| v.shape_id);
        if s.overlaps1 != s.overlaps2 {
            core::sync::atomic::AtomicU64::from_ptr(bits.add(index / 64))
                .fetch_or(1 << (index % 64), core::sync::atomic::Ordering::Relaxed);
        }
    }
}
pub unsafe fn publish(world: usize) {
    let w = state(world);
    for (block, bits) in w.event_bits.iter().copied().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let index = block * 64 + bits.trailing_zeros() as usize;
            let s = &w.sensors[index];
            let (mut a, mut b) = (0, 0);
            while a < s.overlaps1.len() || b < s.overlaps2.len() {
                let old = s.overlaps1.get(a);
                let new = s.overlaps2.get(b);
                if old == new {
                    a += 1;
                    b += 1;
                    continue;
                }
                let end = match (old, new) {
                    (Some(x), Some(y)) => (x.shape_id, x.generation) < (y.shape_id, y.generation),
                    (Some(_), None) => true,
                    _ => false,
                };
                let v = if end {
                    let v = s.overlaps1[a];
                    a += 1;
                    v
                } else {
                    let v = s.overlaps2[b];
                    b += 1;
                    v
                };
                crate::events::sensor_touch(world, s.shape_id, v, end);
            }
            bits &= bits - 1;
        }
    }
}
#[export_name = "sensorDestroy"]
pub unsafe extern "C" fn destroy(world: usize, id: usize) {
    shapes::shape_set_active_world(world as u32);
    let index = shapes::col().get(id * shapes::SHAPE_STRIDE + 4) as usize;
    let w = state(world);
    for v in &w.sensors[index].overlaps2 {
        crate::events::sensor_touch(world, id, *v, true);
    }
    w.sensors.swap_remove(index);
    if index < w.sensors.len() {
        crate::shape_lifecycle::attach_sensor(world, w.sensors[index].shape_id, index as i32);
    }
    crate::shape_lifecycle::attach_sensor(world, id, -1);
}
#[export_name = "sensorVisitorCount"]
pub unsafe extern "C" fn visitor_count(world: usize, index: usize) -> usize {
    state(world).sensors[index].overlaps2.len()
}
#[export_name = "sensorVisitorWord"]
pub unsafe extern "C" fn visitor_word(world: usize, index: usize, i: usize, word: usize) -> u32 {
    let v = state(world).sensors[index].overlaps2[i];
    if word == 0 {
        v.shape_id
    } else {
        v.generation as u32
    }
}
