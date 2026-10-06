//! physics_world.h event arrays; types.h event records.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Id {
    pub index1: u32,
    pub world0: u16,
    pub generation: u16,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SensorTouch {
    pub sensor: Id,
    pub visitor: Id,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactEnd {
    pub a: Id,
    pub b: Id,
    pub contact: Id,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactBegin {
    pub a: Id,
    pub b: Id,
    pub contact: Id,
    pub normal_impulse: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactHit {
    pub a: Id,
    pub b: Id,
    pub contact: Id,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub speed: f32,
    pub padding: u32,
    pub material_a: u64,
    pub material_b: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JointEvent {
    pub joint: Id,
    pub user_data: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WorldTransform {
    pub p: [f32; 3],
    pub q: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BodyMove {
    pub user_data: u32,
    pub transform: WorldTransform,
    pub body: Id,
    pub fell_asleep: bool,
    pub padding: [u8; 3],
}
#[cfg(target_arch = "wasm32")]
pub(crate) use runtime::*;

#[cfg(target_arch = "wasm32")]
mod runtime {
    use super::*;
    use crate::{bodies, body, geo, manifold_abi::*, manifolds, math::Vec3, regions, shapes};
    struct Events {
        sensor_begin: Vec<SensorTouch>,
        sensor_end: [Vec<SensorTouch>; 2],
        contact_begin: Vec<ContactBegin>,
        contact_end: [Vec<ContactEnd>; 2],
        contact_hit: Vec<ContactHit>,
        joints: Vec<JointEvent>,
        end_index: usize,
        move_count: usize,
    }
    impl Events {
        const EMPTY: Self = Self {
            sensor_begin: Vec::new(),
            sensor_end: [Vec::new(), Vec::new()],
            contact_begin: Vec::new(),
            contact_end: [Vec::new(), Vec::new()],
            contact_hit: Vec::new(),
            joints: Vec::new(),
            end_index: 0,
            move_count: 0,
        };
    }
    static mut WORLDS: [Events; regions::MAX_WORLDS] =
        [const { Events::EMPTY }; regions::MAX_WORLDS];
    unsafe fn state(world: usize) -> &'static mut Events {
        &mut (*(&raw mut WORLDS))[world]
    }
    pub unsafe fn reset(world: usize) {
        *state(world) = Events::EMPTY;
    }
    unsafe fn save<T: Copy>(array: &[T], out: &mut Vec<u8>) {
        regions::write_word(out, array.len());
        out.extend_from_slice(core::slice::from_raw_parts(
            array.as_ptr().cast::<u8>(),
            core::mem::size_of_val(array),
        ));
    }
    unsafe fn load<T: Copy>(input: &mut &[u8]) -> Vec<T> {
        let count = regions::read_word(input);
        let bytes = count * core::mem::size_of::<T>();
        let (data, rest) = input.split_at(bytes);
        *input = rest;
        let mut out = Vec::<T>::with_capacity(count);
        core::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes);
        out.set_len(count);
        out
    }
    pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
        let w = state(world);
        regions::write_word(out, w.end_index);
        regions::write_word(out, w.move_count);
        save(&w.sensor_begin, out);
        save(&w.sensor_end[0], out);
        save(&w.sensor_end[1], out);
        save(&w.contact_begin, out);
        save(&w.contact_end[0], out);
        save(&w.contact_end[1], out);
        save(&w.contact_hit, out);
        save(&w.joints, out);
    }
    pub unsafe fn restore(world: usize, input: &mut &[u8]) {
        let w = state(world);
        w.end_index = regions::read_word(input);
        w.move_count = regions::read_word(input);
        w.sensor_begin = load(input);
        w.sensor_end[0] = load(input);
        w.sensor_end[1] = load(input);
        w.contact_begin = load(input);
        w.contact_end[0] = load(input);
        w.contact_end[1] = load(input);
        w.contact_hit = load(input);
        w.joints = load(input);
    }
    pub fn id(world: usize, index: usize, generation: u32) -> Id {
        Id {
            index1: index as u32 + 1,
            world0: world as u16,
            generation: generation as u16,
        }
    }
    pub unsafe fn shape_id(world: usize, index: usize) -> Id {
        id(
            world,
            index,
            shapes::shape_generation(world as u32, index as u32),
        )
    }
    pub unsafe fn sensor_touch(
        world: usize,
        sensor: usize,
        visitor: crate::sensor::Visitor,
        end: bool,
    ) {
        let e = SensorTouch {
            sensor: shape_id(world, sensor),
            visitor: id(world, visitor.shape_id as usize, visitor.generation as u32),
        };
        let w = state(world);
        if end {
            w.sensor_end[w.end_index].push(e);
        } else {
            w.sensor_begin.push(e);
        }
    }
    #[export_name = "eventBeginStep"]
    pub unsafe extern "C" fn begin_step(world: usize) {
        let w = state(world);
        w.sensor_begin.clear();
        w.contact_begin.clear();
        w.contact_hit.clear();
        w.joints.clear();
        w.move_count = 0;
    }
    #[export_name = "eventEndStep"]
    pub unsafe extern "C" fn end_step(world: usize) {
        let w = state(world);
        w.end_index = 1 - w.end_index;
        w.sensor_end[w.end_index].clear();
        w.contact_end[w.end_index].clear();
    }
    #[export_name = "eventContactTouch"]
    pub unsafe extern "C" fn contact_touch(world: usize, contact: usize, begin: bool) {
        regions::select(world as u32);
        let d = manifolds::dir_col();
        let o = contact * DIR_STRIDE;
        let a = shape_id(world, d.get(o + DIR_SHAPE_A) as usize);
        let b = shape_id(world, d.get(o + DIR_SHAPE_B) as usize);
        let contact = id(world, contact, d.get(o + DIR_GENERATION));
        let w = state(world);
        if begin {
            w.contact_begin.push(ContactBegin {
                a,
                b,
                contact,
                normal_impulse: 0.0,
            });
        } else {
            w.contact_end[w.end_index].push(ContactEnd { a, b, contact });
        }
    }
    unsafe fn total_impulse(contact: usize) -> f32 {
        let entry = read_dir(manifolds::dir_col(), contact);
        let m = block_col(
            manifolds::pool_col(),
            entry.manifold_base,
            entry.manifold_count,
        );
        let mut total = 0.0;
        for i in 0..entry.manifold_count {
            let o = i * MANIFOLD_STRIDE;
            for p in 0..m.get(o + M_POINT_COUNT).to_bits() as usize {
                total += m.get(o + M_POINTS + p * POOL_POINT_STRIDE + P_TOTAL_NORMAL_IMPULSE);
            }
        }
        total
    }
    unsafe fn material(shape: usize, child: usize, triangle: i32) -> u64 {
        let index = geo::shape_material_index(regions::active(), shape, child, triangle as usize);
        if shapes::shape_material_count(regions::active() as u32, shape as u32) == 0 {
            return 0;
        }
        let r = shapes::material(shape, index);
        ((r[7] as u64) << 32) | r[6] as u64
    }
    #[export_name = "eventFinishContacts"]
    pub unsafe extern "C" fn finish_contacts(world: usize, threshold: f32) {
        regions::select(world as u32);
        let w = state(world);
        let d = manifolds::dir_col();
        for e in &mut w.contact_begin {
            let contact = e.contact.index1 as usize - 1;
            let o = contact * DIR_STRIDE;
            if d.get(o + DIR_CONTACT_ID) != u32::MAX
                && d.get(o + DIR_GENERATION) == e.contact.generation as u32
            {
                e.normal_impulse = total_impulse(contact);
            }
        }
        for contact in 0..manifolds::contact_record_capacity(world) {
            let o = contact * DIR_STRIDE;
            if d.get(o + 11) == 0 {
                continue;
            }
            d.set(o + 11, 0);
            if d.get(o + DIR_CONTACT_ID) == u32::MAX {
                continue;
            }
            let a = d.get(o + DIR_SHAPE_A) as usize;
            let b = d.get(o + DIR_SHAPE_B) as usize;
            let ar = bodies::record(
                world,
                shapes::col().get(a * shapes::SHAPE_STRIDE + shapes::S_QUERY_BODY) as usize,
            );
            let br = bodies::record(
                world,
                shapes::col().get(b * shapes::SHAPE_STRIDE + shapes::S_QUERY_BODY) as usize,
            );
            let center = body::read_fin(
                crate::col::Col::new(
                    crate::solver_set::body_ptr(ar.set_index as usize, ar.local_index as usize, 2)
                        as *mut f32,
                    12,
                ),
                0,
            )
            .center;
            let center_b = body::read_fin(
                crate::col::Col::new(
                    crate::solver_set::body_ptr(br.set_index as usize, br.local_index as usize, 2)
                        as *mut f32,
                    12,
                ),
                0,
            )
            .center;
            let mid = center.lerp(center_b, 0.5);
            let entry = read_dir(d, contact);
            let m = block_col(
                manifolds::pool_col(),
                entry.manifold_base,
                entry.manifold_count,
            );
            let mut speed = threshold;
            let mut best = None;
            for i in 0..entry.manifold_count {
                let o = i * MANIFOLD_STRIDE;
                for p in 0..m.get(o + M_POINT_COUNT).to_bits() as usize {
                    let n = o + M_POINTS + p * POOL_POINT_STRIDE;
                    let approach = -m.get(n + P_NORMAL_VELOCITY);
                    if approach > speed && m.get(n + P_TOTAL_NORMAL_IMPULSE) > 0.0 {
                        speed = approach;
                        let a = Vec3::new(m.get(n), m.get(n + 1), m.get(n + 2));
                        let b = Vec3::new(m.get(n + 3), m.get(n + 4), m.get(n + 5));
                        best = Some((
                            mid.add(a.lerp(b, 0.5)),
                            Vec3::new(m.get(o), m.get(o + 1), m.get(o + 2)),
                            m.get(n + P_TRIANGLE_INDEX).to_bits() as i32,
                        ));
                    }
                }
            }
            if let Some((point, normal, triangle)) = best {
                w.contact_hit.push(ContactHit {
                    a: shape_id(world, a),
                    b: shape_id(world, b),
                    contact: id(world, contact, d.get(o + DIR_GENERATION)),
                    point: [point.x, point.y, point.z],
                    normal: [normal.x, normal.y, normal.z],
                    speed,
                    padding: 0,
                    material_a: material(a, d.get(o + DIR_CHILD_INDEX) as usize, triangle),
                    material_b: material(b, 0, triangle),
                });
            }
        }
    }
    pub unsafe fn joint(world: usize, index: usize, generation: u32) {
        state(world).joints.push(JointEvent {
            joint: id(world, index, generation),
            user_data: index as u32,
        });
    }
    pub unsafe fn clear_joints(world: usize) {
        state(world).joints.clear();
    }
    pub unsafe fn set_move_count(world: usize, count: usize) {
        state(world).move_count = count;
    }
    pub unsafe fn write_move(index: usize) {
        let world = regions::active();
        let sim2 = bodies::sim2_base() as *const u32;
        let body_id = *sim2.add(index * body::SIM2_STRIDE + body::S2_BODY_ID) as usize;
        let fin = (bodies::fin_base() as *const f32).add(index * 12 + 9);
        let rotation = (bodies::sim_base() as *const crate::math::Quat)
            .cast::<f32>()
            .add(index * 32 + 28)
            .cast::<crate::math::Quat>();
        *(bodies::move_base() as *mut BodyMove).add(index) = BodyMove {
            user_data: body_id as u32,
            transform: WorldTransform {
                p: [*fin, *fin.add(1), *fin.add(2)],
                q: [
                    (*rotation).v.x,
                    (*rotation).v.y,
                    (*rotation).v.z,
                    (*rotation).s,
                ],
            },
            body: id(world, body_id, bodies::active_generation(body_id as u32)),
            fell_asleep: false,
            padding: [0; 3],
        };
    }
    #[export_name = "eventCount"]
    pub unsafe extern "C" fn count(world: usize, kind: usize) -> usize {
        let w = state(world);
        match kind {
            0 => w.sensor_begin.len(),
            1 => w.sensor_end[1 - w.end_index].len(),
            2 => w.contact_begin.len(),
            3 => w.contact_end[1 - w.end_index].len(),
            4 => w.contact_hit.len(),
            5 => w.joints.len(),
            6 => w.move_count,
            7 => w.sensor_end[w.end_index].len(),
            _ => panic!("event kind"),
        }
    }
    unsafe fn pointer(world: usize, kind: usize, index: usize) -> *const u32 {
        regions::select(world as u32);
        let w = state(world);
        match kind {
            0 => w.sensor_begin.as_ptr().add(index).cast(),
            1 => w.sensor_end[1 - w.end_index].as_ptr().add(index).cast(),
            2 => w.contact_begin.as_ptr().add(index).cast(),
            3 => w.contact_end[1 - w.end_index].as_ptr().add(index).cast(),
            4 => w.contact_hit.as_ptr().add(index).cast(),
            5 => w.joints.as_ptr().add(index).cast(),
            6 => (bodies::move_base() as *const BodyMove).add(index).cast(),
            _ => panic!("event kind"),
        }
    }
    #[export_name = "eventWord"]
    pub unsafe extern "C" fn word(world: usize, kind: usize, index: usize, lane: usize) -> u32 {
        *pointer(world, kind, index).add(lane)
    }
    #[export_name = "eventFloat"]
    pub unsafe extern "C" fn float(world: usize, kind: usize, index: usize, lane: usize) -> f32 {
        f32::from_bits(word(world, kind, index, lane))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};
    #[test]
    fn records_match_types_h_with_the_authorized_begin_impulse_extension() {
        assert_eq!(size_of::<Id>(), 8);
        assert_eq!(size_of::<SensorTouch>(), 16);
        assert_eq!(size_of::<ContactEnd>(), 24);
        assert_eq!(size_of::<ContactBegin>(), 28);
        assert_eq!(offset_of!(ContactBegin, normal_impulse), 24);
        assert_eq!(size_of::<ContactHit>(), 72);
        assert_eq!(offset_of!(ContactHit, material_a), 56);
        assert_eq!(size_of::<JointEvent>(), 12);
        assert_eq!(size_of::<BodyMove>(), 44);
        assert_eq!(offset_of!(BodyMove, transform), 4);
        assert_eq!(offset_of!(BodyMove, body), 32);
        assert_eq!(offset_of!(BodyMove, fell_asleep), 40);
    }
}
