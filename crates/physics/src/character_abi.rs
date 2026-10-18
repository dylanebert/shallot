//! Resident-world adapter for the sample character movement phase.
use crate::{character::*, math::Vec3};
use core::mem::MaybeUninit;
static mut INPUT: MaybeUninit<Input> = MaybeUninit::uninit();
static mut OUTPUT: MaybeUninit<Output> = MaybeUninit::uninit();
#[export_name = "characterInputPtr"]
pub extern "C" fn input_ptr() -> *mut Input {
    (&raw mut INPUT).cast()
}
#[export_name = "characterOutputPtr"]
pub extern "C" fn output_ptr() -> *const Output {
    (&raw const OUTPUT).cast()
}
struct Resident {
    world: usize,
    own: u32,
    max_speed: f32,
}
impl Resident {
    unsafe fn query(&self, op: u32, origin: Vec3, translation: Vec3) -> &'static [u32] {
        let h = core::slice::from_raw_parts_mut(crate::world_query::header_ptr(), 20);
        h[6] = 0;
        h[7] = 1;
        h[8] = u32::MAX;
        h[9] = if op == 5 { u32::MAX } else { 0xfffffffd };
        h[10] = origin.x.to_bits();
        h[11] = origin.y.to_bits();
        h[12] = origin.z.to_bits();
        h[19] = self.own + 1;
        let q = core::slice::from_raw_parts_mut(crate::query_abi::input_ptr(), 398);
        q[9] = translation.x;
        q[10] = translation.y;
        q[11] = translation.z;
        crate::world_query::run_in_world(self.world, op, 0);
        core::slice::from_raw_parts(crate::world_query::result_ptr(), 80)
    }
}
fn vector(r: &[u32], i: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(r[i]),
        f32::from_bits(r[i + 1]),
        f32::from_bits(r[i + 2]),
    )
}
impl World for Resident {
    fn ray(&mut self, origin: Vec3, translation: Vec3) -> Option<Ground> {
        unsafe {
            let r = self.query(3, origin, translation);
            (r[0] != u32::MAX).then(|| Ground {
                shape: r[0],
                fraction: f32::from_bits(r[5]),
                point: origin.add(vector(r, 6)),
                normal: vector(r, 9),
            })
        }
    }
    fn collide(
        &mut self,
        position: Vec3,
        a: Vec3,
        b: Vec3,
        radius: f32,
        contacts: &mut [Contact; 8],
    ) -> usize {
        unsafe {
            let q = core::slice::from_raw_parts_mut(crate::query_abi::input_ptr(), 398);
            q[7] = 2.0;
            q[8] = radius;
            q[14..20].copy_from_slice(&[a.x, a.y, a.z, b.x, b.y, b.z]);
            let r = self.query(5, position, Vec3::ZERO);
            let count = r[0] as usize;
            for (i, c) in contacts[..count].iter_mut().enumerate() {
                let n = 16 + 8 * i;
                *c = Contact {
                    shape: r[n],
                    plane: crate::math::Plane {
                        normal: vector(r, n + 1),
                        offset: f32::from_bits(r[n + 4]),
                    },
                    point: position.add(vector(r, n + 5)),
                };
            }
            count
        }
    }
    fn cast(&mut self, position: Vec3, delta: Vec3) -> f32 {
        unsafe { f32::from_bits(self.query(6, position, delta)[3]) }
    }
    fn body(&mut self, shape: u32) -> Body {
        unsafe {
            let id = crate::shapes::col(self.world)
                .get(shape as usize * crate::shapes::SHAPE_STRIDE + crate::shapes::S_QUERY_BODY)
                as usize;
            let record = crate::bodies::record(self.world, id);
            let fin = crate::body::read_fin(
                crate::bodies::column(self.world, id, 1, crate::body::SIM_STRIDE),
                0,
            );
            let sim = crate::body::read_sim(
                crate::bodies::column(self.world, id, 1, crate::body::SIM_STRIDE),
                0,
            );
            let awake = record.set_index == 2;
            let (linear, angular) = if awake {
                let s = crate::body::read_state(
                    crate::bodies::column(self.world, id, 0, crate::body::STATE_STRIDE),
                    0,
                );
                (s.linear_velocity, s.angular_velocity)
            } else {
                (Vec3::ZERO, Vec3::ZERO)
            };
            Body {
                id: id as u32,
                dynamic: record.body_type == 2,
                awake,
                center: fin.center,
                linear,
                angular,
                inv_mass: sim.inv_mass,
                inv_inertia: sim.inv_inertia_world,
            }
        }
    }
    fn impulse(&mut self, body: u32, impulse: Vec3, point: Vec3) {
        unsafe {
            crate::body_record::runtime::apply_in_world(
                self.world,
                body as usize,
                3,
                impulse.x,
                impulse.y,
                impulse.z,
                point.x,
                point.y,
                point.z,
                self.max_speed,
                true,
            );
        }
    }
}
#[export_name = "characterMove"]
pub unsafe extern "C" fn run(world: usize, own: u32) {
    crate::regions::select(world as u32);
    let input = &*input_ptr();
    let result = crate::character::run(
        &mut Resident {
            world,
            own,
            max_speed: input.max_speed,
        },
        input,
    );
    (&raw mut OUTPUT).write(MaybeUninit::new(result));
}
