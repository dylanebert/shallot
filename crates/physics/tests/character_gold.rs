use serde_json::Value;
use shallot_physics::{
    character::{self, Body, Contact, Ground, Input, Output, World},
    math::{Mat3, Plane, Quat, Vec3},
};
use std::collections::VecDeque;
fn words(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}
fn scalar(w: &[u32], i: usize) -> f32 {
    f32::from_bits(w[i])
}
fn vector(w: &[u32], i: usize) -> Vec3 {
    Vec3::new(scalar(w, i), scalar(w, i + 1), scalar(w, i + 2))
}
fn bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}
struct Transcript {
    name: String,
    events: VecDeque<Value>,
}
impl Transcript {
    fn event(&mut self, kind: &str, args: &[u32]) -> Vec<u32> {
        let event = self.events.pop_front().expect("missing native event");
        assert_eq!(event["kind"].as_str().unwrap(), kind, "{}", self.name);
        assert_eq!(words(&event["args"]), args, "{} {kind} request", self.name);
        words(&event["result"])
    }
}
impl World for Transcript {
    fn ray(&mut self, origin: Vec3, translation: Vec3) -> Option<Ground> {
        let r = self.event("ray", &[bits(origin), bits(translation)].concat());
        (!r.is_empty()).then(|| Ground {
            shape: r[0],
            fraction: scalar(&r, 1),
            point: vector(&r, 2),
            normal: vector(&r, 5),
        })
    }
    fn collide(
        &mut self,
        position: Vec3,
        a: Vec3,
        b: Vec3,
        radius: f32,
        contacts: &mut [Contact; 8],
    ) -> usize {
        let mut args = [bits(position), bits(a), bits(b)].concat();
        args.push(radius.to_bits());
        let r = self.event("collide", &args);
        for (c, w) in contacts.iter_mut().zip(r.as_chunks::<8>().0) {
            *c = Contact {
                shape: w[0],
                plane: Plane {
                    normal: vector(w, 1),
                    offset: scalar(w, 4),
                },
                point: vector(w, 5),
            };
        }
        r.len() / 8
    }
    fn cast(&mut self, position: Vec3, delta: Vec3) -> f32 {
        scalar(
            &self.event("cast", &[bits(position), bits(delta)].concat()),
            0,
        )
    }
    fn body(&mut self, shape: u32) -> Body {
        let r = self.event("body", &[shape]);
        Body {
            id: r[0],
            dynamic: r[1] != 0,
            awake: r[2] != 0,
            center: vector(&r, 3),
            linear: vector(&r, 6),
            angular: vector(&r, 9),
            inv_mass: scalar(&r, 12),
            inv_inertia: Mat3 {
                cx: vector(&r, 13),
                cy: vector(&r, 16),
                cz: vector(&r, 19),
            },
        }
    }
    fn impulse(&mut self, body: u32, impulse: Vec3, point: Vec3) {
        let mut args = vec![body];
        args.extend(bits(impulse));
        args.extend(bits(point));
        assert!(self.event("impulse", &args).is_empty());
    }
}
#[test]
fn sample_character_matches_native_queries_motion_and_impulses() {
    assert_eq!(core::mem::size_of::<Input>(), 112);
    assert_eq!(core::mem::offset_of!(Input, max_slope), 80);
    assert_eq!(core::mem::offset_of!(Input, rotation), 88);
    assert_eq!(core::mem::offset_of!(Input, max_speed), 104);
    let cases: Value = serde_json::from_str(include_str!("character.gold.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let w = words(&case["input"]);
        let input = Input {
            position: vector(&w, 0),
            velocity: vector(&w, 3),
            up: vector(&w, 6),
            center1: vector(&w, 9),
            center2: vector(&w, 12),
            radius: scalar(&w, 15),
            dt: scalar(&w, 16),
            pogo: scalar(&w, 17),
            rest: scalar(&w, 18),
            steep: w[19],
            max_slope: f64::from_bits(w[20] as u64 | ((w[21] as u64) << 32)),
            rotation: Quat {
                v: vector(&w, 25),
                s: scalar(&w, 28),
            },
            max_speed: scalar(&w, 29),
        };
        let name = case["name"].as_str().unwrap().to_owned();
        let mut world = Transcript {
            name: name.clone(),
            events: case["events"].as_array().unwrap().clone().into(),
        };
        let out = character::run(&mut world, &input);
        assert!(world.events.is_empty(), "{name}: unused native events");
        assert_eq!(core::mem::size_of::<Output>(), 63 * 4);
        // Output is repr(C), contains only initialized f32/u32 lanes and has no padding.
        let actual =
            unsafe { core::slice::from_raw_parts((&out as *const Output).cast::<u32>(), 63) };
        assert_eq!(actual, words(&case["output"]), "{name}");
    }
}
