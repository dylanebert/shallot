use serde_json::Value;
use shallot_physics::{
    col::Col,
    joint_read::{self, Motion},
    math::{Quat, Transform, Vec3},
};
fn words(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect()
}
fn motion(v: &Value) -> Motion {
    let p = words(v);
    let v = |i| Vec3::new(p[i], p[i + 1], p[i + 2]);
    Motion {
        pose: Transform {
            p: v(0),
            q: Quat { v: v(3), s: p[6] },
        },
        center: v(7),
        local_center: v(10),
        linear: v(13),
        angular: v(16),
    }
}
fn check(operation: usize) {
    let gold: Value = serde_json::from_str(include_str!("joint_motion.gold.json")).unwrap();
    let mut count = 0;
    for c in gold.as_array().unwrap() {
        if c["output"][operation - 6].is_null() {
            continue;
        }
        let mut memory = words(&c["joint"]);
        let col = unsafe { Col::of(&mut memory) };
        let a = motion(&c["a"]);
        let b = motion(&c["b"]);
        let actual = if operation < 8 {
            joint_read::read(col, a.pose, b.pose, operation as u32)
        } else {
            joint_read::speed(col, a, b, operation as u32)
        };
        assert_eq!(
            actual.to_bits(),
            c["output"][operation - 6].as_u64().unwrap() as u32,
            "reader {operation}, kind {}, mode {}, sample {}",
            c["kind"],
            c["mode"],
            c["sample"]
        );
        count += 1;
    }
    assert!(count > 0);
}
#[test]
fn prismatic_translation() {
    check(6);
}
#[test]
fn distance_length() {
    check(7);
}
#[test]
fn prismatic_speed() {
    check(8);
}
#[test]
fn wheel_spin_speed() {
    check(9);
}
