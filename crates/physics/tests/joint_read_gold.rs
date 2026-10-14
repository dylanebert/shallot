use serde_json::Value;
use shallot_physics::{
    col::Col,
    joint_read,
    math::{Quat, Transform, Vec3},
};
fn words(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect()
}
fn pose(v: &Value) -> Transform {
    let p = words(v);
    Transform {
        p: Vec3::new(p[0], p[1], p[2]),
        q: Quat {
            v: Vec3::new(p[3], p[4], p[5]),
            s: p[6],
        },
    }
}
fn check(operation: usize) {
    let gold: Value = serde_json::from_str(include_str!("joint_read.gold.json")).unwrap();
    let mut count = 0;
    for c in gold.as_array().unwrap() {
        if c["output"][operation].is_null() {
            continue;
        }
        let mut memory = words(&c["joint"]);
        let col = unsafe { Col::of(&mut memory) };
        let actual = joint_read::read(col, pose(&c["a"]), pose(&c["b"]), operation as u32);
        assert_eq!(
            actual.to_bits(),
            c["output"][operation].as_u64().unwrap() as u32,
            "reader {operation}, kind {}, sample {}",
            c["kind"],
            c["sample"]
        );
        count += 1;
    }
    assert!(count > 0);
}
#[test]
fn revolute_angle() {
    check(0);
}
#[test]
fn cone_angle() {
    check(1);
}
#[test]
fn twist_angle() {
    check(2);
}
#[test]
fn steering_angle() {
    check(3);
}
#[test]
fn angular_separation() {
    check(4);
}
#[test]
fn linear_separation() {
    check(5);
}
