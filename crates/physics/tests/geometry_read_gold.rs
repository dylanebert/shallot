use serde_json::Value;
use shallot_physics::geometry_read::read;

fn check(operation: u32) {
    let cases: Value = serde_json::from_str(include_str!("geometry_read.gold.json")).unwrap();
    let mut count = 0;
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        if c["operation"].as_u64().unwrap() as u32 != operation {
            continue;
        }
        let words = |key: &str| {
            c[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            read(operation, &words("input")).as_slice(),
            words("output"),
            "native case {i}, reader {operation}"
        );
        count += 1;
    }
    assert!(count > 0);
}
#[test]
fn sphere_aabb() {
    check(0);
}
#[test]
fn capsule_aabb() {
    check(1);
}
#[test]
fn hull_aabb() {
    check(2);
}
#[test]
fn mesh_aabb() {
    check(3);
}
#[test]
fn safe_scale() {
    check(4);
}
#[test]
fn mesh_triangle() {
    check(5);
}
#[test]
fn height_triangle() {
    check(6);
}
