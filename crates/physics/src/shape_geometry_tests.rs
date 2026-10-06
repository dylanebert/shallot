use crate::{
    math::Vec3,
    shape_geometry::{capsule_mass, sphere_mass, MassData},
};

fn check(m: MassData, g: &serde_json::Value) {
    let mut values = vec![m.mass, m.center.x, m.center.y, m.center.z];
    for c in [m.inertia.cx, m.inertia.cy, m.inertia.cz] {
        values.extend([c.x, c.y, c.z]);
    }
    let mut gold = vec![g["mass"].as_str().unwrap()];
    gold.extend(
        g["center"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap()),
    );
    gold.extend(
        g["inertia"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap()),
    );
    for (value, expected) in values.into_iter().zip(gold) {
        assert_eq!(
            value.to_bits(),
            u32::from_str_radix(expected.trim_start_matches("0x"), 16).unwrap()
        );
    }
}
#[test]
fn shape_mass_matches_native_geometry_gold() {
    let gold: serde_json::Value = serde_json::from_str(include_str!(
        "../../../src/standard/physics/shapes/geometry.gold.json"
    ))
    .unwrap();
    let sphere = gold["spheres"].as_array().unwrap();
    check(sphere_mass(Vec3::ZERO, 1.0, 1.0), &sphere[0]);
    check(
        sphere_mass(Vec3::new(0.5, -1.0, 2.0), 0.35, 2.5),
        &sphere[1],
    );
    let capsule = gold["capsules"].as_array().unwrap();
    check(
        capsule_mass(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            1.0,
        ),
        &capsule[0],
    );
    check(
        capsule_mass(
            Vec3::new(-1.0, 0.5, 0.25),
            Vec3::new(1.5, -0.5, 0.75),
            0.3,
            3.0,
        ),
        &capsule[1],
    );
    check(
        capsule_mass(
            Vec3::new(0.06, 0.0, 0.0),
            Vec3::new(-0.06, 0.0, 0.0),
            0.12,
            1000.0,
        ),
        &capsule[2],
    );
}
