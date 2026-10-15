use shallot_physics::math::*;

fn near(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "{a} != {b}");
}
fn vector_near(a: Vec3, b: Vec3, tolerance: f32) {
    near(a.x, b.x, tolerance);
    near(a.y, b.y, tolerance);
    near(a.z, b.z, tolerance);
}

#[test]
fn normalized_vector_has_unit_length() {
    near(
        Vec3::new(0.2, -0.5, 3.0).normalize().length(),
        1.0,
        4.0 * FLT_EPSILON,
    );
}

#[test]
fn inverse_transform_undoes_rotation_and_translation() {
    let t = Transform {
        p: Vec3::new(3.0, -5.0, 2.0),
        q: Quat::from_axis_angle(Vec3::new(0.3, -0.7, 0.5).normalize(), 0.4),
    };
    let v = Vec3::new(0.5, -0.25, 1.5);
    vector_near(t.inv_point(t.point(v)), v, 1e-5);
}

#[test]
fn matrix_inverse_and_solve_agree() {
    let m = Mat3 {
        cx: Vec3::new(3.0, 1.0, -1.0),
        cy: Vec3::new(-1.0, 3.0, 1.0),
        cz: Vec3::new(1.0, -1.0, 3.0),
    };
    let id = m.mul(m.invert());
    vector_near(id.cx, Vec3::new(1.0, 0.0, 0.0), 2.0 * FLT_EPSILON);
    vector_near(id.cy, Vec3::new(0.0, 1.0, 0.0), 2.0 * FLT_EPSILON);
    vector_near(id.cz, Vec3::new(0.0, 0.0, 1.0), 2.0 * FLT_EPSILON);
    let v = Vec3::new(1.0, -2.0, 3.0);
    vector_near(m.invert().mul_v(v), m.solve(v), FLT_EPSILON);
}

#[test]
fn relative_quaternion_is_a_left_inverse() {
    let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -0.5 * PI);
    let r = Quat {
        v: Vec3::new(1.0, -2.0, 3.0),
        s: 4.0,
    }
    .normalize();
    let result = r.mul(r.inv_mul(q));
    vector_near(q.v, result.v, FLT_EPSILON);
    near(q.s, result.s, FLT_EPSILON);
}

#[test]
fn quaternion_between_unit_vectors_reaches_target() {
    let a = Vec3::new(0.2, -0.5, 3.0).normalize();
    let b = Vec3::new(-0.3, 0.8, 0.1).normalize();
    let q = compute_quat_between_unit_vectors(a, b);
    assert!(q.s.is_finite() && q.v.x.is_finite() && q.v.y.is_finite() && q.v.z.is_finite());
    vector_near(q.rotate(a), b, 0.001);
}

#[test]
fn nlerp_twist_stays_within_one_degree_of_quarter_turn_interpolation() {
    let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.5 * PI);
    for i in 0..=20 {
        let alpha = i as f32 / 20.0;
        near(
            Quat::IDENTITY.nlerp(q, alpha).get_twist_angle(),
            alpha * 0.5 * PI,
            PI / 180.0,
        );
    }
}

#[test]
fn arbitrary_perpendicular_is_orthogonal() {
    let n = Vec3::new(0.50405544, 0.62154806, 0.59967154);
    near(n.dot(arbitrary_perp(n)), 0.0, 2.0 * FLT_EPSILON);
}

#[test]
fn ordered_select_min_max_preserve_nan_and_signed_zero_behavior() {
    assert_eq!(minf(f32::NAN, 5.0), 5.0);
    assert_eq!(minf(-0.0, 0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(maxf(f32::NAN, 5.0), 5.0);
}

#[test]
fn point_segment_clamps_beyond_either_endpoint() {
    let a = Vec3::ZERO;
    let b = Vec3::new(2.0, 0.0, 0.0);
    vector_near(
        point_to_segment_distance(a, b, Vec3::new(1.0, 5.0, 0.0)),
        Vec3::new(1.0, 0.0, 0.0),
        FLT_EPSILON,
    );
    assert_eq!(
        point_to_segment_distance(a, b, Vec3::new(-3.0, 1.0, 0.0)),
        a
    );
    assert_eq!(point_to_segment_distance(a, b, Vec3::new(9.0, 1.0, 0.0)), b);
}
