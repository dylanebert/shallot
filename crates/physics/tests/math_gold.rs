//! Bit-exact scalar math parity against frozen native Box3D vectors and the immutable v6 corpus.

use serde_json::Value;
use shallot_physics::math::*;

const GOLD: &str = include_str!("math.gold.json");

#[test]
fn point_to_segment_box3d_corpus() {
    let corpus: Value = serde_json::from_str(include_str!("../../../src/standard/physics/oracle/box3d/47d7f7cc7e091142c08d11dc7d2e493c5d34f536/v6/cases.json")).unwrap();
    let mut count = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if case["symbol"] != "b3PointToSegmentDistance" {
            continue;
        }
        count += 1;
        let input = &case["input"];
        let vec = |name: &str| {
            let a = input[name].as_array().unwrap();
            Vec3::new(
                from_bits(a[0].as_str().unwrap()),
                from_bits(a[1].as_str().unwrap()),
                from_bits(a[2].as_str().unwrap()),
            )
        };
        let result = point_to_segment_distance(vec("a"), vec("b"), vec("q"));
        for (i, value) in [result.x, result.y, result.z].iter().enumerate() {
            assert_bits(
                *value,
                case["output"]["closest"][i].as_str().unwrap(),
                case["id"].as_str().unwrap(),
            );
        }
    }
    assert_eq!(count, 2);
}

#[test]
fn math_box3d_corpus() {
    let corpus: Value = serde_json::from_str(include_str!("../../../src/standard/physics/oracle/box3d/47d7f7cc7e091142c08d11dc7d2e493c5d34f536/v6/cases.json")).unwrap();
    let mut count = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if case["family"] != "math" {
            continue;
        }
        count += 1;
        let input = &case["input"];
        let output = &case["output"];
        let label = case["id"].as_str().unwrap();
        let vec = |name: &str| {
            let a = input[name].as_array().unwrap();
            Vec3::new(
                from_bits(a[0].as_str().unwrap()),
                from_bits(a[1].as_str().unwrap()),
                from_bits(a[2].as_str().unwrap()),
            )
        };
        match case["symbol"].as_str().unwrap() {
            "b3Add" => {
                let v = vec("a").add(vec("b"));
                for (i, n) in [v.x, v.y, v.z].iter().enumerate() {
                    assert_bits(*n, output["value"][i].as_str().unwrap(), label);
                }
            }
            "b3Dot" => assert_bits(
                vec("a").dot(vec("b")),
                output["value"].as_str().unwrap(),
                label,
            ),
            "b3ComputeCosSin" => {
                let cs = compute_cos_sin(from_bits(input["radians"].as_str().unwrap()));
                assert_bits(cs.cosine, output["cosine"].as_str().unwrap(), label);
                assert_bits(cs.sine, output["sine"].as_str().unwrap(), label);
            }
            symbol => panic!("unmapped math symbol: {symbol}"),
        }
    }
    assert_eq!(count, 6);
}

// Reconstruct an exact f32 from the raw hex bits the C generator emitted — no decimal round-trip.
fn from_bits(hex: &str) -> f32 {
    let bits = u32::from_str_radix(hex.trim_start_matches("0x"), 16).expect("hex");
    f32::from_bits(bits)
}

// Bit-exact: compare raw bit patterns so ±0 and NaN payloads count as divergence (like Object.is).
fn assert_bits(got: f32, want: &str, label: &str) {
    let want_bits = u32::from_str_radix(want.trim_start_matches("0x"), 16).expect("hex");
    assert_eq!(
        got.to_bits(),
        want_bits,
        "{label}: got 0x{:08x} ({got}), want {want} ({})",
        got.to_bits(),
        from_bits(want),
    );
}

// --- flat reconstruct / flatten, matching the C generator's field order ----------------------

fn v3(a: &[f32], o: usize) -> Vec3 {
    Vec3::new(a[o], a[o + 1], a[o + 2])
}
fn quat(a: &[f32], o: usize) -> Quat {
    Quat {
        v: v3(a, o),
        s: a[o + 3],
    }
}
fn mat3(a: &[f32], o: usize) -> Mat3 {
    Mat3 {
        cx: v3(a, o),
        cy: v3(a, o + 3),
        cz: v3(a, o + 6),
    }
}
fn mat2(a: &[f32], o: usize) -> Mat2 {
    Mat2 {
        cx: Vec2::new(a[o], a[o + 1]),
        cy: Vec2::new(a[o + 2], a[o + 3]),
    }
}
fn xf(a: &[f32], o: usize) -> Transform {
    Transform {
        p: v3(a, o),
        q: quat(a, o + 3),
    }
}

fn flat_v(v: Vec3) -> Vec<f32> {
    vec![v.x, v.y, v.z]
}
fn flat_v2(v: Vec2) -> Vec<f32> {
    vec![v.x, v.y]
}
fn flat_q(q: Quat) -> Vec<f32> {
    vec![q.v.x, q.v.y, q.v.z, q.s]
}
fn flat_m(m: Mat3) -> Vec<f32> {
    vec![
        m.cx.x, m.cx.y, m.cx.z, m.cy.x, m.cy.y, m.cy.z, m.cz.x, m.cz.y, m.cz.z,
    ]
}
fn flat_m2(m: Mat2) -> Vec<f32> {
    vec![m.cx.x, m.cx.y, m.cy.x, m.cy.y]
}
fn flat_t(t: Transform) -> Vec<f32> {
    vec![t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s]
}

// Native generator field order.
fn dispatch(name: &str, a: &[f32]) -> Vec<f32> {
    match name {
        "normalizeVec3" => flat_v(v3(a, 0).normalize()),
        "lengthVec3" => vec![v3(a, 0).length()],
        "arbitraryPerp" => flat_v(arbitrary_perp(v3(a, 0))),
        "scalarTripleProduct" => vec![scalar_triple_product(v3(a, 0), v3(a, 3), v3(a, 6))],
        "normalizeQuat" => flat_q(quat(a, 0).normalize()),
        "makeQuatFromAxisAngle" => flat_q(Quat::from_axis_angle(v3(a, 0), a[3])),
        "mulQuat" => flat_q(quat(a, 0).mul(quat(a, 4))),
        "invMulQuat" => flat_q(quat(a, 0).inv_mul(quat(a, 4))),
        "rotateVector" => flat_v(quat(a, 0).rotate(v3(a, 4))),
        "invRotateVector" => flat_v(quat(a, 0).inv_rotate(v3(a, 4))),
        "computeQuatBetween" | "computeQuatBetweenAntiparallel" => {
            flat_q(compute_quat_between_unit_vectors(v3(a, 0), v3(a, 3)))
        }
        "makeQuatFromMatrix" => flat_q(make_quat_from_matrix(&mat3(a, 0))),
        "makeMatrixFromQuat" => flat_m(Mat3::from_quat(quat(a, 0))),
        "invertMatrix" => flat_m(mat3(a, 0).invert()),
        "solve3" => flat_v(mat3(a, 0).solve(v3(a, 9))),
        "mulMV" => flat_v(mat3(a, 0).mul_v(v3(a, 9))),
        "mulMM" => flat_m(mat3(a, 0).mul(mat3(a, 9))),
        "invert2" => flat_m2(mat2(a, 0).invert()),
        "solve2" => flat_v2(mat2(a, 0).solve(Vec2::new(a[4], a[5]))),
        "nlerp" => flat_q(quat(a, 0).nlerp(quat(a, 4), a[8])),
        "getTwistAngle" => vec![quat(a, 0).get_twist_angle()],
        "getSwingAngle" => vec![quat(a, 0).get_swing_angle()],
        "transformPoint" => flat_v(xf(a, 0).point(v3(a, 7))),
        "invTransformPoint" => flat_v(xf(a, 0).inv_point(v3(a, 7))),
        "mulTransforms" => flat_t(xf(a, 0).mul(xf(a, 7))),
        // Gold `in` is [t2, t1]; the reference computed b3InvMulTransforms(t1, t2), so swap.
        "invMulTransforms" => flat_t(xf(a, 7).inv_mul(xf(a, 0))),
        other => panic!("unmapped gold case fn: {other}"),
    }
}

fn gold() -> Value {
    serde_json::from_str(GOLD).expect("parse math.gold.json")
}

#[test]
fn atan2_sweep() {
    let g = gold();
    for row in g["atan2"].as_array().unwrap() {
        let y = from_bits(row[0].as_str().unwrap());
        let x = from_bits(row[1].as_str().unwrap());
        let want = row[2].as_str().unwrap();
        assert_bits(atan2(y, x), want, "atan2");
    }
}

#[test]
fn compute_cos_sin_sweep() {
    let g = gold();
    for row in g["cosSin"].as_array().unwrap() {
        let angle = from_bits(row[0].as_str().unwrap());
        let cs = compute_cos_sin(angle);
        assert_bits(cs.cosine, row[1].as_str().unwrap(), "cos");
        assert_bits(cs.sine, row[2].as_str().unwrap(), "sin");
    }
}

#[test]
fn vec_quat_matrix_transform_cases() {
    let g = gold();
    for case in g["cases"].as_array().unwrap() {
        let name = case["fn"].as_str().unwrap();
        let input: Vec<f32> = case["in"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| from_bits(h.as_str().unwrap()))
            .collect();
        let out = case["out"].as_array().unwrap();
        let got = dispatch(name, &input);
        assert_eq!(got.len(), out.len(), "{name}: output arity");
        for (i, want) in out.iter().enumerate() {
            assert_bits(got[i], want.as_str().unwrap(), &format!("{name}[{i}]"));
        }
    }
}
