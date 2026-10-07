//! Bit-exact parity of the Rust scalar contact solver against frozen historical oracle vectors,
//! read from the committed gold vectors (`src/contact.gold.json`). Current target evidence is
//! produced by the standalone oracle; this drives the Rust port over the same shared columns and
//! asserts every output float bit-for-bit.

use serde_json::Value;
use shallot_physics::body::{SIM_STRIDE, STATE_STRIDE};
use shallot_physics::col::Col;
use shallot_physics::contact::{
    prepare, restitution, solve, store, warm_start, Columns, ContactConstraint, ManifoldConstraint,
    Softness, NULL_INDEX,
};
use shallot_physics::contact_spans::{ContactPrepareSpan, ContactSpec};
use shallot_physics::manifold_abi::*;
const MCP_STRIDE: usize = 12;

/// SAFETY: a gold harness is single-threaded and each column has exactly one user, so `Col`'s
/// disjoint-write promise holds trivially.
fn col<T: Copy>(v: &mut [T]) -> Col<'_, T> {
    unsafe { Col::of(v) }
}

const GOLD: &str = include_str!("../../../src/standard/physics/collision/contact.gold.json");

fn from_bits(hex: &str) -> f32 {
    let bits = u32::from_str_radix(hex.trim_start_matches("0x"), 16).expect("hex");
    f32::from_bits(bits)
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| from_bits(e.as_str().expect("hex str")))
        .collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| e.as_u64().expect("u64") as u32)
        .collect()
}

fn soft(v: &Value) -> Softness {
    let a = floats(v);
    Softness {
        bias_rate: a[0],
        mass_scale: a[1],
        impulse_scale: a[2],
    }
}

fn assert_bits(got: f32, want: f32, label: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}: got 0x{:08x} ({got}), want 0x{:08x} ({want})",
        got.to_bits(),
        want.to_bits(),
    );
}

fn sim_map() -> Vec<usize> {
    [
        vec![26, 51, 49, 50, 20, 21, 22, 23, 24, 25],
        (27..36).collect(),
        (36..45).collect(),
        (3..7).collect(),
    ]
    .concat()
}

fn repack(input: &[f32], old_stride: usize, stride: usize, map: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0; input.len() / old_stride * stride];
    for (i, record) in input.chunks_exact(old_stride).enumerate() {
        for (j, &offset) in map.iter().enumerate() {
            out[i * stride + offset] = record[j];
        }
    }
    out
}

fn observe(
    c: &ContactConstraint,
    manifolds: &[ManifoldConstraint],
) -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    use shallot_physics::math::{Mat3, Vec3};
    fn v(v: Vec3) -> Vec<f32> {
        vec![v.x, v.y, v.z]
    }
    fn m(m: Mat3) -> Vec<f32> {
        [v(m.cx), v(m.cy), v(m.cz)].concat()
    }
    let cc = [
        vec![c.inv_mass_a, c.inv_mass_b],
        m(c.inv_ia),
        m(c.inv_ib),
        m(c.rolling_mass),
        vec![
            c.softness.bias_rate,
            c.softness.mass_scale,
            c.softness.impulse_scale,
            c.friction,
            c.restitution,
            c.rolling_resistance,
        ],
    ]
    .concat();
    let meta = vec![c.index_a, c.index_b, c.manifold_count as u32, 0];
    let mut mc = Vec::new();
    let mut mc_meta = Vec::new();
    let mut points = Vec::new();
    for a in manifolds {
        mc.extend(
            [
                v(a.normal),
                v(a.tangent1),
                v(a.tangent2),
                vec![
                    a.tangent_mass.cx.x,
                    a.tangent_mass.cx.y,
                    a.tangent_mass.cy.x,
                    a.tangent_mass.cy.y,
                    a.friction_impulse.x,
                    a.friction_impulse.y,
                    a.twist_mass,
                    a.twist_impulse,
                ],
                v(a.rolling_impulse),
                vec![a.tangent_velocity1, a.tangent_velocity2],
                v(a.center_a),
                v(a.center_b),
            ]
            .concat(),
        );
        mc_meta.extend([a.point_count as u32, (points.len() / MCP_STRIDE) as u32]);
        for p in &a.points[..a.point_count as usize] {
            points.extend(
                [
                    v(p.r_a),
                    v(p.r_b),
                    vec![
                        p.base_separation,
                        p.normal_impulse,
                        p.total_normal_impulse,
                        p.normal_mass,
                        p.relative_velocity,
                        p.lever_arm,
                    ],
                ]
                .concat(),
            );
        }
    }
    (cc, meta, mc, mc_meta, points)
}

#[test]
fn contact_phases_match_c() {
    let gold: Value = serde_json::from_str(GOLD).expect("gold json");
    let cases = gold["prepare"].as_array().expect("prepare");
    assert!(!cases.is_empty());

    for (c, case) in cases.iter().enumerate() {
        let n = case["pointCount"].as_u64().expect("pointCount") as usize;
        let warm = from_bits(case["warmStartScale"].as_str().expect("warmStartScale"));
        let contact_softness = soft(&case["contactSoftness"]);
        let static_softness = soft(&case["staticSoftness"]);

        let mut state = repack(
            &floats(&case["state"]),
            16,
            STATE_STRIDE,
            &(0..13).collect::<Vec<_>>(),
        );
        let mut sim = repack(&floats(&case["sim"]), 32, SIM_STRIDE, &sim_map());
        // Both bodies flagged dynamic so warm_start writes velocities back; a static body B is
        // reached via NULL_INDEX (its flag slot is never consulted).
        let mut flags = vec![0; 2 * STATE_STRIDE];
        flags[0] = shallot_physics::body::flags::DYNAMIC;
        flags[STATE_STRIDE] = shallot_physics::body::flags::DYNAMIC;

        // The gold input is in the old IN_* handoff layout; repack it into the column-resident store
        // shapes the solver now gathers through: one scalar slot record → contactId 0, and contactId
        // 0's directory record (material + indices + block descriptor) + pool manifolds.
        let in_contact = floats(&case["inContact"]);
        let in_contact_meta = u32s(&case["inContactMeta"]);
        let in_manifold = floats(&case["inManifold"]);
        let in_manifold_meta = u32s(&case["inManifoldMeta"]);
        let in_point = floats(&case["inPoint"]);
        let index_a = in_contact_meta[0];
        let index_b = in_contact_meta[1];
        let manifold_count = in_contact_meta[2] as usize;

        let specs = [ContactSpec {
            contact_id: 0,
            manifold_start: 0,
            manifold_count: manifold_count as u16,
        }];
        let mut spans = [
            ContactPrepareSpan {
                start: 0,
                count: 1,
                contacts: specs.as_ptr(),
            },
            ContactPrepareSpan {
                start: i32::MAX,
                count: 0,
                contacts: std::ptr::null(),
            },
        ];

        // directory record for contactId 0.
        let mut dir = vec![0u32; DIR_STRIDE];
        dir[DIR_FRICTION] = in_contact[0].to_bits();
        dir[DIR_RESTITUTION] = in_contact[1].to_bits();
        dir[DIR_ROLLING_RESISTANCE] = in_contact[2].to_bits();
        for k in 0..3 {
            dir[DIR_TANGENT_VELOCITY + k] = in_contact[3 + k].to_bits();
        }
        dir[DIR_FLAGS] = in_contact_meta[4];
        dir[DIR_MANIFOLD_COUNT] = manifold_count as u32;
        dir[DIR_MANIFOLD_BASE] = 0;
        dir[DIR_INDEX_A] = index_a;
        dir[DIR_INDEX_B] = index_b;

        // pool: the manifolds as b3Manifold records (header + inline points).
        let mut pool = vec![0.0f32; manifold_count.max(1) * MANIFOLD_STRIDE];
        for m in 0..manifold_count {
            let imo = m * 10; // old IN_MANIFOLD_STRIDE
            let pc = in_manifold_meta[m * 2] as usize;
            let ps = in_manifold_meta[m * 2 + 1] as usize;
            let mpo = m * MANIFOLD_STRIDE;
            for (k, offset) in [
                M_NORMAL,
                M_NORMAL + 1,
                M_NORMAL + 2,
                M_FRICTION,
                M_FRICTION + 1,
                M_FRICTION + 2,
                M_TWIST,
                M_ROLLING,
                M_ROLLING + 1,
                M_ROLLING + 2,
            ]
            .into_iter()
            .enumerate()
            {
                pool[mpo + offset] = in_manifold[imo + k];
            }
            pool[mpo + M_POINT_COUNT] = f32::from_bits(pc as u32); // pointCount
            for p in 0..pc {
                let ipo = (ps + p) * 10; // old IN_POINT_STRIDE
                let pp = mpo + M_POINTS + p * POOL_POINT_STRIDE;
                pool[pp] = in_point[ipo]; // anchorA
                pool[pp + 1] = in_point[ipo + 1];
                pool[pp + 2] = in_point[ipo + 2];
                pool[pp + 3] = in_point[ipo + 3]; // anchorB
                pool[pp + 4] = in_point[ipo + 4];
                pool[pp + 5] = in_point[ipo + 5];
                pool[pp + 6] = in_point[ipo + 6]; // separation
                pool[pp + 8] = in_point[ipo + 7]; // normalImpulse (IN_POINT 7 → pool 8)
            }
        }

        // Zero is valid for these plain numeric records and raw pointers; prepare initializes
        // every active field and pointer before any solver phase reads it.
        let mut cc_records = vec![unsafe { std::mem::zeroed::<ContactConstraint>() }; 1];
        let mut mc_records =
            vec![unsafe { std::mem::zeroed::<ManifoldConstraint>() }; manifold_count.max(1)];

        let cols = Columns {
            state: col(&mut state),
            flags: col(&mut flags),
            sim: col(&mut sim),
            spans: col(&mut spans),
            dir: col(&mut dir),
            pool: col(&mut pool),
            cc: col(&mut cc_records),
            mc: col(&mut mc_records),
        };

        // Run both phases before reading columns back: consecutive `cols` uses keep the borrow
        // checker happy, and warm_start only mutates the state column, leaving prepare's outputs.
        prepare(&cols, 0, 1, contact_softness, static_softness, warm);
        warm_start(&cols, 0, 1);

        let (cc, cc_meta, mc, mc_meta, mcp) =
            observe(&cc_records[0], &mc_records[..manifold_count]);
        assert_eq!(cc_records[0].constraints, mc_records.as_mut_ptr());
        let out_cc = floats(&case["outCc"]);
        for (k, want) in out_cc.iter().enumerate() {
            assert_bits(cc[k], *want, &format!("prep[{c}].cc[{k}]"));
        }
        assert_eq!(cc_meta, u32s(&case["outCcMeta"]), "prep[{c}].ccMeta");

        let out_mc = floats(&case["outMc"]);
        for (k, want) in out_mc.iter().enumerate() {
            assert_bits(mc[k], *want, &format!("prep[{c}].mc[{k}]"));
        }
        assert_eq!(mc_meta, u32s(&case["outMcMeta"]), "prep[{c}].mcMeta");

        let out_mcp = floats(&case["outMcp"]);
        for (k, want) in out_mcp.iter().enumerate() {
            assert_bits(mcp[k], *want, &format!("prep[{c}].mcp[{k}]"));
        }

        // warm_start wrote velocities back into the state column. Body A is index 0; body B is
        // index 1 when the contact has a real (non-static) B.
        let ws_a = floats(&case["outWsA"]);
        for k in 0..6 {
            assert_bits(state[k], ws_a[k], &format!("ws[{c}].A[{k}]"));
        }
        if index_b != NULL_INDEX {
            let ws_b = floats(&case["outWsB"]);
            let o = index_b as usize * STATE_STRIDE;
            for k in 0..6 {
                assert_bits(state[o + k], ws_b[k], &format!("ws[{c}].B[{k}]"));
            }
        }

        // solve(bias) then relax(no bias), chained onto the warm-started state. Re-bundle the
        // columns (the prior borrow ended above) so solve mutates the same in-place data.
        let inv_h = from_bits(case["invH"].as_str().expect("invH"));
        let contact_speed = from_bits(case["contactSpeed"].as_str().expect("contactSpeed"));
        {
            let cols = Columns {
                state: col(&mut state),
                flags: col(&mut flags),
                sim: col(&mut sim),
                spans: col(&mut spans),
                dir: col(&mut dir),
                pool: col(&mut pool),
                cc: col(&mut cc_records),
                mc: col(&mut mc_records),
            };
            solve(&cols, 0, 1, true, inv_h, contact_speed);
            solve(&cols, 0, 1, false, inv_h, contact_speed);
        }

        let slv_a = floats(&case["outSlvA"]);
        for k in 0..6 {
            assert_bits(state[k], slv_a[k], &format!("slv[{c}].A[{k}]"));
        }
        if index_b != NULL_INDEX {
            let slv_b = floats(&case["outSlvB"]);
            let o = index_b as usize * STATE_STRIDE;
            for k in 0..6 {
                assert_bits(state[o + k], slv_b[k], &format!("slv[{c}].B[{k}]"));
            }
        }

        let (_, _, mc, _, mcp) = observe(&cc_records[0], &mc_records[..manifold_count]);
        // Accumulated impulses in the transient records after both passes.
        let out_ni = floats(&case["outNormalImpulse"]);
        let out_tni = floats(&case["outTotalNormalImpulse"]);
        for p in 0..n {
            assert_bits(
                mcp[p * MCP_STRIDE + 7],
                out_ni[p],
                &format!("slv[{c}].normalImpulse[{p}]"),
            );
            assert_bits(
                mcp[p * MCP_STRIDE + 8],
                out_tni[p],
                &format!("slv[{c}].totalNormalImpulse[{p}]"),
            );
        }
        assert_bits(
            mc[16],
            from_bits(case["outTwistImpulse"].as_str().unwrap()),
            &format!("slv[{c}].twist"),
        );
        let out_rolling = floats(&case["outRollingImpulse"]);
        for k in 0..3 {
            assert_bits(
                mc[17 + k],
                out_rolling[k],
                &format!("slv[{c}].rolling[{k}]"),
            );
        }
        let out_friction = floats(&case["outFrictionImpulse"]);
        assert_bits(mc[13], out_friction[0], &format!("slv[{c}].friction.x"));
        assert_bits(mc[14], out_friction[1], &format!("slv[{c}].friction.y"));

        let mut hit = 0;
        // restitution then store, chained onto the solved state.
        let rest_threshold = from_bits(case["restThreshold"].as_str().expect("restThreshold"));
        let hit_threshold = from_bits(case["hitThreshold"].as_str().expect("hitThreshold"));
        {
            let cols = Columns {
                state: col(&mut state),
                flags: col(&mut flags),
                sim: col(&mut sim),
                spans: col(&mut spans),
                dir: col(&mut dir),
                pool: col(&mut pool),
                cc: col(&mut cc_records),
                mc: col(&mut mc_records),
            };
            restitution(&cols, 0, 1, rest_threshold);
            store(&cols, 0, 1, hit_threshold, |_| {
                hit = 1;
            });
        }

        let rest_a = floats(&case["outRestA"]);
        for k in 0..6 {
            assert_bits(state[k], rest_a[k], &format!("rest[{c}].A[{k}]"));
        }
        if index_b != NULL_INDEX {
            let rest_b = floats(&case["outRestB"]);
            let o = index_b as usize * STATE_STRIDE;
            for k in 0..6 {
                assert_bits(state[o + k], rest_b[k], &format!("rest[{c}].B[{k}]"));
            }
        }

        // store wrote the solved impulses straight back into the persistent pool manifold (record 0
        // — the gold is single-manifold) and the hit flag into the directory.
        let stored_friction = floats(&case["storedFrictionImpulse"]);
        for k in 0..3 {
            assert_bits(
                pool[M_FRICTION + k],
                stored_friction[k],
                &format!("store[{c}].friction[{k}]"),
            );
        }
        assert_bits(
            pool[M_TWIST],
            from_bits(case["storedTwistImpulse"].as_str().unwrap()),
            &format!("store[{c}].twist"),
        );
        let stored_rolling = floats(&case["storedRollingImpulse"]);
        for k in 0..3 {
            assert_bits(
                pool[M_ROLLING + k],
                stored_rolling[k],
                &format!("store[{c}].rolling[{k}]"),
            );
        }
        let stored_ni = floats(&case["storedNormalImpulse"]);
        let stored_tni = floats(&case["storedTotalNormalImpulse"]);
        let stored_nv = floats(&case["storedNormalVelocity"]);
        for p in 0..n {
            let pp = M_POINTS + p * POOL_POINT_STRIDE;
            assert_bits(pool[pp + 8], stored_ni[p], &format!("store[{c}].ni[{p}]"));
            assert_bits(pool[pp + 9], stored_tni[p], &format!("store[{c}].tni[{p}]"));
            assert_bits(pool[pp + 10], stored_nv[p], &format!("store[{c}].nv[{p}]"));
        }
        assert_eq!(
            hit,
            case["hitFlag"].as_u64().expect("hitFlag") as u32,
            "store[{c}].hitFlag"
        );
    }
}
