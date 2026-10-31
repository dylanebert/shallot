use super::*;

fn node(index: usize, parent: i32, x: f64) -> Node {
    Node {
        index,
        parent,
        pose: vec![x, 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.25],
    }
}
fn animation(mode: usize, path: usize, output: Vec<f64>) -> Animation {
    Animation {
        samplers: vec![Sampler {
            mode,
            input: vec![1., 3.],
            output,
        }],
        channels: vec![Channel {
            sampler: 0,
            node: 4,
            path,
        }],
    }
}
#[test]
fn step_clamps_and_retains_unchanneled_properties() {
    let nodes = [node(4, -1, 7.)];
    let clip = animation(0, 3, vec![2., 6.]);
    let mut poses = vec![0.; 11];
    for (time, expected) in [(-1., 2.), (1., 2.), (2., 2.), (3., 6.), (10., 6.)] {
        evaluate(&nodes, Some(&clip), time, &mut poses);
        assert_eq!(poses[10], expected);
        assert_eq!(poses[..10], nodes[0].pose[..10]);
    }
}
#[test]
fn linear_interpolates_and_ignores_channels_outside_scene() {
    let nodes = [node(4, -1, 7.)];
    let mut clip = animation(1, 0, vec![0., 2., 4., 4., 6., 8.]);
    clip.channels.push(Channel {
        sampler: 0,
        node: 99,
        path: 0,
    });
    let mut poses = vec![0.; 11];
    evaluate(&nodes, Some(&clip), 1.5, &mut poses);
    assert_eq!(poses[..3], [1., 3., 5.]);
}
#[test]
fn linear_rotation_uses_shortest_arc_and_near_equal_limit() {
    let nodes = [node(4, -1, 0.)];
    let mut poses = vec![0.; 11];
    let clip = animation(1, 1, vec![0., 0., 0., 1., 0., 0., -1., 0.]);
    evaluate(&nodes, Some(&clip), 2., &mut poses);
    assert!((poses[5] + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    assert!((poses[6] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    let clip = animation(1, 1, vec![0., 0., 0., 1., 0., 0., 0., -1.]);
    evaluate(&nodes, Some(&clip), 2., &mut poses);
    assert_eq!(&poses[3..7], &[0., 0., 0., 1.]);
}
#[test]
fn cubic_scales_tangents_by_duration_and_normalizes_only_interior_rotations() {
    let nodes = [node(4, -1, 0.)];
    let mut poses = vec![0.; 11];
    let clip = animation(2, 3, vec![0., 2., 4., 0., 6., 0.]);
    evaluate(&nodes, Some(&clip), 2., &mut poses);
    assert_eq!(poses[10], 5.);
    let clip = animation(
        2,
        1,
        vec![
            0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
            0., 0.,
        ],
    );
    evaluate(&nodes, Some(&clip), 2., &mut poses);
    assert!((poses[5] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    assert!((poses[6] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    evaluate(&nodes, Some(&clip), 1., &mut poses);
    assert_eq!(poses[6], 2.);
}
#[test]
fn globals_compose_parent_first_at_f32_boundaries() {
    let mut root = node(0, -1, 3.);
    root.pose[7] = 2.;
    let a = Asset {
        nodes: vec![root, node(1, 0, 4.)],
        skins: vec![],
        primitives: vec![],
    };
    let mut i = Instance {
        poses: a.nodes.iter().flat_map(|n| n.pose.clone()).collect(),
        globals: vec![0.; 32],
        input: vec![],
        output: vec![],
    };
    compose_globals(&a, &mut i);
    assert_eq!(i.globals[12], 3.);
    assert_eq!(i.globals[28], 11.);
    assert_eq!(i.globals[16], 2.);
}
#[test]
fn palette_and_sphere_include_inverse_binds_empty_joints_and_morph_reach() {
    let global = compose(&node(0, -1, 3.).pose);
    let mut bind = compose(&node(0, -1, -1.).pose);
    bind[0] = 2.;
    let skin = Skin {
        joints: vec![0, 0],
        binds: [bind, bind].concat(),
    };
    let boxes = [vec![-1., -1., -1., 1., 1., 1.], vec![f32::INFINITY; 6]].concat();
    let mut palette = [0.; 32];
    let mut bounds = [0.; 4];
    write_skin_pose(&skin, &boxes, &global, 0.5, &mut palette, &mut bounds);
    assert_eq!(palette[12], 2.);
    assert_eq!(&bounds[..3], &[2., 0., 0.]);
    assert_eq!(bounds[3], (libm::sqrt(36. + 9. + 9.) / 2.) as f32);
    let p = Primitive {
        pose_offset: 0,
        skin: -1,
        boxes: vec![],
        reach: vec![3.],
        bounds: vec![0.; 4],
    };
    let mut pose = node(0, -1, 0.).pose;
    pose[10] = -1. / 3.;
    let mut weights = [0.];
    let reach = write_morph_pose(&p, &pose, 1, &mut weights);
    assert_eq!(weights[0], (-1_f64 / 3.) as f32);
    assert_eq!(reach, f64::from(weights[0]).abs() * 3.);
}

#[test]
fn present_zero_weight_count_resets_unskinned_cull_bounds() {
    let a = Asset {
        nodes: vec![node(0, -1, 0.)],
        skins: vec![],
        primitives: vec![Primitive {
            pose_offset: 0,
            skin: -1,
            boxes: vec![],
            reach: vec![3.],
            bounds: vec![1., 2., 3., 4.],
        }],
    };
    let mut i = Instance {
        poses: vec![0.; 11],
        globals: vec![0.; 16],
        input: vec![0, 1],
        output: vec![99.; 5],
    };
    // The second input lane is zero for no MorphedMesh, otherwise weightCount + 1.
    unsafe {
        pose(
            (&a as *const Asset).cast(),
            (&mut i as *mut Instance).cast(),
            std::ptr::null(),
            0.,
        );
    }
    assert_eq!(&i.output[..4], &[1., 2., 3., 4.]);
}

#[test]
fn standalone_evaluation_does_not_require_parent_first_nodes() {
    let a = Asset {
        nodes: vec![node(4, 1, 7.), node(5, -1, 0.)],
        skins: vec![],
        primitives: vec![],
    };
    let clip = animation(1, 0, vec![0., 2., 4., 4., 6., 8.]);
    let mut i = Instance {
        poses: vec![0.; 22],
        globals: vec![],
        input: vec![],
        output: vec![],
    };
    unsafe {
        evaluate_only(
            (&a as *const Asset).cast(),
            (&mut i as *mut Instance).cast(),
            (&clip as *const Animation).cast(),
            2.,
        );
    }
    assert_eq!(&i.poses[..3], &[2., 4., 6.]);
}
