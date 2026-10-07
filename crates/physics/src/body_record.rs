use crate::math::Mat3;

/// Body identity and topology, separate from its solver-set simulation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BodyRecord {
    pub set_index: i32,
    pub local_index: i32,
    pub head_contact_key: i32,
    pub contact_count: i32,
    pub head_shape_id: i32,
    pub shape_count: i32,
    pub head_chain_id: i32,
    pub head_joint_key: i32,
    pub joint_count: i32,
    pub island_id: i32,
    pub island_index: i32,
    pub sleep_threshold: f32,
    pub sleep_time: f32,
    pub sleep_velocity: f32,
    pub mass: f32,
    pub inertia: Mat3,
    pub body_move_index: i32,
    pub id: i32,
    pub flags: u32,
    pub body_type: i32,
    pub generation: u16,
    pub(crate) padding: u16,
}

impl BodyRecord {
    pub const EMPTY: Self = Self {
        set_index: -1,
        local_index: -1,
        head_contact_key: -1,
        contact_count: 0,
        head_shape_id: -1,
        shape_count: 0,
        head_chain_id: -1,
        head_joint_key: -1,
        joint_count: 0,
        island_id: -1,
        island_index: -1,
        sleep_threshold: 0.0,
        sleep_time: 0.0,
        sleep_velocity: 0.0,
        mass: 0.0,
        inertia: Mat3 {
            cx: crate::math::Vec3::ZERO,
            cy: crate::math::Vec3::ZERO,
            cz: crate::math::Vec3::ZERO,
        },
        body_move_index: -1,
        id: -1,
        flags: 0,
        body_type: 0,
        generation: 0,
        padding: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::BodyRecord;
    use std::mem::{offset_of, size_of};
    #[test]
    fn body_record_layout_matches_the_plain_word_bindings() {
        assert_eq!(size_of::<BodyRecord>(), 29 * 4);
        let offsets = [
            offset_of!(BodyRecord, set_index),
            offset_of!(BodyRecord, local_index),
            offset_of!(BodyRecord, head_contact_key),
            offset_of!(BodyRecord, contact_count),
            offset_of!(BodyRecord, head_shape_id),
            offset_of!(BodyRecord, shape_count),
            offset_of!(BodyRecord, head_chain_id),
            offset_of!(BodyRecord, head_joint_key),
            offset_of!(BodyRecord, joint_count),
            offset_of!(BodyRecord, island_id),
            offset_of!(BodyRecord, island_index),
            offset_of!(BodyRecord, sleep_threshold),
            offset_of!(BodyRecord, sleep_time),
            offset_of!(BodyRecord, sleep_velocity),
            offset_of!(BodyRecord, mass),
            offset_of!(BodyRecord, inertia),
        ];
        for (word, offset) in offsets.into_iter().enumerate() {
            assert_eq!(offset, word * 4);
        }
        assert_eq!(offset_of!(BodyRecord, body_move_index), 24 * 4);
        assert_eq!(offset_of!(BodyRecord, id), 25 * 4);
        assert_eq!(offset_of!(BodyRecord, flags), 26 * 4);
        assert_eq!(offset_of!(BodyRecord, body_type), 27 * 4);
        assert_eq!(offset_of!(BodyRecord, generation), 28 * 4);
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(target_arch = "wasm32")]
pub(crate) mod runtime {
    use crate::math::Mat3;
    use crate::{bodies, body, island};
    #[export_name = "bodyFinish"]
    pub unsafe extern "C" fn finish(count: usize, time_step: f32, enable_sleep: bool) {
        unsafe { finish_in_world(crate::regions::active(), count, time_step, enable_sleep) }
    }

    pub unsafe extern "C" fn finish_in_world(
        world_index: usize,
        count: usize,
        time_step: f32,
        enable_sleep: bool,
    ) {
        let sim2 = bodies::sim2_base(world_index) as *mut u32;
        let state_flags = bodies::flags_base(world_index) as *mut u32;
        let transient =
            body::flags::IS_FAST | body::flags::IS_SPEED_CAPPED | body::flags::HAD_TIME_OF_IMPACT;
        crate::events::set_move_count(world_index, count);
        for index in 0..count {
            let row = sim2.add(index * body::SIM2_STRIDE);
            let id = *row.add(body::S2_BODY_ID) as usize;
            let sim_flags = *row.add(body::S2_FLAGS);
            let flags = *state_flags.add(index * body::STATE_STRIDE);
            let record = bodies::record_mut(world_index, id);
            record.body_move_index = index as i32;
            record.flags = (record.flags & !transient)
                | ((sim_flags | flags)
                    & (body::flags::IS_SPEED_CAPPED | body::flags::HAD_TIME_OF_IMPACT));
            *row.add(body::S2_FLAGS) =
                (sim_flags & !transient) | (sim_flags & body::flags::IS_FAST);
            *state_flags.add(index * body::STATE_STRIDE) = flags & !transient;
            if !enable_sleep
                || record.flags & body::flags::ENABLE_SLEEP == 0
                || record.sleep_velocity > record.sleep_threshold
            {
                record.sleep_time = 0.0;
            } else {
                record.sleep_time += time_step;
            }
        }
    }
    pub unsafe fn gather_split(world_index: usize, count: usize) -> i32 {
        let sim2 = bodies::sim2_base(world_index) as *const u32;
        let mut split_id = -1;
        let mut split_sleep = 0.0;
        for index in 0..count {
            let id = *sim2.add(index * body::SIM2_STRIDE + body::S2_BODY_ID) as usize;
            let record = bodies::record(world_index, id);
            if record.sleep_time >= 0.5
                && island::field_in_world(world_index, record.island_id as usize, 3) > 0
                && (record.sleep_time > split_sleep
                    || (record.sleep_time == split_sleep && record.island_id > split_id))
            {
                split_id = record.island_id;
                split_sleep = record.sleep_time;
            }
        }
        split_id
    }

    #[export_name = "bodyVelocitySet"]
    pub unsafe extern "C" fn velocity_set(
        world: usize,
        id: usize,
        angular: bool,
        x: f32,
        y: f32,
        z: f32,
    ) -> bool {
        crate::regions::select(world as u32);
        unsafe { velocity_set_in_world(world, id, angular, x, y, z) }
    }

    pub unsafe extern "C" fn velocity_set_in_world(
        world: usize,
        id: usize,
        angular: bool,
        x: f32,
        y: f32,
        z: f32,
    ) -> bool {
        let record = *bodies::record(world, id);
        if record.body_type == 0 {
            return false;
        }
        let v = if angular {
            crate::math::Vec3::new(
                if record.flags & body::flags::LOCK_ANGULAR_X != 0 {
                    0.0
                } else {
                    x
                },
                if record.flags & body::flags::LOCK_ANGULAR_Y != 0 {
                    0.0
                } else {
                    y
                },
                if record.flags & body::flags::LOCK_ANGULAR_Z != 0 {
                    0.0
                } else {
                    z
                },
            )
        } else {
            crate::math::Vec3::new(x, y, z)
        };
        if record.set_index >= 3 && v.length_sq() != 0.0 {
            crate::body_mutation::wake_body_in_world(world, id);
        }
        let record = bodies::record(world, id);
        if record.set_index == 2 {
            let state = crate::col::Col::new(
                bodies::state_base(world as usize) as *mut f32,
                (record.local_index as usize + 1) * body::STATE_STRIDE,
            );
            let offset =
                record.local_index as usize * body::STATE_STRIDE + if angular { 3 } else { 0 };
            state.set(offset, v.x);
            state.set(offset + 1, v.y);
            state.set(offset + 2, v.z);
        }
        false
    }

    #[export_name = "bodyApply"]
    pub unsafe extern "C" fn apply(
        world: usize,
        id: usize,
        kind: u32,
        x: f32,
        y: f32,
        z: f32,
        px: f32,
        py: f32,
        pz: f32,
        max_speed: f32,
        wake: bool,
    ) {
        crate::regions::select(world as u32);
        unsafe { apply_in_world(world, id, kind, x, y, z, px, py, pz, max_speed, wake) }
    }

    pub unsafe extern "C" fn apply_in_world(
        world: usize,
        id: usize,
        kind: u32,
        x: f32,
        y: f32,
        z: f32,
        px: f32,
        py: f32,
        pz: f32,
        max_speed: f32,
        wake: bool,
    ) {
        use crate::math::Vec3;

        if wake {
            crate::body_mutation::wake_body_in_world(world, id);
        }
        let record = bodies::record(world, id);
        if record.set_index != 2 {
            return;
        }
        let sim = bodies::column(world as usize, id, 1, body::SIM_STRIDE);
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        let v = Vec3::new(x, y, z);
        if kind <= 2 {
            let offset = if kind == 2 { body::TORQUE } else { body::FORCE };
            for (lane, value) in [v.x, v.y, v.z].into_iter().enumerate() {
                sim.set(offset + lane, sim.get(offset + lane) + value);
            }
            if kind == 0 {
                let center = body::read_fin(fin, 0).center;
                let torque = Vec3::new(px, py, pz).sub(center).cross(v);
                sim.set(body::TORQUE, sim.get(body::TORQUE) + torque.x);
                sim.set(body::TORQUE + 1, sim.get(body::TORQUE + 1) + torque.y);
                sim.set(body::TORQUE + 2, sim.get(body::TORQUE + 2) + torque.z);
            }
            return;
        }
        let state_col = bodies::column(world as usize, id, 0, body::STATE_STRIDE);
        let mut state = body::read_state(state_col, 0);
        let data = body::read_sim(sim, 0);
        if kind == 3 || kind == 4 {
            let mut linear = state.linear_velocity.mul_add(data.inv_mass, v);
            let length_sq = linear.length_sq();
            if length_sq > max_speed * max_speed {
                linear = if length_sq > 1000.0 * f32::MIN_POSITIVE {
                    linear.scale(1.0 / length_sq.sqrt()).scale(max_speed)
                } else {
                    Vec3::ZERO
                };
            }
            state.linear_velocity = linear;
            if kind == 3 {
                let offset = Vec3::new(px, py, pz).sub(body::read_fin(fin, 0).center);
                state.angular_velocity = state
                    .angular_velocity
                    .add(data.inv_inertia_world.mul_v(offset.cross(v)));
            }
        } else {
            let local = data.rotation.inv_rotate(v);
            let delta = data.rotation.rotate(data.inv_inertia_local.mul_v(local));
            state.angular_velocity = state.angular_velocity.add(delta);
        }
        body::write_state(state_col, 0, &state);
    }

    #[export_name = "bodySetPose"]
    pub unsafe extern "C" fn set_pose(
        world: usize,
        id: usize,
        x: f32,
        y: f32,
        z: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
    ) {
        crate::regions::select(world as u32);
        unsafe { set_pose_in_world(world, id, x, y, z, qx, qy, qz, qs) }
    }

    pub unsafe extern "C" fn set_pose_in_world(
        world: usize,
        id: usize,
        x: f32,
        y: f32,
        z: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
    ) {
        use crate::math::{Quat, Vec3};

        let sim = bodies::column(world as usize, id, 1, body::SIM_STRIDE);
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        let sim2 = bodies::column(world as usize, id, 5, body::SIM2_STRIDE);
        let q = Quat {
            v: Vec3::new(qx, qy, qz),
            s: qs,
        };
        let p = Vec3::new(x, y, z);
        body::write_fin_transform_p(fin, 0, p);
        body::write_sim_rotation(sim, 0, q);
        let center = q.rotate(body::read_fin(fin, 0).local_center).add(p);
        body::write_fin_center(fin, 0, center);
        let rotation = Mat3::from_quat(q);
        let inertia = rotation
            .mul(body::read_sim(sim, 0).inv_inertia_local)
            .mul(rotation.transpose());
        body::write_sim_inv_inertia_world(sim, 0, inertia);
        for (lane, value) in [qx, qy, qz, qs].into_iter().enumerate() {
            sim2.set(body::S2_ROTATION0 + lane, value);
        }
        for (lane, value) in [center.x, center.y, center.z].into_iter().enumerate() {
            sim2.set(body::S2_CENTER0 + lane, value);
        }
        crate::shape_lifecycle::sync_body_bounds_in_world(world, id);
    }

    #[export_name = "bodyTransfer"]
    pub unsafe extern "C" fn transfer(
        world: usize,
        id: usize,
        target: usize,
        clear_transient: bool,
    ) -> u32 {
        crate::regions::select(world as u32);
        unsafe { transfer_in_world(world, id, target, clear_transient) }
    }

    pub unsafe extern "C" fn transfer_in_world(
        world: usize,
        id: usize,
        target: usize,
        clear_transient: bool,
    ) -> u32 {
        let body = *bodies::record(world, id);
        if body.set_index as usize == target {
            return u32::MAX;
        }
        let result = crate::solver_set::transfer_body_in_world(
            world as usize,
            body.set_index as usize,
            body.local_index as usize,
            target,
            body.flags,
            body.head_shape_id,
            clear_transient,
        ) as *const u32;
        let moved = *result.add(1);
        moved
    }

    #[export_name = "bodyWakeRecord"]
    pub unsafe extern "C" fn wake_record(world: usize, id: usize) {
        crate::regions::select(world as u32);
        unsafe { wake_record_in_world(world, id) }
    }

    pub unsafe extern "C" fn wake_record_in_world(world: usize, id: usize) {
        let body = *bodies::record(world, id);
        crate::solver_set::wake_body_in_world(
            world as usize,
            body.set_index as usize,
            body.local_index as usize,
            body.flags,
            body.head_shape_id,
        );
    }

    #[export_name = "bodyCreateSim"]
    pub unsafe extern "C" fn create_sim(
        world: u32,
        body_type: i32,
        flags: u32,
        awake: bool,
        enabled: bool,
        threshold: f32,
        px: f32,
        py: f32,
        pz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        wx: f32,
        wy: f32,
        wz: f32,
        linear_damping: f32,
        angular_damping: f32,
        gravity_scale: f32,
    ) -> u32 {
        crate::regions::select(world as u32);
        unsafe {
            create_sim_in_world(
                world,
                body_type,
                flags,
                awake,
                enabled,
                threshold,
                px,
                py,
                pz,
                qx,
                qy,
                qz,
                qs,
                vx,
                vy,
                vz,
                wx,
                wy,
                wz,
                linear_damping,
                angular_damping,
                gravity_scale,
            )
        }
    }

    pub unsafe extern "C" fn create_sim_in_world(
        world: u32,
        body_type: i32,
        flags: u32,
        awake: bool,
        enabled: bool,
        threshold: f32,
        px: f32,
        py: f32,
        pz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        wx: f32,
        wy: f32,
        wz: f32,
        linear_damping: f32,
        angular_damping: f32,
        gravity_scale: f32,
    ) -> u32 {
        use crate::math::{Quat, Vec3};

        let flags = (flags & !body::flags::DYNAMIC)
            | if body_type == 2 {
                body::flags::DYNAMIC
            } else {
                0
            };
        let set = if !enabled {
            1
        } else if body_type == 0 {
            0
        } else if awake || flags & body::flags::ENABLE_SLEEP == 0 {
            2
        } else {
            crate::solver_set::create_in_world(world as usize)
        };
        let id = bodies::body_create_in_world(world as usize, world);
        let index = crate::solver_set::body_append_in_world(world as usize, set);
        let record = bodies::record_mut(world as usize, id as usize);
        record.set_index = set as i32;
        record.local_index = index as i32;
        record.sleep_threshold = threshold;
        record.flags = flags;
        record.body_type = body_type;
        crate::solver_set::body_ptr(world as usize, set, index, 1).write_bytes(0, body::SIM_STRIDE);
        let sim = bodies::column(world as usize, id as usize, 1, body::SIM_STRIDE);
        let fin = bodies::column(world as usize, id as usize, 2, body::FIN_STRIDE);
        let sim2 = bodies::column(world as usize, id as usize, 5, body::SIM2_STRIDE);
        let p = Vec3::new(px, py, pz);
        let q = Quat {
            v: Vec3::new(qx, qy, qz),
            s: qs,
        };
        body::write_fin_center(fin, 0, p);
        body::write_fin_transform_p(fin, 0, p);
        body::write_sim_rotation(sim, 0, q);
        sim.set(body::GRAVITY_SCALE, gravity_scale);
        sim.set(body::LINEAR_DAMPING, linear_damping);
        sim.set(body::ANGULAR_DAMPING, angular_damping);
        for (lane, value) in [qx, qy, qz, qs].into_iter().enumerate() {
            sim2.set(body::S2_ROTATION0 + lane, value);
        }
        for (lane, value) in [px, py, pz].into_iter().enumerate() {
            sim2.set(body::S2_CENTER0 + lane, value);
        }
        sim2.set(body::S2_MIN_EXTENT, 1.0e5);
        sim2.set(body::S2_BODY_ID, f32::from_bits(id));
        sim2.set(body::S2_FLAGS, f32::from_bits(flags));
        if set == 2 {
            crate::solver_set::body_ptr(world as usize, set, index, 0)
                .write_bytes(0, body::STATE_STRIDE);
            let state = body::State {
                linear_velocity: Vec3::new(vx, vy, vz),
                angular_velocity: Vec3::new(wx, wy, wz),
                delta_position: Vec3::ZERO,
                delta_rotation: Quat::IDENTITY,
            };
            body::write_state(
                bodies::column(world as usize, id as usize, 0, body::STATE_STRIDE),
                0,
                &state,
            );
            *(crate::solver_set::body_ptr(world as usize, set, index, 4)) = flags;
        }
        if set >= 2 {
            let island = island::create_in_world(world as usize, set);
            island::add_body_in_world(world as usize, island, id as i32);
        }
        id
    }

    // Preserve the JS API's rounding point for host translation and dt: subtraction and reciprocal
    // precede f32 conversion. The resident pose and velocities remain f32.
    #[export_name = "bodyTargetVelocity"]
    pub unsafe extern "C" fn target_velocity(
        world: usize,
        id: usize,
        tx: f64,
        ty: f64,
        tz: f64,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
        time_step: f64,
        wake: bool,
    ) -> bool {
        crate::regions::select(world as u32);
        unsafe { target_velocity_in_world(world, id, tx, ty, tz, qx, qy, qz, qs, time_step, wake) }
    }

    pub unsafe extern "C" fn target_velocity_in_world(
        world: usize,
        id: usize,
        tx: f64,
        ty: f64,
        tz: f64,
        qx: f32,
        qy: f32,
        qz: f32,
        qs: f32,
        time_step: f64,
        wake: bool,
    ) -> bool {
        use crate::math::{Quat, Vec3};

        let record = bodies::record(world, id);
        if record.set_index == 1
            || record.body_type == 0
            || time_step <= 0.0
            || (record.set_index != 2 && !wake)
        {
            return false;
        }
        let (pose, fin, _) = bodies::geometry(world as usize, id);
        let q = Quat {
            v: Vec3::new(qx, qy, qz),
            s: qs,
        };
        let rotated = q.rotate(fin.local_center);
        let center = Vec3::new(
            (rotated.x as f64 + tx) as f32,
            (rotated.y as f64 + ty) as f32,
            (rotated.z as f64 + tz) as f32,
        );
        let inv_dt = (1.0 / time_step) as f32;
        let linear = center.sub(fin.center).scale(inv_dt);
        let sign = if pose.q.dot(q) < 0.0 { -1.0 } else { 1.0 };
        let difference = q.v.scale(sign).sub(pose.q.v);
        let ds = sign * qs - pose.q.s;
        let conjugate = pose.q.v.neg();
        let angular = difference
            .cross(conjugate)
            .add(conjugate.scale(ds))
            .add(difference.scale(pose.q.s))
            .scale(2.0 * inv_dt);
        if record.set_index != 2 {
            let speed = linear.length()
                + Vec3::new(
                    angular.x * fin.max_extent.x,
                    angular.y * fin.max_extent.y,
                    angular.z * fin.max_extent.z,
                )
                .length();
            if speed < record.sleep_threshold {
                return false;
            }
            crate::body_mutation::wake_body_in_world(world, id);
        }
        let state = bodies::column(world as usize, id, 0, body::STATE_STRIDE);
        for (lane, value) in [
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        ]
        .into_iter()
        .enumerate()
        {
            state.set(lane, value);
        }
        false
    }

    #[export_name = "bodySyncFlags"]
    pub unsafe extern "C" fn sync_flags(world: usize, id: usize) {
        crate::regions::select(world as u32);
        unsafe { sync_flags_in_world(world, id) }
    }

    pub unsafe extern "C" fn sync_flags_in_world(world: usize, id: usize) {
        let record = bodies::record(world, id);
        let flags = record.flags
            & !(body::flags::IS_FAST
                | body::flags::IS_SPEED_CAPPED
                | body::flags::HAD_TIME_OF_IMPACT);
        bodies::column(world as usize, id, 5, body::SIM2_STRIDE)
            .set(body::S2_FLAGS, f32::from_bits(flags));
        if record.set_index == 2 {
            *crate::solver_set::body_ptr(world as usize, 2, record.local_index as usize, 4) = flags;
        }
    }

    #[export_name = "bodyChangeType"]
    pub unsafe extern "C" fn change_type(world: usize, id: usize, body_type: i32) {
        crate::regions::select(world as u32);
        unsafe { change_type_in_world(world, id, body_type) }
    }

    pub unsafe extern "C" fn change_type_in_world(world: usize, id: usize, body_type: i32) {
        let record = bodies::record_mut(world, id);
        record.body_type = body_type;
        if body_type == 2 {
            record.flags |= body::flags::DYNAMIC;
        } else {
            record.flags &= !body::flags::DYNAMIC;
        }
        sync_flags_in_world(world, id);
    }

    fn next_shape(world_index: usize, id: usize) -> i32 {
        crate::shapes::col(world_index)
            .get(id * crate::shapes::SHAPE_STRIDE + crate::shapes::S_NEXT) as i32
    }

    unsafe fn mass_begin(world: usize, id: usize) {
        let record = bodies::record_mut(world, id);
        record.mass = 0.0;
        record.inertia = Mat3::ZERO;
        let sim = bodies::column(world as usize, id, 1, body::SIM_STRIDE);
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        sim.set(body::INV_MASS, 0.0);
        for lane in body::INV_INERTIA_LOCAL..body::S2_MIN_EXTENT {
            sim.set(lane, 0.0);
        }
        for lane in 0..3 {
            fin.set(body::LOCAL_CENTER + lane, 0.0);
            fin.set(body::MAX_EXTENT + lane, 0.0);
        }
        bodies::column(world as usize, id, 5, body::SIM2_STRIDE).set(body::S2_MIN_EXTENT, 1.0e5);
    }

    unsafe fn mass_finish(world: usize, id: usize) {
        use crate::math::Vec3;

        let record = bodies::record_mut(world, id);
        if record.shape_count == 0 {
            return;
        }
        let sim = bodies::column(world as usize, id, 1, body::SIM_STRIDE);
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        let sim2 = bodies::column(world as usize, id, 5, body::SIM2_STRIDE);
        let pose = bodies::geometry(world as usize, id).0;
        let mut center = Vec3::ZERO;
        if record.body_type == 2 {
            let mut shape = record.head_shape_id;
            while shape != -1 {
                let id = shape as usize;
                shape = next_shape(world as usize, id);
                if crate::shapes::col_f(world as usize)
                    .get(id * crate::shapes::SHAPE_STRIDE + crate::shapes::S_DENSITY)
                    == 0.0
                {
                    continue;
                }
                let entry = crate::shape_geometry::mass(world as usize, id);
                record.mass += entry.mass;
                center = center.mul_add(entry.mass, entry.center);
            }
            if record.mass > 0.0 {
                sim.set(body::INV_MASS, 1.0 / record.mass);
                center = center.scale(sim.get(body::INV_MASS));
            }
            let mut shape = record.head_shape_id;
            while shape != -1 {
                let id = shape as usize;
                shape = next_shape(world as usize, id);
                if crate::shapes::col_f(world as usize)
                    .get(id * crate::shapes::SHAPE_STRIDE + crate::shapes::S_DENSITY)
                    == 0.0
                {
                    continue;
                }
                let entry = crate::shape_geometry::mass(world as usize, id);
                if entry.mass == 0.0 {
                    continue;
                }
                let r = center.sub(entry.center);
                let nm = -entry.mass;
                let xy = nm * r.x * r.y;
                let xz = nm * r.x * r.z;
                let yz = nm * r.y * r.z;
                let offset = Mat3 {
                    cx: Vec3::new(entry.mass * (r.y * r.y + r.z * r.z), xy, xz),
                    cy: Vec3::new(xy, entry.mass * (r.x * r.x + r.z * r.z), yz),
                    cz: Vec3::new(xz, yz, entry.mass * (r.x * r.x + r.y * r.y)),
                };
                record.inertia = record.inertia.add(entry.inertia.add(offset));
            }
            if record.inertia.det() > 0.0 {
                let inverse = record.inertia.invert().transpose();
                let values = [
                    inverse.cx.x,
                    inverse.cx.y,
                    inverse.cx.z,
                    inverse.cy.x,
                    inverse.cy.y,
                    inverse.cy.z,
                    inverse.cz.x,
                    inverse.cz.y,
                    inverse.cz.z,
                ];
                for lane in 0..9 {
                    sim.set(body::INV_INERTIA_LOCAL + lane, values[lane]);
                }
                let rotation = Mat3::from_quat(pose.q);
                body::write_sim_inv_inertia_world(
                    sim,
                    0,
                    rotation.mul(inverse).mul(rotation.transpose()),
                );
            }
            let old_center = body::read_fin(fin, 0).center;
            fin.set(body::LOCAL_CENTER, center.x);
            fin.set(body::LOCAL_CENTER + 1, center.y);
            fin.set(body::LOCAL_CENTER + 2, center.z);
            let next_center = pose.point(center);
            body::write_fin_center(fin, 0, next_center);
            if record.set_index == 2 {
                let state = bodies::column(world as usize, id, 0, body::STATE_STRIDE);
                let velocity = Vec3::new(state.get(0), state.get(1), state.get(2));
                let angular = Vec3::new(state.get(3), state.get(4), state.get(5));
                let next = velocity.add(angular.cross(next_center.sub(old_center)));
                state.set(0, next.x);
                state.set(1, next.y);
                state.set(2, next.z);
            }
        } else {
            body::write_fin_center(fin, 0, pose.p);
        }
        let center = body::read_fin(fin, 0).center;
        sim2.set(body::S2_CENTER0, center.x);
        sim2.set(body::S2_CENTER0 + 1, center.y);
        sim2.set(body::S2_CENTER0 + 2, center.z);
        if record.flags & 0x38 == 0x38 {
            record.inertia = Mat3::ZERO;
            for lane in body::INV_INERTIA_LOCAL..body::S2_MIN_EXTENT {
                sim.set(lane, 0.0);
            }
        }
    }

    unsafe fn mass_extent(world: usize, id: usize, minimum: f32, x: f32, y: f32, z: f32) {
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        let sim2 = bodies::column(world as usize, id, 5, body::SIM2_STRIDE);
        sim2.set(
            body::S2_MIN_EXTENT,
            crate::math::minf(sim2.get(body::S2_MIN_EXTENT), minimum),
        );
        for (lane, value) in [x, y, z].into_iter().enumerate() {
            fin.set(
                body::MAX_EXTENT + lane,
                crate::math::maxf(fin.get(body::MAX_EXTENT + lane), value),
            );
        }
    }

    #[export_name = "bodyUpdateMass"]
    pub unsafe extern "C" fn update_mass(world: usize, id: usize) {
        unsafe { update_mass_in_world(world, id) }
    }

    pub unsafe extern "C" fn update_mass_in_world(world: usize, id: usize) {
        mass_begin(world, id);
        mass_finish(world, id);
        let fin = bodies::column(world as usize, id, 2, body::FIN_STRIDE);
        let center = body::read_fin(fin, 0).local_center;
        let record = bodies::record(world, id);
        let mut shape = if record.body_type == 0 {
            -1
        } else {
            record.head_shape_id
        };
        while shape != -1 {
            let (minimum, maximum) =
                crate::shape_geometry::extent(world as usize, shape as usize, center);
            mass_extent(world, id, minimum, maximum.x, maximum.y, maximum.z);
            shape = next_shape(world as usize, shape as usize);
        }
    }

    #[export_name = "bodyRemoveIsland"]
    pub unsafe extern "C" fn remove_island(world: usize, id: usize) {
        crate::regions::select(world as u32);
        unsafe { remove_island_in_world(world, id) }
    }

    pub unsafe extern "C" fn remove_island_in_world(world: usize, id: usize) {
        let body = *bodies::record(world, id);
        if body.island_id == -1 {
            return;
        }
        let island = body.island_id as usize;
        island::remove_body_in_world(world as usize, island, body.island_index as usize);
        if island::array_count_in_world(world as usize, island, 0) == 0 {
            island::destroy_in_world(world as usize, island);
        }
    }

    #[export_name = "bodyCreateIsland"]
    pub unsafe extern "C" fn create_island(world: usize, id: usize) {
        crate::regions::select(world as u32);
        unsafe { create_island_in_world(world, id) }
    }

    pub unsafe extern "C" fn create_island_in_world(world: usize, id: usize) {
        let set = bodies::record(world, id).set_index as usize;
        island::add_body_in_world(
            world as usize,
            island::create_in_world(world as usize, set),
            id as i32,
        );
    }

    #[export_name = "bodyColumnPtr"]
    pub unsafe extern "C" fn column_ptr(world: usize, id: usize, column: usize) -> usize {
        let record = bodies::record(world, id);
        crate::solver_set::body_ptr_world(
            world,
            record.set_index as usize,
            record.local_index as usize,
            column,
        ) as usize
    }

    #[export_name = "bodyStateIndex"]
    pub unsafe extern "C" fn state_index(world: usize, id: usize) -> i32 {
        let record = bodies::record(world, id);
        if record.set_index == 2 {
            record.local_index
        } else {
            -1
        }
    }
}
