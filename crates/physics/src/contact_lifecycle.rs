//! Contact creation and destruction, including body topology and owner-array membership.
use crate::{bodies, events, island, manifold_abi::*, manifolds, shapes, solver_set};

pub unsafe fn create(world: usize, mut shape_a: usize, mut shape_b: usize, child: i32) -> usize {
    let shapes = shapes::col(world);
    let stride = crate::shapes::SHAPE_STRIDE;
    match manifolds::contact_pair_order(
        shapes.get(shape_a * stride + crate::shapes::S_TYPE) as usize,
        shapes.get(shape_b * stride + crate::shapes::S_TYPE) as usize,
    ) {
        0 => return usize::MAX,
        2 => core::mem::swap(&mut shape_a, &mut shape_b),
        _ => {}
    }
    let kind = shapes.get(shape_a * stride + crate::shapes::S_TYPE);
    let a = shapes.get(shape_a * stride + crate::shapes::S_QUERY_BODY) as usize;
    let b = shapes.get(shape_b * stride + crate::shapes::S_QUERY_BODY) as usize;
    let body_a = bodies::record(world, a);
    let body_b = bodies::record(world, b);
    let set = if body_a.set_index == 2 || body_b.set_index == 2 {
        2
    } else {
        1
    };
    let id = manifolds::alloc_contact_in_world(world);
    let d = manifolds::dir_col(world);
    let o = id * DIR_STRIDE;
    d.set(o + DIR_SET_INDEX, set as u32);
    d.set(
        o + DIR_LOCAL_INDEX,
        solver_set::array_count_in_world(world, set, 0) as u32,
    );
    d.set(o + DIR_SHAPE_A, shape_a as u32);
    d.set(o + DIR_SHAPE_B, shape_b as u32);
    d.set(o + DIR_CHILD_INDEX, child as u32);
    let mut flags = 0;
    if body_a.flags & 0x4000 != 0 && body_b.flags & 0x4000 != 0 {
        flags |= 0x10;
    }
    if kind == 2
        || kind == 4
        || (kind == 1 && crate::geo::shape_compound_child_type(world, shape_a, child as usize) == 4)
    {
        flags |= 0x0040_0000;
    }
    if body_a.body_type == 0 || body_b.body_type == 0 {
        flags |= 8;
    }
    let flags_a = shapes.get(shape_a * stride + crate::shapes::S_FLAGS) >> 16;
    let flags_b = shapes.get(shape_b * stride + crate::shapes::S_FLAGS) >> 16;
    if (flags_a | flags_b) & 2 != 0 {
        flags |= 4;
    }
    if shapes.get(shape_a * stride + crate::shapes::S_FLAGS) & crate::shapes::SPECULATIVE_FLAG != 0
        && shapes.get(shape_b * stride + crate::shapes::S_FLAGS) & crate::shapes::SPECULATIVE_FLAG
            != 0
    {
        flags |= 0x0100_0000;
    }
    for (side, body_id) in [a, b].into_iter().enumerate() {
        let body = bodies::record_mut(world, body_id);
        let edge = o + DIR_EDGE_A + 3 * side;
        d.set(edge, body_id as u32);
        d.set(edge + 2, body.head_contact_key as u32);
        let key = ((id as i32) << 1) | side as i32;
        if body.head_contact_key != -1 {
            let prev = body.head_contact_key;
            d.set(
                (prev >> 1) as usize * DIR_STRIDE + DIR_EDGE_A + 1 + 3 * (prev & 1) as usize,
                key as u32,
            );
        }
        body.head_contact_key = key;
        body.contact_count += 1;
        d.set(
            o + DIR_INDEX_A + side,
            if body.body_type == 0 {
                u32::MAX
            } else {
                body.local_index as u32
            },
        );
    }
    crate::table::add_pair_in_world(world, shape_a as u32, shape_b as u32, child as u32);
    solver_set::array_push_in_world(world, set, 0, id as i32);
    let radius = |shape: usize| match shapes.get(shape * stride + crate::shapes::S_TYPE) {
        5 => f32::from_bits(shapes.get(shape * stride + crate::shapes::S_GEOM + 3)),
        0 => f32::from_bits(shapes.get(shape * stride + crate::shapes::S_GEOM + 6)),
        _ => 0.0,
    };
    let rolling_a = f32::from_bits(crate::shapes::material(world, shape_a, 0)[2]);
    let rolling_b = f32::from_bits(crate::shapes::material(world, shape_b, 0)[2]);
    d.set(
        o + DIR_ROLLING_RESISTANCE,
        (crate::math::maxf(rolling_a, rolling_b)
            * crate::math::maxf(radius(shape_a), radius(shape_b)))
        .to_bits(),
    );
    if (flags_a | flags_b) & 16 != 0 {
        flags |= 0x0020_0000;
    }
    d.set(o + DIR_FLAGS, flags);
    id
}

pub unsafe fn destroy(world: usize, id: usize, wake: bool) {
    let d = manifolds::dir_col(world);
    let o = id * DIR_STRIDE;
    crate::table::remove_pair_in_world(
        world,
        d.get(o + DIR_SHAPE_A),
        d.get(o + DIR_SHAPE_B),
        d.get(o + DIR_CHILD_INDEX),
    );
    manifolds::free_manifolds_in_world(world, id);
    let flags = d.get(o + DIR_FLAGS);
    let a = d.get(o + DIR_EDGE_A) as usize;
    let b = d.get(o + DIR_EDGE_B) as usize;
    if flags & 5 == 5 {
        events::contact_touch_in_world(world, id, false);
    }
    for side in 0..2 {
        let edge = o + DIR_EDGE_A + 3 * side;
        let body = bodies::record_mut(world, d.get(edge) as usize);
        let prev = d.get(edge + 1) as i32;
        let next = d.get(edge + 2) as i32;
        if prev != -1 {
            d.set(
                (prev >> 1) as usize * DIR_STRIDE + DIR_EDGE_A + 2 + 3 * (prev & 1) as usize,
                next as u32,
            );
        }
        if next != -1 {
            d.set(
                (next >> 1) as usize * DIR_STRIDE + DIR_EDGE_A + 1 + 3 * (next & 1) as usize,
                prev as u32,
            );
        }
        if body.head_contact_key == (((id as i32) << 1) | side as i32) {
            body.head_contact_key = next;
        }
        body.contact_count -= 1;
    }
    manifolds::free_mesh_cache_in_world(world, id);
    if d.get(o + DIR_ISLAND_ID) != u32::MAX {
        island::unlink_contact_in_world(world, id as i32);
    }
    let color = d.get(o + DIR_COLOR_INDEX);
    let index = d.get(o + DIR_LOCAL_INDEX) as usize;
    if color != u32::MAX {
        crate::constraint_graph::remove_contact_in_world(
            world,
            a,
            b,
            color as usize,
            index,
            flags & 0x0040_0000 != 0,
        );
    } else {
        let set = d.get(o + DIR_SET_INDEX) as usize;
        if solver_set::array_remove_in_world(world, set, 0, index) != -1 {
            let moved = solver_set::array_get_in_world(world, set, 0, index) as usize;
            d.set(moved * DIR_STRIDE + DIR_LOCAL_INDEX, index as u32);
        }
    }
    manifolds::free_contact_in_world(world, id);
    if wake && flags & 1 != 0 {
        solver_set::wake(world, bodies::record(world, a).set_index as usize);
        solver_set::wake(world, bodies::record(world, b).set_index as usize);
    }
}
