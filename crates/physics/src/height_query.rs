//! Box3D height_field.c cell queries and leading-corner DDA casts.
use crate::distance::{
    shape_cast, shape_distance, CastOutput, DistanceInput, ShapeCastPairInput, ShapeProxy,
    SimplexCache,
};
use crate::manifold::Capsule;
use crate::math::{absf, clampf, maxf, minf, Plane, Transform, Vec3, FLT_EPSILON, FLT_MAX};
use crate::mesh_query::{
    bounds_overlap, bounds_triangle_overlap, component, intersect_ray_triangle, max, min, mul,
    proxy_bounds, signed_volume,
};
use crate::query::{PlaneResult, RayCastInput, ShapeCastInput};
#[derive(Clone, Copy)]
pub struct HeightField<'a> {
    pub lower: Vec3,
    pub upper: Vec3,
    pub min_height: f32,
    pub height_scale: f32,
    pub scale: Vec3,
    pub columns: usize,
    pub rows: usize,
    pub clockwise: bool,
    pub heights: &'a [u32],
    pub materials: &'a [u32],
}
impl HeightField<'_> {
    fn corners(self, row: usize, col: usize) -> [Vec3; 4] {
        let indices = [
            row * self.columns + col,
            row * self.columns + col + 1,
            (row + 1) * self.columns + col,
            (row + 1) * self.columns + col + 1,
        ];
        let x = [col, col + 1, col, col + 1];
        let z = [row, row, row + 1, row + 1];
        core::array::from_fn(|i| {
            mul(
                self.scale,
                Vec3::new(
                    x[i] as f32,
                    self.min_height + self.height_scale * self.heights[indices[i]] as f32,
                    z[i] as f32,
                ),
            )
        })
    }
    pub(crate) fn visit_cells(
        self,
        lower: Vec3,
        upper: Vec3,
        mut visit: impl FnMut(usize, [Vec3; 4]) -> bool,
    ) {
        let min_row = (lower.z / self.scale.z).floor() as i32;
        let max_row = (upper.z / self.scale.z).floor() as i32;
        let min_col = (lower.x / self.scale.x).floor() as i32;
        let max_col = (upper.x / self.scale.x).floor() as i32;
        for row in min_row.max(0)..=max_row.min(self.rows as i32 - 2) {
            for col in min_col.max(0)..=max_col.min(self.columns as i32 - 2) {
                let index = row as usize * (self.columns - 1) + col as usize;
                if self.materials[index] != 255
                    && !visit(index, self.corners(row as usize, col as usize))
                {
                    return;
                }
            }
        }
    }
}
fn ray_bounds(lower: Vec3, upper: Vec3, p1: Vec3, p2: Vec3) -> Option<(f32, f32)> {
    let d = p2.sub(p1);
    let length = d.length();
    if length < FLT_EPSILON {
        return if bounds_overlap(lower, upper, p1, p1) {
            Some((0.0, 0.0))
        } else {
            None
        };
    }
    let dir = d.scale(1.0 / length);
    let mut t_min = 0.0;
    let mut t_max = length;
    for i in 0..3 {
        let c = component(dir, i);
        let start = component(p1, i);
        let lo = component(lower, i);
        let hi = component(upper, i);
        if absf(c) < FLT_EPSILON {
            if start < lo || start > hi {
                return None;
            }
        } else {
            let mut t1 = (lo - start) / c;
            let mut t2 = (hi - start) / c;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            t_min = maxf(t_min, t1);
            t_max = minf(t_max, t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max < 0.0 {
        return None;
    }
    Some((
        clampf(t_min / length, 0.0, 1.0),
        clampf(t_max / length, 0.0, 1.0),
    ))
}
fn dda(start: i32, end: i32, scale: f32, position: f32, delta: f32) -> (f32, f32, i32) {
    if start < end {
        (
            scale / delta,
            (scale * (start + 1) as f32 - position) / delta,
            1,
        )
    } else if end < start {
        (scale / delta, (position - scale * start as f32) / delta, -1)
    } else {
        (0.0, FLT_MAX, 0)
    }
}
pub fn ray_cast_height(field: HeightField, input: &RayCastInput) -> CastOutput {
    let points = [input.origin];
    shape_cast_height(
        field,
        &ShapeCastInput {
            proxy: ShapeProxy {
                points: &points,
                count: 1,
                radius: 0.0,
            },
            translation: input.translation,
            max_fraction: input.max_fraction,
            can_encroach: false,
        },
    )
}
pub fn shape_cast_height(field: HeightField, input: &ShapeCastInput) -> CastOutput {
    let (lower, upper) = proxy_bounds(input.proxy);
    let start = lower.add(upper).scale(0.5);
    let extent = upper.sub(start);
    let delta = input.translation.scale(input.max_fraction);
    let margin = Vec3::new(0.05, 0.05, 0.05);
    let mut result = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    let Some((min_fraction, max_fraction)) = ray_bounds(
        field.lower.sub(extent).sub(margin),
        field.upper.add(extent).add(margin),
        start,
        start.add(delta),
    ) else {
        return result;
    };
    let mut clamped_start = start.mul_add(min_fraction, delta);
    let clamped_delta = delta.scale(max_fraction - min_fraction);
    let center_start = clamped_start;
    let center_end = clamped_start.add(clamped_delta);
    let sign_x = if input.translation.x >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let sign_z = if input.translation.z >= 0.0 {
        1.0
    } else {
        -1.0
    };
    clamped_start.x += sign_x * extent.x;
    clamped_start.z += sign_z * extent.z;
    let clamped_end = clamped_start.add(clamped_delta);
    let col_start = (clamped_start.x / field.scale.x).floor() as i32;
    let col_end = (clamped_end.x / field.scale.x).floor() as i32;
    let row_start = (clamped_start.z / field.scale.z).floor() as i32;
    let row_end = (clamped_end.z / field.scale.z).floor() as i32;
    let (delta_x, mut next_x, step_col) = dda(
        col_start,
        col_end,
        field.scale.x,
        clamped_start.x,
        absf(clamped_delta.x),
    );
    let (delta_z, mut next_z, step_row) = dda(
        row_start,
        row_end,
        field.scale.z,
        clamped_start.z,
        absf(clamped_delta.z),
    );
    let mut head_col = col_start;
    let mut head_row = row_start;
    let mut tail_col = ((clamped_start.x - 2.0 * sign_x * extent.x) / field.scale.x).floor() as i32;
    let mut tail_row = ((clamped_start.z - 2.0 * sign_z * extent.z) / field.scale.z).floor() as i32;
    let mut best = input.max_fraction;
    let grid_scale = input.max_fraction * (max_fraction - min_fraction);
    let grid_offset = input.max_fraction * min_fraction;
    let bounds_min = min(center_start, center_end).sub(extent);
    let bounds_max = max(center_start, center_end).add(extent);
    loop {
        for row in tail_row.min(head_row)..=tail_row.max(head_row) {
            if row < 0 || row >= field.rows as i32 - 1 {
                continue;
            }
            for col in tail_col.min(head_col)..=tail_col.max(head_col) {
                if col < 0 || col >= field.columns as i32 - 1 {
                    continue;
                }
                let cell = row as usize * (field.columns - 1) + col as usize;
                let material = field.materials[cell];
                if material == 255 {
                    continue;
                }
                let mut corners = field.corners(row as usize, col as usize);
                if field.clockwise {
                    corners.swap(1, 2);
                }
                let [a, b, c, d] = corners;
                let lo = min(min(a, b), min(c, d));
                let hi = max(max(a, b), max(c, d));
                if !bounds_overlap(bounds_min, bounds_max, lo, hi) {
                    continue;
                }
                if input.proxy.count == 1 && input.proxy.radius == 0.0 {
                    for (i, vertices) in [[a, c, b], [d, b, c]].into_iter().enumerate() {
                        let alpha = intersect_ray_triangle(start, input.translation, vertices);
                        if alpha < best {
                            let normal = if i == 0 {
                                c.sub(a).cross(b.sub(a))
                            } else {
                                d.sub(c).cross(b.sub(c))
                            };
                            result.point = start.mul_add(alpha, input.translation);
                            result.normal = normal.normalize();
                            result.fraction = alpha;
                            result.triangle_index = (2 * cell + i) as i32;
                            result.material_index = material as i32;
                            result.hit = true;
                            best = alpha;
                        }
                    }
                } else {
                    for (i, vertices) in [[a, c, b], [c, d, b]].into_iter().enumerate() {
                        if signed_volume(vertices, start) < 0.0 {
                            continue;
                        }
                        let origin = vertices[0];
                        let shifted =
                            [Vec3::ZERO, vertices[1].sub(origin), vertices[2].sub(origin)];
                        let pair = shape_cast(&ShapeCastPairInput {
                            proxy_a: ShapeProxy {
                                points: &shifted,
                                count: 3,
                                radius: 0.0,
                            },
                            proxy_b: input.proxy,
                            transform: Transform {
                                p: origin.neg(),
                                ..Transform::IDENTITY
                            },
                            translation_b: input.translation,
                            max_fraction: best,
                            can_encroach: input.can_encroach,
                        });
                        if pair.hit {
                            best = pair.fraction;
                            result = pair;
                            result.point = result.point.add(origin);
                            result.triangle_index = (2 * cell + i) as i32;
                            result.material_index = material as i32;
                        }
                    }
                }
            }
        }
        let fraction_x = if next_x == FLT_MAX {
            FLT_MAX
        } else {
            grid_offset + next_x * grid_scale
        };
        let fraction_z = if next_z == FLT_MAX {
            FLT_MAX
        } else {
            grid_offset + next_z * grid_scale
        };
        if fraction_x > best && fraction_z > best {
            break;
        }
        if next_x <= next_z {
            if head_col == col_end {
                break;
            }
            head_col += step_col;
            tail_col = head_col;
            tail_row = if extent.z == 0.0 {
                head_row
            } else {
                ((clamped_start.z + next_x * clamped_delta.z - 2.0 * sign_z * extent.z)
                    / field.scale.z)
                    .floor() as i32
            };
            next_x += delta_x;
        } else {
            if head_row == row_end {
                break;
            }
            head_row += step_row;
            tail_row = head_row;
            tail_col = if extent.x == 0.0 {
                head_col
            } else {
                ((clamped_start.x + next_z * clamped_delta.x - 2.0 * sign_x * extent.x)
                    / field.scale.x)
                    .floor() as i32
            };
            next_z += delta_z;
        }
    }
    result
}
pub fn overlap_height(field: HeightField, transform: Transform, proxy: ShapeProxy) -> bool {
    let mut points = [Vec3::ZERO; 128];
    let count = proxy.count.min(128);
    for i in 0..count {
        points[i] = transform.inv_point(proxy.points[i]);
    }
    let local = ShapeProxy {
        points: &points,
        count,
        radius: proxy.radius,
    };
    let (lower, upper) = proxy_bounds(local);
    let center = lower.add(upper).scale(0.5);
    let extent = upper.sub(center);
    let mut overlap = false;
    field.visit_cells(lower, upper, |_, [a, b, c, d]| {
        for (bounds, vertices) in [([a, c, b], [a, c, b]), ([c, d, b], [d, b, c])] {
            if !bounds_triangle_overlap(center, extent, bounds) {
                continue;
            }
            let input = DistanceInput {
                proxy_a: ShapeProxy {
                    points: &vertices,
                    count: 3,
                    radius: 0.0,
                },
                proxy_b: local,
                transform: Transform::IDENTITY,
                use_radii: true,
            };
            if shape_distance(&input, &mut SimplexCache::empty()).distance < 0.1 * 0.005 {
                overlap = true;
                return false;
            }
        }
        true
    });
    overlap
}
pub fn collide_mover_height(
    planes: &mut [PlaneResult],
    field: HeightField,
    mover: &Capsule,
) -> usize {
    if planes.is_empty() {
        return 0;
    }
    let points = [mover.center1, mover.center2];
    let (lower, upper) = proxy_bounds(ShapeProxy {
        points: &points,
        count: 2,
        radius: mover.radius,
    });
    let center = mover.center1.lerp(mover.center2, 0.5);
    let bounds_center = lower.add(upper).scale(0.5);
    let extent = upper.sub(bounds_center);
    let mut count = 0;
    field.visit_cells(lower, upper, |cell, mut corners| {
        if field.clockwise {
            corners.swap(1, 2);
        }
        let [a, b, c, d] = corners;
        for (i, (bounds, vertices)) in [([a, c, b], [a, c, b]), ([c, d, b], [d, b, c])]
            .into_iter()
            .enumerate()
        {
            if !bounds_triangle_overlap(bounds_center, extent, bounds)
                || signed_volume(vertices, center) < 0.0
            {
                continue;
            }
            let input = DistanceInput {
                proxy_a: ShapeProxy {
                    points: &vertices,
                    count: 3,
                    radius: 0.0,
                },
                proxy_b: ShapeProxy {
                    points: &points,
                    count: 2,
                    radius: 0.0,
                },
                transform: Transform::IDENTITY,
                use_radii: false,
            };
            let distance = shape_distance(&input, &mut SimplexCache::empty());
            if distance.distance != 0.0 && distance.distance <= mover.radius {
                planes[count] = PlaneResult {
                    plane: Plane {
                        normal: distance.normal,
                        offset: mover.radius - distance.distance,
                    },
                    point: distance.point_a,
                    triangle_index: (2 * cell + i) as i32,
                    child_index: 0,
                    material_index: field.materials[cell] as i32,
                };
                count += 1;
                if count == planes.len() {
                    return false;
                }
            }
        }
        true
    });
    count
}
