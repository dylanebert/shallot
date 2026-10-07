//! Box3D mesh.c query traversal over builder-authored BVH records.
use crate::distance::{
    shape_cast, shape_distance, CastOutput, DistanceInput, ShapeCastPairInput, ShapeProxy,
    SimplexCache,
};
use crate::manifold::Capsule;
use crate::math::{maxf, minf, Plane, Transform, Vec3};
use crate::query::{PlaneResult, RayCastInput, ShapeCastInput};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshNode {
    pub lower: Vec3,
    pub data: u32,
    pub upper: Vec3,
    pub triangle_offset: u32,
}
impl MeshNode {
    pub fn is_leaf(&self) -> bool {
        self.data & 3 == 3
    }
    pub fn axis(&self) -> usize {
        (self.data & 3) as usize
    }
    pub fn child_offset(&self) -> usize {
        (self.data >> 2) as usize
    }
    pub fn triangle_count(&self) -> u32 {
        self.data >> 2
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshTriangle {
    pub indices: [u32; 3],
}
#[derive(Clone, Copy)]
pub struct Mesh<'a> {
    pub nodes: &'a [MeshNode],
    pub vertices: &'a [Vec3],
    pub triangles: &'a [MeshTriangle],
    pub materials: &'a [u8],
    pub scale: Vec3,
}

pub(crate) fn mul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}
pub(crate) fn min(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z))
}
pub(crate) fn max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z))
}
pub(crate) fn component(a: Vec3, i: usize) -> f32 {
    [a.x, a.y, a.z][i]
}
fn positive(a: Vec3) -> bool {
    a.x > 0.0 || a.y > 0.0 || a.z > 0.0
}
pub(crate) fn bounds_overlap(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    !positive(a.sub(d)) && !positive(c.sub(b))
}
pub(crate) fn bounds_ray_overlap(lower: Vec3, upper: Vec3, start: Vec3, delta: Vec3) -> bool {
    let center = lower.add(upper).scale(0.5);
    let extent = upper.sub(center);
    let separation = delta
        .cross(start.sub(center))
        .abs()
        .sub(delta.abs().modified_cross(extent));
    !positive(separation)
}
pub(crate) fn bounds_triangle_overlap(center: Vec3, extent: Vec3, vertices: [Vec3; 3]) -> bool {
    crate::simd::bounds_triangle_overlap(center, extent, vertices)
}
pub(crate) fn signed_volume(vertices: [Vec3; 3], p: Vec3) -> f32 {
    let [a, b, c] = vertices;
    b.sub(a).cross(c.sub(a)).dot(p.sub(a))
}
pub(crate) fn intersect_ray_triangle(start: Vec3, delta: Vec3, vertices: [Vec3; 3]) -> f32 {
    crate::simd::intersect_ray_triangle(start, delta, vertices)
}
pub(crate) fn proxy_bounds(proxy: ShapeProxy) -> (Vec3, Vec3) {
    let mut lower = proxy.points[0];
    let mut upper = lower;
    for p in &proxy.points[1..proxy.count] {
        lower = min(lower, *p);
        upper = max(upper, *p);
    }
    let r = Vec3::new(proxy.radius, proxy.radius, proxy.radius);
    (lower.sub(r), upper.add(r))
}
impl Mesh<'_> {
    fn inverse_scale(self) -> Vec3 {
        Vec3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z)
    }
    fn reflected(self) -> bool {
        self.scale.x * self.scale.y * self.scale.z <= 0.0
    }
    fn triangle(self, index: usize, flip: bool) -> [Vec3; 3] {
        let mut indices = self.triangles[index].indices;
        if flip {
            indices.swap(1, 2);
        }
        indices.map(|i| self.vertices[i as usize])
    }
    fn unscaled_bounds(self, lower: Vec3, upper: Vec3) -> (Vec3, Vec3) {
        let inv = self.inverse_scale();
        let a = mul(inv, lower);
        let b = mul(inv, upper);
        (min(a, b), max(a, b))
    }
}
const STACK_SIZE: usize = 256;

pub fn ray_cast_mesh(mesh: Mesh, input: &RayCastInput) -> CastOutput {
    let mut output = CastOutput {
        fraction: input.max_fraction,
        ..CastOutput::MISS
    };
    let start = mul(mesh.inverse_scale(), input.origin);
    let delta = mul(mesh.inverse_scale(), input.translation);
    let reflected = mesh.reflected();
    let end = start.add(delta.scale(output.fraction));
    let mut swept_lower = min(start, end);
    let mut swept_upper = max(start, end);
    let mut stack = [0; STACK_SIZE];
    let mut count = 0;
    let mut index = 0;
    loop {
        let node = &mesh.nodes[index];
        if bounds_overlap(node.lower, node.upper, swept_lower, swept_upper)
            && bounds_ray_overlap(node.lower, node.upper, start, delta)
        {
            if node.is_leaf() {
                for t in node.triangle_offset..node.triangle_offset + node.triangle_count() {
                    let vertices = mesh
                        .triangle(t as usize, reflected)
                        .map(|v| mul(mesh.scale, v));
                    let alpha = intersect_ray_triangle(input.origin, input.translation, vertices);
                    if alpha < output.fraction {
                        output.normal = vertices[1]
                            .sub(vertices[0])
                            .cross(vertices[2].sub(vertices[0]))
                            .normalize();
                        output.point = input.origin.add(input.translation.scale(alpha));
                        output.fraction = alpha;
                        output.triangle_index = t as i32;
                        output.material_index = mesh.materials[t as usize] as i32;
                        output.hit = true;
                        let end = start.add(delta.scale(alpha));
                        swept_lower = min(start, end);
                        swept_upper = max(start, end);
                    }
                }
            } else {
                let left = index + 1;
                let right = index + node.child_offset();
                let (near, far) = if component(delta, node.axis()) > 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                stack[count] = far;
                count += 1;
                index = near;
                continue;
            }
        }
        if count == 0 {
            break;
        }
        count -= 1;
        index = stack[count];
    }
    output
}

pub fn shape_cast_mesh(mesh: Mesh, input: &ShapeCastInput) -> CastOutput {
    let mut output = CastOutput {
        fraction: input.max_fraction,
        ..CastOutput::MISS
    };
    let (lower, upper) = proxy_bounds(input.proxy);
    let center = lower.add(upper).scale(0.5);
    let extent = upper.sub(center);
    let inv = mesh.inverse_scale();
    let start = mul(inv, center);
    let delta = mul(inv, input.translation);
    let inv_extent = mul(inv.abs(), extent);
    let reflected = mesh.reflected();
    let end = start.add(delta.scale(output.fraction));
    let mut swept_lower = min(start, end);
    let mut swept_upper = max(start, end);
    let scaled_end = center.add(input.translation.scale(output.fraction));
    let mut scaled_lower = min(center, scaled_end);
    let mut scaled_upper = max(center, scaled_end);
    let mut stack = [0; STACK_SIZE];
    let mut count = 0;
    let mut index = 0;
    loop {
        let node = &mesh.nodes[index];
        let node_min = node.lower.sub(inv_extent);
        let node_max = node.upper.add(inv_extent);
        if bounds_overlap(node_min, node_max, swept_lower, swept_upper)
            && bounds_ray_overlap(node_min, node_max, start, delta)
        {
            if node.is_leaf() {
                for t in node.triangle_offset..node.triangle_offset + node.triangle_count() {
                    let vertices = mesh
                        .triangle(t as usize, reflected)
                        .map(|v| mul(mesh.scale, v));
                    let [a, b, c] = vertices;
                    let triangle_min = min(a, min(b, c)).sub(extent);
                    let triangle_max = max(a, max(b, c)).add(extent);
                    if !bounds_overlap(triangle_min, triangle_max, scaled_lower, scaled_upper)
                        || signed_volume(vertices, center) < 0.0
                    {
                        continue;
                    }
                    let shifted = [Vec3::ZERO, b.sub(a), c.sub(a)];
                    let mut pair = shape_cast(&ShapeCastPairInput {
                        proxy_a: ShapeProxy {
                            points: &shifted,
                            count: 3,
                            radius: 0.0,
                        },
                        proxy_b: input.proxy,
                        transform: Transform {
                            p: a.neg(),
                            ..Transform::IDENTITY
                        },
                        translation_b: input.translation,
                        max_fraction: output.fraction,
                        can_encroach: input.can_encroach,
                    });
                    if pair.hit {
                        pair.point = pair.point.add(a);
                        output = pair;
                        output.triangle_index = t as i32;
                        output.material_index = mesh.materials[t as usize] as i32;
                        let scaled_end = center.add(input.translation.scale(output.fraction));
                        scaled_lower = min(center, scaled_end);
                        scaled_upper = max(center, scaled_end);
                        let end = start.add(delta.scale(output.fraction));
                        swept_lower = min(start, end);
                        swept_upper = max(start, end);
                    }
                }
            } else {
                let left = index + 1;
                let right = index + node.child_offset();
                let (near, far) = if component(delta, node.axis()) > 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                stack[count] = far;
                count += 1;
                index = near;
                continue;
            }
        }
        if count == 0 {
            break;
        }
        count -= 1;
        index = stack[count];
    }
    output
}

pub(crate) fn visit_triangles(
    mesh: Mesh,
    lower: Vec3,
    upper: Vec3,
    flip: bool,
    mut visit: impl FnMut(usize, [Vec3; 3]) -> bool,
) {
    let (lower, upper) = mesh.unscaled_bounds(lower, upper);
    let center = lower.add(upper).scale(0.5);
    let extent = upper.sub(center);
    let mut stack = [0; STACK_SIZE];
    let mut count = 0;
    let mut index = 0;
    loop {
        let node = &mesh.nodes[index];
        if bounds_overlap(node.lower, node.upper, lower, upper) {
            if node.is_leaf() {
                for t in node.triangle_offset..node.triangle_offset + node.triangle_count() {
                    let vertices = mesh.triangle(t as usize, flip);
                    if bounds_triangle_overlap(center, extent, vertices)
                        && !visit(t as usize, vertices.map(|v| mul(mesh.scale, v)))
                    {
                        return;
                    }
                }
            } else {
                stack[count] = index + node.child_offset();
                count += 1;
                index += 1;
                continue;
            }
        }
        if count == 0 {
            break;
        }
        count -= 1;
        index = stack[count];
    }
}

pub fn overlap_mesh(mesh: Mesh, transform: Transform, proxy: ShapeProxy) -> bool {
    let mut points = [core::mem::MaybeUninit::<Vec3>::uninit(); 128];
    let count = proxy.count.min(points.len());
    let inv = transform.invert();
    let matrix = crate::math::Mat3::from_quat(inv.q);
    for i in 0..count {
        points[i].write(matrix.mul_v(proxy.points[i]).add(inv.p));
    }
    // b3MakeLocalProxy writes only the active prefix through the inverse-transform matrix.
    let points = unsafe { core::slice::from_raw_parts(points.as_ptr().cast::<Vec3>(), count) };
    let local = ShapeProxy {
        points,
        count,
        radius: proxy.radius,
    };
    let (lower, upper) = proxy_bounds(local);
    let mut overlap = false;
    visit_triangles(mesh, lower, upper, false, |_, triangle| {
        let input = DistanceInput {
            proxy_a: ShapeProxy {
                points: &triangle,
                count: 3,
                radius: 0.0,
            },
            proxy_b: local,
            transform: Transform::IDENTITY,
            use_radii: true,
        };
        overlap = shape_distance(&input, &mut SimplexCache::empty()).distance < 0.1 * 0.005;
        !overlap
    });
    overlap
}

pub fn collide_mover_mesh(planes: &mut [PlaneResult], mesh: Mesh, mover: &Capsule) -> usize {
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
    let mut count = 0;
    visit_triangles(mesh, lower, upper, mesh.reflected(), |index, triangle| {
        if signed_volume(triangle, center) < 0.0 {
            return true;
        }
        let input = DistanceInput {
            proxy_a: ShapeProxy {
                points: &triangle,
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
                triangle_index: index as i32,
                child_index: 0,
                material_index: mesh.materials[index] as i32,
            };
            count += 1;
        }
        count < planes.len()
    });
    count
}
