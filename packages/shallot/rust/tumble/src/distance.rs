//! GJK closest-point distance, ported op-for-op from box3d's `distance.c` (Erin Catto, MIT) via the
//! tumble.js TS port (`src/distance.ts`). Only the `shape_distance` path lives here — the narrowphase
//! (hull/sphere, hull/capsule) consumes it; shape-cast and time-of-impact are CCD and stay TS-side.
//!
//! Rust `f32` is native IEEE-754 with no FMA contraction, so each TS `f32(...)`-wrapped op maps to one
//! Rust op with the same operand order (see `math.rs`).

use crate::math::{blend2, blend3, scalar_triple_product, Transform, Vec3, FLT_EPSILON, FLT_MAX, FLT_MIN};

const MAX_SIMPLEX_VERTICES: usize = 4;
const MAX_GJK_ITERATIONS: i32 = 32;

/// A convex point cloud with an external radius (b3ShapeProxy). `points` may hold more than `count`
/// entries; only the first `count` participate.
#[derive(Clone, Copy)]
pub struct ShapeProxy<'a> {
    pub points: &'a [Vec3],
    pub count: usize,
    pub radius: f32,
}

/// Warm-start data for the GJK simplex; zero-initialize on the first call (b3SimplexCache).
#[derive(Clone, Copy)]
pub struct SimplexCache {
    pub metric: f32,
    pub count: usize,
    pub index_a: [usize; 4],
    pub index_b: [usize; 4],
}

impl SimplexCache {
    /// A fresh, empty simplex cache (b3_emptySimplexCache).
    pub fn empty() -> SimplexCache {
        SimplexCache {
            metric: 0.0,
            count: 0,
            index_a: [0; 4],
            index_b: [0; 4],
        }
    }
}

/// Input for {@link shape_distance} (b3DistanceInput).
pub struct DistanceInput<'a> {
    pub proxy_a: ShapeProxy<'a>,
    pub proxy_b: ShapeProxy<'a>,
    /// Transform of shape B in shape A's frame.
    pub transform: Transform,
    pub use_radii: bool,
}

/// Output of {@link shape_distance} (b3DistanceOutput).
#[derive(Clone, Copy)]
pub struct DistanceOutput {
    pub point_a: Vec3,
    pub point_b: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub iterations: i32,
}

/// One simplex vertex: the Minkowski support point and its barycentric weight (b3SimplexVertex).
#[derive(Clone, Copy)]
struct SimplexVertex {
    /// Support point in proxy A.
    w_a: Vec3,
    /// Support point in proxy B.
    w_b: Vec3,
    /// w_b - w_a.
    w: Vec3,
    /// Barycentric coordinate.
    a: f32,
    index_a: usize,
    index_b: usize,
}

impl SimplexVertex {
    const ZERO: SimplexVertex = SimplexVertex {
        w_a: Vec3::ZERO,
        w_b: Vec3::ZERO,
        w: Vec3::ZERO,
        a: 0.0,
        index_a: 0,
        index_b: 0,
    };
}

/// The GJK simplex: up to four vertices (b3Simplex).
#[derive(Clone, Copy)]
struct Simplex {
    vertices: [SimplexVertex; 4],
    count: usize,
}

impl Simplex {
    fn empty() -> Simplex {
        Simplex {
            vertices: [SimplexVertex::ZERO; 4],
            count: 0,
        }
    }
}

/// Index of the proxy point furthest along `axis` (b3GetProxySupport).
pub fn get_proxy_support(proxy: &ShapeProxy, axis: Vec3) -> usize {
    let points = proxy.points;
    let origin = points[0];
    let mut max_index = 0;
    let mut max_projection = 0.0;
    for index in 1..proxy.count {
        let projection = axis.dot(points[index].sub(origin));
        if projection > max_projection {
            max_index = index;
            max_projection = projection;
        }
    }
    max_index
}

/// Index of the point furthest along `axis` in a raw point cloud (b3GetPointSupport).
pub fn get_point_support(points: &[Vec3], count: usize, axis: Vec3) -> usize {
    let origin = points[0];
    let mut max_index = 0;
    let mut max_projection = 0.0;
    for index in 1..count {
        let projection = axis.dot(points[index].sub(origin));
        if projection > max_projection {
            max_index = index;
            max_projection = projection;
        }
    }
    max_index
}

// --- barycentric coordinates ----------------------------------------------------------------

fn barycentric_edge(a: Vec3, b: Vec3) -> [f32; 3] {
    let ab = b.sub(a);
    let divisor = ab.dot(ab);
    [b.dot(ab), -a.dot(ab), divisor]
}

fn barycentric_tri(a: Vec3, b: Vec3, c: Vec3) -> [f32; 4] {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let b_x_c = b.cross(c);
    let c_x_a = c.cross(a);
    let a_x_b = a.cross(b);
    let ab_x_ac = ab.cross(ac);
    let divisor = ab_x_ac.dot(ab_x_ac);
    [b_x_c.dot(ab_x_ac), c_x_a.dot(ab_x_ac), a_x_b.dot(ab_x_ac), divisor]
}

fn barycentric_tet(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [f32; 5] {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ad = d.sub(a);
    let divisor = scalar_triple_product(ab, ac, ad);
    let sign = if divisor < 0.0 { -1.0 } else { 1.0 };
    [
        sign * scalar_triple_product(b, c, d),
        sign * scalar_triple_product(a, d, c),
        sign * scalar_triple_product(a, b, d),
        sign * scalar_triple_product(a, c, b),
        sign * divisor,
    ]
}

// --- metric ---------------------------------------------------------------------------------

fn get_metric(simplex: &Simplex) -> f32 {
    let vs = &simplex.vertices;
    match simplex.count {
        1 => 0.0,
        2 => vs[0].w.distance(vs[1].w),
        3 => {
            let cross = vs[1].w.sub(vs[0].w).cross(vs[2].w.sub(vs[0].w));
            cross.length() / 2.0
        }
        4 => {
            scalar_triple_product(
                vs[1].w.sub(vs[0].w),
                vs[2].w.sub(vs[0].w),
                vs[3].w.sub(vs[0].w),
            ) / 6.0
        }
        _ => 0.0,
    }
}

fn write_cache(cache: &mut SimplexCache, simplex: &Simplex) {
    let count = simplex.count;
    cache.metric = get_metric(simplex);
    cache.count = count;
    for index in 0..count {
        cache.index_a[index] = simplex.vertices[index].index_a;
        cache.index_b[index] = simplex.vertices[index].index_b;
    }
}

// --- simplex solvers ------------------------------------------------------------------------

fn solve_simplex2(simplex: &mut Simplex) -> bool {
    let vs = &mut simplex.vertices;
    let a = vs[0].w;
    let b = vs[1].w;
    let ab = b.sub(a);
    let divisor = ab.dot(ab);
    let u = b.dot(ab);
    let v = -a.dot(ab);

    // V( A )
    if v <= 0.0 {
        simplex.count = 1;
        vs[0].a = 1.0;
        return true;
    }
    // V( B )
    if u <= 0.0 {
        simplex.count = 1;
        vs[0] = vs[1];
        vs[0].a = 1.0;
        return true;
    }
    // Edge region
    if divisor <= 0.0 {
        return false;
    }

    let denominator = 1.0 / divisor;
    vs[0].a = denominator * u;
    vs[1].a = denominator * v;
    true
}

fn solve_simplex3(simplex: &mut Simplex) -> bool {
    // Snapshot the simplex (aliasing: the slots below get overwritten).
    let v1 = simplex.vertices[0];
    let v2 = simplex.vertices[1];
    let v3 = simplex.vertices[2];

    let w_ab = barycentric_edge(v1.w, v2.w);
    let w_bc = barycentric_edge(v2.w, v3.w);
    let w_ca = barycentric_edge(v3.w, v1.w);

    let vs = &mut simplex.vertices;

    // VR( A )
    if w_ab[1] <= 0.0 && w_ca[0] <= 0.0 {
        simplex.count = 1;
        vs[0] = v1;
        vs[0].a = 1.0;
        return true;
    }
    // VR( B )
    if w_bc[1] <= 0.0 && w_ab[0] <= 0.0 {
        simplex.count = 1;
        vs[0] = v2;
        vs[0].a = 1.0;
        return true;
    }
    // VR( C )
    if w_ca[1] <= 0.0 && w_bc[0] <= 0.0 {
        simplex.count = 1;
        vs[0] = v3;
        vs[0].a = 1.0;
        return true;
    }

    let w_abc = barycentric_tri(v1.w, v2.w, v3.w);

    // VR( AB )
    if w_abc[2] <= 0.0 && w_ab[0] > 0.0 && w_ab[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v1;
        vs[1] = v2;
        let divisor = w_ab[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_ab[0] / divisor;
        vs[1].a = w_ab[1] / divisor;
        return true;
    }
    // VR( BC )
    if w_abc[0] <= 0.0 && w_bc[0] > 0.0 && w_bc[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v2;
        vs[1] = v3;
        let divisor = w_bc[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_bc[0] / divisor;
        vs[1].a = w_bc[1] / divisor;
        return true;
    }
    // VR( CA )
    if w_abc[1] <= 0.0 && w_ca[0] > 0.0 && w_ca[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v3;
        vs[1] = v1;
        let divisor = w_ca[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_ca[0] / divisor;
        vs[1].a = w_ca[1] / divisor;
        return true;
    }

    // Face region
    let divisor = w_abc[3];
    if divisor <= 0.0 {
        return false;
    }
    vs[0].a = w_abc[0] / divisor;
    vs[1].a = w_abc[1] / divisor;
    vs[2].a = w_abc[2] / divisor;
    true
}

fn solve_simplex4(simplex: &mut Simplex) -> bool {
    let v_a = simplex.vertices[0];
    let v_b = simplex.vertices[1];
    let v_c = simplex.vertices[2];
    let v_d = simplex.vertices[3];

    let w_ab = barycentric_edge(v_a.w, v_b.w);
    let w_ac = barycentric_edge(v_a.w, v_c.w);
    let w_ad = barycentric_edge(v_a.w, v_d.w);
    let w_bc = barycentric_edge(v_b.w, v_c.w);
    let w_cd = barycentric_edge(v_c.w, v_d.w);
    let w_db = barycentric_edge(v_d.w, v_b.w);

    let vs = &mut simplex.vertices;

    // VR( A )
    if w_ab[1] <= 0.0 && w_ac[1] <= 0.0 && w_ad[1] <= 0.0 {
        simplex.count = 1;
        vs[0] = v_a;
        vs[0].a = 1.0;
        return true;
    }
    // VR( B )
    if w_ab[0] <= 0.0 && w_db[0] <= 0.0 && w_bc[1] <= 0.0 {
        simplex.count = 1;
        vs[0] = v_b;
        vs[0].a = 1.0;
        return true;
    }
    // VR( C )
    if w_ac[0] <= 0.0 && w_bc[0] <= 0.0 && w_cd[1] <= 0.0 {
        simplex.count = 1;
        vs[0] = v_c;
        vs[0].a = 1.0;
        return true;
    }
    // VR( D )
    if w_ad[0] <= 0.0 && w_cd[0] <= 0.0 && w_db[1] <= 0.0 {
        simplex.count = 1;
        vs[0] = v_d;
        vs[0].a = 1.0;
        return true;
    }

    let w_acb = barycentric_tri(v_a.w, v_c.w, v_b.w);
    let w_abd = barycentric_tri(v_a.w, v_b.w, v_d.w);
    let w_adc = barycentric_tri(v_a.w, v_d.w, v_c.w);
    let w_bcd = barycentric_tri(v_b.w, v_c.w, v_d.w);

    // VR( AB )
    if w_abd[2] <= 0.0 && w_acb[1] <= 0.0 && w_ab[0] > 0.0 && w_ab[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_a;
        vs[1] = v_b;
        let divisor = w_ab[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_ab[0] / divisor;
        vs[1].a = w_ab[1] / divisor;
        return true;
    }
    // VR( AC )
    if w_acb[2] <= 0.0 && w_adc[1] <= 0.0 && w_ac[0] > 0.0 && w_ac[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_a;
        vs[1] = v_c;
        let divisor = w_ac[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_ac[0] / divisor;
        vs[1].a = w_ac[1] / divisor;
        return true;
    }
    // VR( AD )
    if w_adc[2] <= 0.0 && w_abd[1] <= 0.0 && w_ad[0] > 0.0 && w_ad[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_a;
        vs[1] = v_d;
        let divisor = w_ad[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_ad[0] / divisor;
        vs[1].a = w_ad[1] / divisor;
        return true;
    }
    // VR( BC )
    if w_acb[0] <= 0.0 && w_bcd[2] <= 0.0 && w_bc[0] > 0.0 && w_bc[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_b;
        vs[1] = v_c;
        let divisor = w_bc[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_bc[0] / divisor;
        vs[1].a = w_bc[1] / divisor;
        return true;
    }
    // VR( CD )
    if w_adc[0] <= 0.0 && w_bcd[0] <= 0.0 && w_cd[0] > 0.0 && w_cd[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_c;
        vs[1] = v_d;
        let divisor = w_cd[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_cd[0] / divisor;
        vs[1].a = w_cd[1] / divisor;
        return true;
    }
    // VR( DB )
    if w_abd[0] <= 0.0 && w_bcd[1] <= 0.0 && w_db[0] > 0.0 && w_db[1] > 0.0 {
        simplex.count = 2;
        vs[0] = v_d;
        vs[1] = v_b;
        let divisor = w_db[2];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_db[0] / divisor;
        vs[1].a = w_db[1] / divisor;
        return true;
    }

    let w_abcd = barycentric_tet(v_a.w, v_b.w, v_c.w, v_d.w);

    // VR( ACB )
    if w_abcd[3] < 0.0 && w_acb[0] > 0.0 && w_acb[1] > 0.0 && w_acb[2] > 0.0 {
        simplex.count = 3;
        vs[0] = v_a;
        vs[1] = v_c;
        vs[2] = v_b;
        let divisor = w_acb[3];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_acb[0] / divisor;
        vs[1].a = w_acb[1] / divisor;
        vs[2].a = w_acb[2] / divisor;
        return true;
    }
    // VR( ABD )
    if w_abcd[2] < 0.0 && w_abd[0] > 0.0 && w_abd[1] > 0.0 && w_abd[2] > 0.0 {
        simplex.count = 3;
        vs[0] = v_a;
        vs[1] = v_b;
        vs[2] = v_d;
        let divisor = w_abd[3];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_abd[0] / divisor;
        vs[1].a = w_abd[1] / divisor;
        vs[2].a = w_abd[2] / divisor;
        return true;
    }
    // VR( ADC )
    if w_abcd[1] < 0.0 && w_adc[0] > 0.0 && w_adc[1] > 0.0 && w_adc[2] > 0.0 {
        simplex.count = 3;
        vs[0] = v_a;
        vs[1] = v_d;
        vs[2] = v_c;
        let divisor = w_adc[3];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_adc[0] / divisor;
        vs[1].a = w_adc[1] / divisor;
        vs[2].a = w_adc[2] / divisor;
        return true;
    }
    // VR( BCD )
    if w_abcd[0] < 0.0 && w_bcd[0] > 0.0 && w_bcd[1] > 0.0 && w_bcd[2] > 0.0 {
        simplex.count = 3;
        vs[0] = v_b;
        vs[1] = v_c;
        vs[2] = v_d;
        let divisor = w_bcd[3];
        if divisor <= 0.0 {
            return false;
        }
        vs[0].a = w_bcd[0] / divisor;
        vs[1].a = w_bcd[1] / divisor;
        vs[2].a = w_bcd[2] / divisor;
        return true;
    }

    // *** Inside tetrahedron ***
    let divisor = w_abcd[4];
    if divisor <= 0.0 {
        return false;
    }
    vs[0].a = w_abcd[0] / divisor;
    vs[1].a = w_abcd[1] / divisor;
    vs[2].a = w_abcd[2] / divisor;
    vs[3].a = w_abcd[3] / divisor;
    true
}

fn compute_witness_points(simplex: &Simplex) -> (Vec3, Vec3) {
    let vs = &simplex.vertices;
    match simplex.count {
        1 => (vs[0].w_a, vs[0].w_b),
        2 => (
            blend2(vs[0].a, vs[0].w_a, vs[1].a, vs[1].w_a),
            blend2(vs[0].a, vs[0].w_b, vs[1].a, vs[1].w_b),
        ),
        3 => (
            blend3(vs[0].a, vs[0].w_a, vs[1].a, vs[1].w_a, vs[2].a, vs[2].w_a),
            blend3(vs[0].a, vs[0].w_b, vs[1].a, vs[1].w_b, vs[2].a, vs[2].w_b),
        ),
        4 => {
            // Force identical points and zero distance.
            let sum = blend2(vs[0].a, vs[0].w_a, vs[1].a, vs[1].w_a).add(blend2(
                vs[2].a, vs[2].w_a, vs[3].a, vs[3].w_a,
            ));
            (sum, sum)
        }
        _ => (Vec3::ZERO, Vec3::ZERO),
    }
}

// --- shape distance -------------------------------------------------------------------------

/// Closest points between two convex proxies via GJK (b3ShapeDistance).
///
/// `cache` warm-starts the simplex and is updated in place; zero-initialize it on the first call.
/// The query runs in shape A's frame using `input.transform`, the relative pose of B in A.
pub fn shape_distance(input: &DistanceInput, cache: &mut SimplexCache) -> DistanceOutput {
    let xf_t = input.transform;
    let m = crate::math::Mat3::from_quat(xf_t.q);
    let mt = m.transpose();

    let proxy_a = &input.proxy_a;
    let proxy_b = &input.proxy_b;

    let mut simplex = Simplex::empty();

    simplex.count = cache.count;
    for i in 0..cache.count {
        let index1 = cache.index_a[i];
        let index2 = cache.index_b[i];
        let vertex1 = proxy_a.points[index1];
        let vertex2 = m.mul_v(proxy_b.points[index2]).add(xf_t.p);
        simplex.vertices[i].index_a = index1;
        simplex.vertices[i].index_b = index2;
        simplex.vertices[i].w_a = vertex1;
        simplex.vertices[i].w_b = vertex2;
        simplex.vertices[i].w = vertex2.sub(vertex1);
        simplex.vertices[i].a = 0.0;
    }

    // Flush the simplex if its metric drifted substantially from the cached one.
    if simplex.count > 0 {
        let metric1 = cache.metric;
        let metric2 = get_metric(&simplex);
        if 2.0 * metric1 < metric2 || metric2 < 0.5 * metric1 || metric2 < FLT_EPSILON {
            simplex.count = 0;
        }
    }

    if simplex.count == 0 {
        let vertex1 = proxy_a.points[0];
        let vertex2 = m.mul_v(proxy_b.points[0]).add(xf_t.p);
        simplex.count = 1;
        simplex.vertices[0].index_a = 0;
        simplex.vertices[0].index_b = 0;
        simplex.vertices[0].w_a = vertex1;
        simplex.vertices[0].w_b = vertex2;
        simplex.vertices[0].w = vertex2.sub(vertex1);
        simplex.vertices[0].a = 0.0;
    }

    let mut backup = Simplex::empty();

    let mut output = DistanceOutput {
        point_a: Vec3::ZERO,
        point_b: Vec3::ZERO,
        normal: Vec3::ZERO,
        distance: 0.0,
        iterations: 0,
    };

    let mut distance_sq = FLT_MAX;
    let mut normal = Vec3::ZERO;

    let mut iteration = 0;
    while iteration < MAX_GJK_ITERATIONS {
        let solved = match simplex.count {
            1 => {
                simplex.vertices[0].a = 1.0;
                true
            }
            2 => solve_simplex2(&mut simplex),
            3 => solve_simplex3(&mut simplex),
            4 => solve_simplex4(&mut simplex),
            _ => false,
        };

        if !solved {
            simplex = backup;
            break;
        }

        if simplex.count == MAX_SIMPLEX_VERTICES {
            let w = compute_witness_points(&simplex);
            output.point_a = w.0;
            output.point_b = w.1;
            return output;
        }

        let old_distance_sq = distance_sq;
        let vs = &simplex.vertices;

        let closest_point = match simplex.count {
            1 => vs[0].w,
            2 => blend2(vs[0].a, vs[0].w, vs[1].a, vs[1].w),
            3 => blend3(vs[0].a, vs[0].w, vs[1].a, vs[1].w, vs[2].a, vs[2].w),
            _ => Vec3::ZERO,
        };

        distance_sq = closest_point.dot(closest_point);

        if distance_sq >= old_distance_sq {
            simplex = backup;
            break;
        }

        let search_direction = match simplex.count {
            1 => vs[0].w.neg(),
            2 => {
                let a = vs[0].w;
                let b = vs[1].w;
                let ab = b.sub(a);
                ab.cross(a.neg()).cross(ab)
            }
            3 => {
                let a = vs[0].w;
                let b = vs[1].w;
                let c = vs[2].w;
                let ab = b.sub(a);
                let ac = c.sub(a);
                let n = ab.cross(ac);
                if n.dot(a) < 0.0 {
                    n
                } else {
                    n.neg()
                }
            }
            _ => Vec3::ZERO,
        };

        if search_direction.length_sq() < 1000.0 * FLT_MIN {
            // The origin is contained by a line segment or triangle: the shapes overlap.
            let w = compute_witness_points(&simplex);
            output.point_a = w.0;
            output.point_b = w.1;
            return output;
        }

        normal = search_direction.neg();

        let index_a = get_proxy_support(&input.proxy_a, search_direction.neg());
        let support_a = input.proxy_a.points[index_a];
        let search_direction2 = mt.mul_v(search_direction);
        let index_b = get_proxy_support(&input.proxy_b, search_direction2);
        let support_b = m.mul_v(input.proxy_b.points[index_b]).add(xf_t.p);

        backup = simplex;

        // Duplicate support point is the main termination criterion.
        let mut duplicate = false;
        for i in 0..simplex.count {
            if simplex.vertices[i].index_a == index_a && simplex.vertices[i].index_b == index_b {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            break;
        }

        let n = simplex.count;
        simplex.vertices[n].index_a = index_a;
        simplex.vertices[n].index_b = index_b;
        simplex.vertices[n].w_a = support_a;
        simplex.vertices[n].w_b = support_b;
        simplex.vertices[n].w = support_b.sub(support_a);
        simplex.count += 1;

        iteration += 1;
    }

    normal = normal.normalize();
    if !normal.is_normalized() {
        // Treat as overlap.
        return output;
    }

    let w = compute_witness_points(&simplex);
    write_cache(cache, &simplex);

    output.point_a = w.0;
    output.point_b = w.1;
    output.distance = w.0.distance(w.1);
    output.normal = normal;
    output.iterations = iteration;

    if input.use_radii {
        let r_a = input.proxy_a.radius;
        let r_b = input.proxy_b.radius;
        output.distance = crate::math::maxf(0.0, output.distance - r_a - r_b);
        // Keep closest points on the perimeter even if overlapped, so they move smoothly.
        output.point_a = output.point_a.mul_add(r_a, normal);
        output.point_b = output.point_b.mul_sub(r_b, normal);
    }

    output
}
