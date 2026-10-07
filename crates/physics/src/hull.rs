//! Convex-hull data + support queries, ported from box3d's `hull.c`/`collision.h` (Erin Catto, MIT)
//! via the upstream TS port (`src/hull.ts`). Only the read-side the narrowphase touches lives here:
//! the half-edge topology (points/vertices/edges/faces/planes/center) and the two support queries.
//! Hull *construction* (quickhull) stays TS-side and runs once at shape creation. `HullData` is a
//! borrowed view over the geometry pools: native `cargo test` borrows owned `Vec`s, the wasm kernel
//! borrows slices reinterpreted over the static geometry columns (3c.2b).

use crate::math::{Plane, Vec3, FLT_MAX, HUGE};
use crate::simd::FloatW;
use std::borrow::Cow;

/// Derive Box3D's padded SoA stream; vertices repeat the first point, normals have zero tails.
pub fn soa_vectors(points: impl ExactSizeIterator<Item = Vec3>, repeat_first: bool) -> Vec<f32> {
    let count = points.len();
    let n = (count + 3) & !3;
    let mut out = vec![0.0; n * 3];
    for (i, p) in points.enumerate() {
        out[i] = p.x;
        out[n + i] = p.y;
        out[2 * n + i] = p.z;
    }
    if repeat_first {
        for i in count..n {
            out[i] = out[0];
            out[n + i] = out[n];
            out[2 * n + i] = out[2 * n];
        }
    }
    out
}

/// A hull vertex: index of one half-edge with this vertex as origin (b3HullVertex).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullVertex {
    pub edge: u8,
}

/// Half-edge: next (CCW), twin, origin vertex, and left face (b3HullHalfEdge).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullHalfEdge {
    pub next: u8,
    pub twin: u8,
    pub origin: u8,
    pub face: u8,
}

/// A hull face, identified by one of its half-edges (b3HullFace).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullFace {
    pub edge: u8,
}

/// The read-side of a convex hull the narrowphase consumes (b3HullData), as a borrowed view over the
/// geometry pools. Native gold borrows owned `Vec`s; the wasm kernel borrows slices reinterpreted over
/// the static geometry columns (3c.2b) — same view type either way.
pub struct HullData<'a> {
    pub center: Vec3,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub points: &'a [Vec3],
    pub soa_points: Cow<'a, [f32]>,
    pub soa_normals: Cow<'a, [f32]>,
    pub vertices: &'a [HullVertex],
    pub edges: &'a [HullHalfEdge],
    pub faces: &'a [HullFace],
    pub planes: &'a [Plane],
}

impl HullData<'_> {
    /// Index of the hull vertex furthest along `direction` (b3FindHullSupportVertex).
    pub fn support_vertex(&self, direction: Vec3) -> usize {
        let mut best_index = 0;
        let mut best_dot = -FLT_MAX;
        for index in 0..self.vertex_count {
            let dot = direction.dot(self.points[index]);
            if dot > best_dot {
                best_index = index;
                best_dot = dot;
            }
        }
        best_index
    }

    /// SIMD hull support reduction used by Box3D's `b3GetSupportWide`.
    ///
    /// The candidate is `bias - dot` with the low seven mantissa bits replaced by the
    /// vertex index.  The stored SoA stream repeats vertex zero in padded lanes; iterating
    /// those lanes here is intentional because the embedded index is what prevents a tail
    /// lane from winning.
    #[inline]
    pub fn support_vertex_wide(&self, direction: Vec3, bias: f32) -> usize {
        let soa_count = (self.vertex_count + 3) & !3;
        let nx = FloatW::splat(direction.x);
        let ny = FloatW::splat(direction.y);
        let nz = FloatW::splat(direction.z);
        let bias = FloatW::splat(bias);
        let mut minimum = FloatW::splat(HUGE);
        for i in (0..soa_count).step_by(4) {
            let x = FloatW::load(&self.soa_points[i..]);
            let y = FloatW::load(&self.soa_points[soa_count + i..]);
            let z = FloatW::load(&self.soa_points[2 * soa_count + i..]);
            // b3GetSupportWide deliberately differs from b3Dot3W's association.
            let dot = nz.mul(z).add(ny.mul(y).add(nx.mul(x)));
            minimum = minimum.min(bias.sub(dot).embed_index(i));
        }
        minimum.min_index(7)
    }

    /// Index of the hull face whose normal is most aligned with `direction` (b3FindHullSupportFace).
    pub fn support_face(&self, direction: Vec3) -> usize {
        let mut best_index = 0;
        let mut best_dot = -FLT_MAX;
        for index in 0..self.face_count {
            let dot = self.planes[index].normal.dot(direction);
            if dot > best_dot {
                best_dot = dot;
                best_index = index;
            }
        }
        best_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soa_padding_repeats_vertices_but_zero_fills_normals() {
        let vectors = [Vec3::new(1.0, 2.0, 3.0)];
        assert_eq!(
            soa_vectors(vectors.into_iter(), true),
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        assert_eq!(
            soa_vectors(vectors.into_iter(), false),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn wide_support_keeps_box3d_huge_sentinel_when_all_candidates_exceed_it() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let hull = HullData {
            center: Vec3::ZERO,
            vertex_count: points.len(),
            edge_count: 0,
            face_count: 0,
            points: &points,
            soa_points: soa_vectors(points.iter().copied(), true).into(),
            soa_normals: Vec::new().into(),
            vertices: &[],
            edges: &[],
            faces: &[],
            planes: &[],
        };
        assert_eq!(
            hull.support_vertex_wide(Vec3::new(1.0, 0.0, 0.0), 200_000.0),
            0
        );
    }

    #[test]
    fn wide_support_embeds_indices_and_never_selects_padding() {
        for count in 1..=128 {
            let points: Vec<_> = (0..count)
                .map(|i| Vec3::new((i % 5) as f32, (i % 7) as f32, (i % 3) as f32))
                .collect();
            let hull = HullData {
                center: Vec3::ZERO,
                vertex_count: count,
                edge_count: 0,
                face_count: 0,
                points: &points,
                soa_points: soa_vectors(points.iter().copied(), true).into(),
                soa_normals: Vec::new().into(),
                vertices: &[],
                edges: &[],
                faces: &[],
                planes: &[],
            };
            for direction in [
                Vec3::ZERO,
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(-1.0, 1.0, -2.0),
            ] {
                let bias = 64.0;
                let mut best = f32::INFINITY;
                let mut index = 0;
                for (i, p) in points.iter().enumerate() {
                    let dot = direction.z * p.z + (direction.y * p.y + direction.x * p.x);
                    let value = f32::from_bits(((bias - dot).to_bits() & !0x7f) | i as u32);
                    if value < best {
                        best = value;
                        index = i;
                    }
                }
                assert_eq!(
                    hull.support_vertex_wide(direction, bias),
                    index,
                    "count {count}"
                );
            }
        }
    }
}
