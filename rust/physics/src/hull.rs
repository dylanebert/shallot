//! Convex-hull data + support queries, ported from box3d's `hull.c`/`collision.h` (Erin Catto, MIT)
//! via the tumble.js TS port (`src/hull.ts`). Only the read-side the narrowphase touches lives here:
//! the half-edge topology (points/vertices/edges/faces/planes/center) and the two support queries.
//! Hull *construction* (quickhull) stays TS-side and runs once at shape creation. `HullData` is a
//! borrowed view over the geometry pools: native `cargo test` borrows owned `Vec`s, the wasm kernel
//! borrows slices reinterpreted over the static geometry columns (3c.2b).

use crate::math::{Plane, Vec3, FLT_MAX};

// The index fields stay `usize`: on the wasm32 target `usize` is a 4-byte word, so a hull's topology
// pools (one u32 per field) reinterpret directly as `&[HullVertex]` / `&[HullHalfEdge]` / `&[HullFace]`
// via `#[repr(C)]`. Native `cargo test` never reinterprets — it builds owned `Vec`s and borrows them
// into the view — so the 8-byte native `usize` is harmless there.

/// A hull vertex: index of one half-edge with this vertex as origin (b3HullVertex).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullVertex {
    pub edge: usize,
}

/// Half-edge: next (CCW), twin, origin vertex, and left face (b3HullHalfEdge).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullHalfEdge {
    pub next: usize,
    pub twin: usize,
    pub origin: usize,
    pub face: usize,
}

/// A hull face, identified by one of its half-edges (b3HullFace).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HullFace {
    pub edge: usize,
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
