//! Hull construction, ported operation-for-operation from Box3D hull.c (Erin Catto,
//! Dirk Gregorius; MIT). Intrusive lists and recycled face/edge pools retain native walk order.
use crate::math::{
    absf, maxf, minf, scalar_triple_product, Mat3, Plane, Transform, Vec3, FLT_EPSILON, FLT_MAX, PI,
};
use std::ptr::null_mut;

const MAX_VERTICES: usize = 128;
const MAX_FACES: usize = 128;
const MAX_HALF_EDGES: usize = 256;
#[repr(C)]
#[derive(Clone, Copy)]
struct Link {
    prev: *mut Link,
    next: *mut Link,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    link: Link,
    conflict: *mut Face,
    position: Vec3,
    index: i32,
    reachable: bool,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Edge {
    prev: *mut Edge,
    next: *mut Edge,
    origin: *mut Vertex,
    face: *mut Face,
    twin: *mut Edge,
    index: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Face {
    link: Link,
    edge: *mut Edge,
    deleted: bool,
    area: f32,
    plane: Plane,
    centroid: Vec3,
    max_distance: f32,
    conflicts: Vertex,
    max_conflict: *mut Vertex,
    index: i32,
    flipped: bool,
}
#[derive(Clone, Copy)]
struct Frame {
    start: *mut Edge,
    edge: *mut Edge,
    started: bool,
}
struct Builder {
    tolerance: f32,
    min_radius: f32,
    min_outside: f32,
    interior: Vec3,
    orphaned: *mut Link,
    vertices: *mut Link,
    faces: *mut Link,
    vertex_pool: Vec<Vertex>,
    edge_pool: Vec<Edge>,
    face_pool: Vec<Face>,
    edge_free: *mut Edge,
    face_free: *mut Face,
    horizon: Vec<*mut Edge>,
    cone: Vec<*mut Face>,
    merged: Vec<*mut Face>,
    stack: Vec<Frame>,
}
unsafe fn init(h: *mut Link) {
    (*h).prev = h;
    (*h).next = h;
}
unsafe fn contains(n: *mut Link) -> bool {
    !(*n).prev.is_null() && !(*n).next.is_null()
}
unsafe fn insert(n: *mut Link, at: *mut Link) {
    (*n).prev = (*at).prev;
    (*n).next = at;
    (*(*n).prev).next = n;
    (*(*n).next).prev = n;
}
unsafe fn remove(n: *mut Link) {
    (*(*n).prev).next = (*n).next;
    (*(*n).next).prev = (*n).prev;
    (*n).prev = null_mut();
    (*n).next = null_mut();
}
unsafe fn push(h: *mut Link, n: *mut Link) {
    insert(n, (*h).prev);
}
fn is_convex(distance: f32, tolerance: f32) -> bool {
    distance < -tolerance
}
fn separation(p: Plane, v: Vec3) -> f32 {
    p.normal.dot(v) - p.offset
}
fn plane(normal: Vec3, p: Vec3) -> Plane {
    Plane {
        normal,
        offset: normal.dot(p),
    }
}
fn normalized(v: Vec3) -> (Vec3, f32) {
    let length = v.length();
    if v.length_sq() > 1000.0 * crate::math::FLT_MIN {
        (v.scale(1.0 / length), length)
    } else {
        (Vec3::ZERO, 0.0)
    }
}
fn bounds(points: &[Vec3]) -> [Vec3; 2] {
    let mut lo = Vec3::new(FLT_MAX, FLT_MAX, FLT_MAX);
    let mut hi = lo.neg();
    for p in points {
        lo = Vec3::new(minf(lo.x, p.x), minf(lo.y, p.y), minf(lo.z, p.z));
        hi = Vec3::new(maxf(hi.x, p.x), maxf(hi.y, p.y), maxf(hi.z, p.z));
    }
    [lo, hi]
}
unsafe fn ring_count(f: *mut Face) -> usize {
    let mut e = (*f).edge;
    let mut n = 0;
    loop {
        n += 1;
        e = (*e).next;
        if e == (*f).edge {
            break;
        }
    }
    n
}
unsafe fn link_face(f: *mut Face, index: usize, twin: *mut Edge) {
    let mut e = (*f).edge;
    for _ in 0..index {
        e = (*e).next;
    }
    (*e).twin = twin;
    (*twin).twin = e;
}
unsafe fn link_faces(a: *mut Face, ai: usize, b: *mut Face, bi: usize) {
    let mut e = (*b).edge;
    for _ in 0..bi {
        e = (*e).next;
    }
    link_face(a, ai, e);
}
unsafe fn newell(f: *mut Face) {
    let mut count = 0;
    let mut center = Vec3::ZERO;
    let mut normal = Vec3::ZERO;
    let mut e = (*f).edge;
    let origin = (*(*e).origin).position;
    loop {
        let a = (*(*e).origin).position.sub(origin);
        let b = (*(*(*e).twin).origin).position.sub(origin);
        count += 1;
        center = center.add(a);
        normal.x += (a.y - b.y) * (a.z + b.z);
        normal.y += (a.z - b.z) * (a.x + b.x);
        normal.z += (a.x - b.x) * (a.y + b.y);
        e = (*e).next;
        if e == (*f).edge {
            break;
        }
    }
    center = center.scale(1.0 / count as f32).add(origin);
    let length = normal.length();
    normal = normal.scale(1.0 / length);
    (*f).centroid = center;
    (*f).plane = plane(normal, center);
    (*f).area = 0.5 * length;
}
unsafe fn recache(f: *mut Face, min_outside: f32) {
    let mut farthest = null_mut();
    let mut distance = min_outside;
    let h = &raw mut (*f).conflicts.link;
    let mut n = (*h).next;
    while n != h {
        let v = n as *mut Vertex;
        let d = separation((*f).plane, (*v).position);
        if d > distance {
            distance = d;
            farthest = v;
        }
        n = (*n).next;
    }
    (*f).max_conflict = farthest;
    (*f).max_distance = distance;
}
impl Builder {
    fn new(n: usize, m: usize) -> Self {
        // Pool capacities are hull.c's b3ComputeHullWorkSizes. They never grow, keeping links stable.
        let b = Self {
            tolerance: 0.0,
            min_radius: 0.0,
            min_outside: 0.0,
            interior: Vec3::ZERO,
            orphaned: Box::into_raw(Box::new(Link {
                prev: null_mut(),
                next: null_mut(),
            })),
            vertices: Box::into_raw(Box::new(Link {
                prev: null_mut(),
                next: null_mut(),
            })),
            faces: Box::into_raw(Box::new(Link {
                prev: null_mut(),
                next: null_mut(),
            })),
            vertex_pool: Vec::with_capacity(n + 4),
            edge_pool: Vec::with_capacity((24 * m - 48).max(48)),
            face_pool: Vec::with_capacity((5 * m - 10).max(16)),
            edge_free: null_mut(),
            face_free: null_mut(),
            horizon: Vec::with_capacity((3 * m - 6).max(6)),
            cone: Vec::with_capacity((3 * m - 6).max(6)),
            merged: Vec::with_capacity((2 * m - 4).max(4)),
            stack: Vec::with_capacity((2 * m - 4).max(4)),
        };
        unsafe {
            init(b.orphaned);
            init(b.vertices);
            init(b.faces);
        }
        b
    }
    unsafe fn vertex(&mut self, position: Vec3) -> *mut Vertex {
        assert!(self.vertex_pool.len() < self.vertex_pool.capacity());
        self.vertex_pool.push(Vertex {
            link: Link {
                prev: null_mut(),
                next: null_mut(),
            },
            conflict: null_mut(),
            position,
            index: -1,
            reachable: false,
        });
        self.vertex_pool
            .as_mut_ptr()
            .add(self.vertex_pool.len() - 1)
    }
    unsafe fn edge(&mut self) -> *mut Edge {
        let e = if !self.edge_free.is_null() {
            let e = self.edge_free;
            self.edge_free = (*e).next;
            e
        } else {
            assert!(self.edge_pool.len() < self.edge_pool.capacity());
            self.edge_pool.push(std::mem::zeroed());
            self.edge_pool.as_mut_ptr().add(self.edge_pool.len() - 1)
        };
        (*e).index = -1;
        e
    }
    unsafe fn retire_edge(&mut self, e: *mut Edge) {
        (*e).next = self.edge_free;
        self.edge_free = e;
    }
    unsafe fn face(&mut self, a: *mut Vertex, b: *mut Vertex, c: *mut Vertex) -> *mut Face {
        let f = if !self.face_free.is_null() {
            let f = self.face_free;
            self.face_free = (*f).link.next as *mut Face;
            f
        } else {
            assert!(self.face_pool.len() < self.face_pool.capacity());
            self.face_pool.push(std::mem::zeroed());
            self.face_pool.as_mut_ptr().add(self.face_pool.len() - 1)
        };
        (*f).link = Link {
            prev: null_mut(),
            next: null_mut(),
        };
        (*f).max_conflict = null_mut();
        (*f).max_distance = 0.0;
        (*f).index = -1;
        let e1 = self.edge();
        let e2 = self.edge();
        let e3 = self.edge();
        let p1 = (*a).position;
        let p2 = (*b).position;
        let p3 = (*c).position;
        let (normal, length) = normalized(p2.sub(p1).cross(p3.sub(p1)));
        (*f).edge = e1;
        (*f).deleted = false;
        (*f).area = 0.5 * length;
        (*f).centroid = p1.add(p2.add(p3)).scale(1.0 / 3.0);
        (*f).plane = plane(normal, p1);
        (*f).flipped = separation((*f).plane, self.interior) > 0.0;
        init(&raw mut (*f).conflicts.link);
        *e1 = Edge {
            prev: e3,
            next: e2,
            origin: a,
            face: f,
            twin: null_mut(),
            index: -1,
        };
        *e2 = Edge {
            prev: e1,
            next: e3,
            origin: b,
            face: f,
            twin: null_mut(),
            index: -1,
        };
        *e3 = Edge {
            prev: e2,
            next: e1,
            origin: c,
            face: f,
            twin: null_mut(),
            index: -1,
        };
        f
    }
    unsafe fn retire_face(&mut self, f: *mut Face) {
        if contains(&raw mut (*f).link) {
            remove(&raw mut (*f).link);
        }
        (*f).edge = null_mut();
        (*f).link.next = self.face_free as *mut Link;
        self.face_free = f;
    }
    fn tolerance(&mut self, points: &[Vec3]) {
        let [lo, hi] = bounds(points);
        let lo = lo.abs();
        let hi = hi.abs();
        let a = Vec3::new(maxf(lo.x, hi.x), maxf(lo.y, hi.y), maxf(lo.z, hi.z));
        let sum = (a.x + a.y) + a.z;
        let coord = maxf(a.x, maxf(a.y, a.z));
        let distance = minf(1.732_050_808 * coord, sum);
        self.tolerance = (3.0 * distance * 1.01 + coord) * FLT_EPSILON;
        self.min_radius = 4.0 * self.tolerance;
        self.min_outside = 2.0 * self.min_radius;
    }
    unsafe fn initial(&mut self, points: &[Vec3]) -> bool {
        let mut lo = [points[0]; 3];
        let mut hi = lo;
        let mut li = [0; 3];
        let mut ui = li;
        let component = |p: Vec3, a: usize| [p.x, p.y, p.z][a];
        for (i, &p) in points.iter().enumerate().skip(1) {
            for a in 0..3 {
                if component(p, a) < component(lo[a], a) {
                    lo[a] = p;
                    li[a] = i;
                } else if component(p, a) > component(hi[a], a) {
                    hi[a] = p;
                    ui[a] = i;
                }
            }
        }
        let d = [hi[0].x - lo[0].x, hi[1].y - lo[1].y, hi[2].z - lo[2].z];
        let axis = if d[0] < d[1] {
            if d[1] < d[2] {
                2
            } else {
                1
            }
        } else if d[0] < d[2] {
            2
        } else {
            0
        };
        if d[axis] <= 2.0 * self.tolerance {
            return false;
        }
        let i1 = li[axis];
        let mut i2 = ui[axis];
        let mut i3 = usize::MAX;
        let mut i4 = usize::MAX;
        let ab = points[i2].sub(points[i1]);
        let inv = 1.0 / ab.dot(ab);
        let mut distance = 4.0 * self.tolerance * self.tolerance;
        for (i, &p) in points.iter().enumerate() {
            if i == i1 || i == i2 {
                continue;
            }
            let cross = p.sub(points[i1]).cross(ab);
            let d = cross.dot(cross) * inv;
            if d > distance {
                distance = d;
                i3 = i;
            }
        }
        if i3 == usize::MAX {
            return false;
        }
        let normal = points[i2]
            .sub(points[i1])
            .cross(points[i3].sub(points[i1]))
            .normalize();
        let p = plane(normal, points[i1]);
        distance = 2.0 * self.tolerance;
        for (i, &v) in points.iter().enumerate() {
            if i == i1 || i == i2 || i == i3 {
                continue;
            }
            let d = absf(separation(p, v));
            if d > distance {
                distance = d;
                i4 = i;
            }
        }
        if i4 == usize::MAX {
            return false;
        }
        if scalar_triple_product(
            points[i1].sub(points[i4]),
            points[i2].sub(points[i4]),
            points[i3].sub(points[i4]),
        ) < 0.0
        {
            std::mem::swap(&mut i2, &mut i3);
        }
        self.interior = Vec3::ZERO
            .add(points[i1])
            .add(points[i2])
            .add(points[i3])
            .add(points[i4])
            .scale(0.25);
        let v1 = self.vertex(points[i1]);
        push(self.vertices, &raw mut (*v1).link);
        let v2 = self.vertex(points[i2]);
        push(self.vertices, &raw mut (*v2).link);
        let v3 = self.vertex(points[i3]);
        push(self.vertices, &raw mut (*v3).link);
        let v4 = self.vertex(points[i4]);
        push(self.vertices, &raw mut (*v4).link);
        let f1 = self.face(v1, v2, v3);
        push(self.faces, &raw mut (*f1).link);
        let f2 = self.face(v4, v2, v1);
        push(self.faces, &raw mut (*f2).link);
        let f3 = self.face(v4, v3, v2);
        push(self.faces, &raw mut (*f3).link);
        let f4 = self.face(v4, v1, v3);
        push(self.faces, &raw mut (*f4).link);
        link_faces(f1, 0, f2, 1);
        link_faces(f1, 1, f3, 1);
        link_faces(f1, 2, f4, 1);
        link_faces(f2, 0, f3, 2);
        link_faces(f3, 0, f4, 2);
        link_faces(f4, 0, f2, 2);
        for (i, &p) in points.iter().enumerate() {
            if i == i1 || i == i2 || i == i3 || i == i4 {
                continue;
            }
            let mut distance = self.min_outside;
            let mut best = null_mut();
            let mut n = (*self.faces).next;
            while n != self.faces {
                let f = n as *mut Face;
                let d = separation((*f).plane, p);
                if d > distance {
                    distance = d;
                    best = f;
                }
                n = (*n).next;
            }
            if !best.is_null() {
                let v = self.vertex(p);
                self.assign(best, v, distance);
            }
        }
        true
    }
    unsafe fn assign(&mut self, f: *mut Face, v: *mut Vertex, distance: f32) {
        (*v).conflict = f;
        push(&raw mut (*f).conflicts.link, &raw mut (*v).link);
        if distance > (*f).max_distance {
            (*f).max_distance = distance;
            (*f).max_conflict = v;
        }
    }
    unsafe fn next_conflict(&mut self) -> *mut Vertex {
        let mut best = null_mut();
        let mut distance = self.min_outside;
        let mut n = (*self.faces).next;
        while n != self.faces {
            let f = n as *mut Face;
            if !(*f).max_conflict.is_null() && (*f).max_distance > distance {
                distance = (*f).max_distance;
                best = (*f).max_conflict;
            }
            n = (*n).next;
        }
        best
    }
    unsafe fn enter(&mut self, f: *mut Face, entry: *mut Edge) -> Frame {
        (*f).deleted = true;
        let h = &raw mut (*f).conflicts.link;
        let mut n = (*h).next;
        while n != h {
            let v = n as *mut Vertex;
            n = (*n).next;
            (*v).conflict = null_mut();
            remove(&raw mut (*v).link);
            push(self.orphaned, &raw mut (*v).link);
        }
        if entry.is_null() {
            Frame {
                start: (*f).edge,
                edge: (*f).edge,
                started: false,
            }
        } else {
            Frame {
                start: entry,
                edge: (*entry).next,
                started: false,
            }
        }
    }
    unsafe fn horizon(&mut self, apex: *mut Vertex, seed: *mut Face) {
        self.stack.clear();
        let frame = self.enter(seed, null_mut());
        self.stack.push(frame);
        while let Some(f) = self.stack.last_mut() {
            if f.started && f.edge == f.start {
                self.stack.pop();
                continue;
            }
            f.started = true;
            let e = f.edge;
            let t = (*e).twin;
            f.edge = (*e).next;
            if (*(*t).face).deleted {
                continue;
            }
            if separation((*(*t).face).plane, (*apex).position) > self.min_radius {
                let frame = self.enter((*t).face, t);
                self.stack.push(frame);
            } else {
                self.horizon.push(e);
            }
        }
    }
    unsafe fn destroy_edges(&mut self, mut e: *mut Edge, end: *mut Edge) {
        while e != end {
            let next = (*e).next;
            self.retire_edge(e);
            e = next;
        }
    }
    unsafe fn connect_edges(&mut self, prev: *mut Edge, next: *mut Edge) {
        if (*(*prev).twin).face == (*(*next).twin).face {
            if (*(*next).face).edge == next {
                (*(*next).face).edge = prev;
            }
            let twin;
            if ring_count((*(*prev).twin).face) == 3 {
                let d0 = (*prev).twin;
                let d1 = (*next).twin;
                let d2 = (*d1).prev;
                twin = (*d2).twin;
                let opposing = (*d0).face;
                (*opposing).deleted = true;
                self.merged.push(opposing);
                (*prev).next = (*next).next;
                (*(*prev).next).prev = prev;
                (*prev).twin = twin;
                (*twin).twin = prev;
                remove(&raw mut (*(*next).origin).link);
                self.retire_edge(d0);
                self.retire_edge(d1);
                self.retire_edge(d2);
            } else {
                twin = (*next).twin;
                if (*(*twin).face).edge == (*prev).twin {
                    (*(*twin).face).edge = twin;
                }
                (*twin).next = (*(*prev).twin).next;
                (*(*twin).next).prev = twin;
                self.retire_edge((*prev).twin);
                (*prev).next = (*next).next;
                (*(*prev).next).prev = prev;
                (*prev).twin = twin;
                (*twin).twin = prev;
                remove(&raw mut (*(*next).origin).link);
            }
            newell((*twin).face);
            recache((*twin).face, self.min_outside);
        } else {
            (*prev).next = next;
            (*next).prev = prev;
        }
    }
    unsafe fn connect_faces(&mut self, e: *mut Edge) {
        let f = (*e).face;
        let t = (*e).twin;
        let mut ep = (*e).prev;
        let mut en = (*e).next;
        let mut tp = (*t).prev;
        let mut tn = (*t).next;
        while (*(*ep).twin).face == (*t).face {
            ep = (*ep).prev;
            tn = (*tn).next;
        }
        while (*(*en).twin).face == (*t).face {
            en = (*en).next;
            tp = (*tp).prev;
        }
        (*f).edge = ep;
        self.merged.clear();
        self.merged.push((*t).face);
        (*(*t).face).deleted = true;
        (*(*t).face).edge = null_mut();
        let mut absorbed = tn;
        while absorbed != (*tp).next {
            (*absorbed).face = f;
            absorbed = (*absorbed).next;
        }
        self.destroy_edges((*ep).next, en);
        self.destroy_edges((*tp).next, tn);
        self.connect_edges(ep, tn);
        self.connect_edges(tp, en);
        newell(f);
        recache(f, self.min_outside);
        for i in 0..self.merged.len() {
            let dead = self.merged[i];
            let h = &raw mut (*dead).conflicts.link;
            let mut n = (*h).next;
            while n != h {
                let v = n as *mut Vertex;
                n = (*n).next;
                remove(&raw mut (*v).link);
                let distance = separation((*f).plane, (*v).position);
                if distance > self.min_outside {
                    self.assign(f, v, distance);
                } else {
                    push(self.orphaned, &raw mut (*v).link);
                    (*v).conflict = null_mut();
                }
            }
            self.retire_face(dead);
        }
    }
    unsafe fn merge(&mut self, f: *mut Face, coplanar: bool) -> bool {
        let mut e = (*f).edge;
        loop {
            let t = (*e).twin;
            let a = separation((*(*e).face).plane, (*(*t).face).centroid);
            let b = separation((*(*t).face).plane, (*(*e).face).centroid);
            if if coplanar {
                !is_convex(a, self.min_radius) || !is_convex(b, self.min_radius)
            } else {
                a > self.min_radius || b > self.min_radius
            } {
                self.connect_faces(e);
                return true;
            }
            e = (*e).next;
            if e == (*f).edge {
                break;
            }
        }
        false
    }
    unsafe fn add(&mut self, v: *mut Vertex) {
        let f = (*v).conflict;
        (*v).conflict = null_mut();
        remove(&raw mut (*v).link);
        push(self.vertices, &raw mut (*v).link);
        self.horizon.clear();
        self.horizon(v, f);
        self.cone.clear();
        for i in 0..self.horizon.len() {
            let e = self.horizon[i];
            let f = self.face(v, (*e).origin, (*(*e).twin).origin);
            self.cone.push(f);
            link_face(f, 1, (*e).twin);
        }
        let mut f1 = *self.cone.last().unwrap();
        for &f2 in &self.cone {
            link_faces(f1, 2, f2, 0);
            f1 = f2;
        }
        for i in 0..self.cone.len() {
            let f = self.cone[i];
            if (*f).deleted || !(*f).flipped {
                continue;
            }
            (*f).flipped = false;
            let mut area = 0.0;
            let mut best = null_mut();
            let mut e = (*f).edge;
            loop {
                let a = (*(*(*e).twin).face).area;
                if a > area {
                    area = a;
                    best = e;
                }
                e = (*e).next;
                if e == (*f).edge {
                    break;
                }
            }
            self.connect_faces(best);
        }
        for coplanar in [false, true] {
            for i in 0..self.cone.len() {
                let f = self.cone[i];
                if !(*f).deleted {
                    while self.merge(f, coplanar) {}
                }
            }
        }
        let mut n = (*self.orphaned).next;
        while n != self.orphaned {
            let v = n as *mut Vertex;
            n = (*n).next;
            remove(&raw mut (*v).link);
            let mut distance = self.min_outside;
            let mut best = null_mut();
            for &f in &self.cone {
                if !(*f).deleted {
                    let d = separation((*f).plane, (*v).position);
                    if d > distance {
                        distance = d;
                        best = f;
                    }
                }
            }
            if !best.is_null() {
                self.assign(best, v, distance);
            }
        }
        let mut n = (*self.faces).next;
        while n != self.faces {
            let f = n as *mut Face;
            n = (*n).next;
            if (*f).deleted && contains(&raw mut (*f).link) {
                let start = (*f).edge;
                let mut e = start;
                loop {
                    let next = (*e).next;
                    self.retire_edge(e);
                    e = next;
                    if e == start {
                        break;
                    }
                }
                self.retire_face(f);
            }
        }
        for &f in &self.cone {
            if !(*f).deleted {
                push(self.faces, &raw mut (*f).link);
            }
        }
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.orphaned));
            drop(Box::from_raw(self.vertices));
            drop(Box::from_raw(self.faces));
        }
    }
}

/// Owned native b3HullData image, aligned to eight bytes like the hull database's images.
pub struct HullImage(pub Vec<u64>);
impl HullImage {
    fn new(v: usize, e: usize, f: usize) -> Self {
        let align = |n: usize| (n + 7) & !7;
        let offsets = [
            144,
            144 + align(v),
            144 + align(v) + align(12 * v),
            144 + align(v) + align(12 * v) + align(4 * e),
        ];
        let face = offsets[3] + align(16 * f);
        let soa = face + align(f);
        let normals = soa + 12 * ((v + 3) & !3);
        let bytes = normals + 12 * ((f + 3) & !3);
        let mut h = Self(vec![0; bytes / 8]);
        h.0[0] = 0x4a4c9587de57485c;
        for (i, value) in [
            (25, v),
            (26, offsets[0]),
            (27, offsets[1]),
            (28, e),
            (29, offsets[2]),
            (30, f),
            (31, offsets[3]),
            (32, face),
            (33, soa),
            (34, normals),
            (35, bytes),
        ] {
            h.u_mut()[i] = value as u32;
        }
        h
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, self.0.len() * 8) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut u8, self.0.len() * 8) }
    }
    fn u(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u32, self.0.len() * 2) }
    }
    fn u_mut(&mut self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut u32, self.0.len() * 2) }
    }
    fn f_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut f32, self.0.len() * 2) }
    }
    fn vector(&mut self, word: usize, p: Vec3) {
        self.f_mut()[word..word + 3].copy_from_slice(&[p.x, p.y, p.z]);
    }
    fn point(&self, i: usize) -> Vec3 {
        let o = self.u()[27] as usize / 4 + 3 * i;
        let u = self.u();
        Vec3::new(
            f32::from_bits(u[o]),
            f32::from_bits(u[o + 1]),
            f32::from_bits(u[o + 2]),
        )
    }
    fn plane(&self, i: usize) -> Plane {
        let o = self.u()[31] as usize / 4 + 4 * i;
        let u = self.u();
        Plane {
            normal: Vec3::new(
                f32::from_bits(u[o]),
                f32::from_bits(u[o + 1]),
                f32::from_bits(u[o + 2]),
            ),
            offset: f32::from_bits(u[o + 3]),
        }
    }
    fn edge(&self, i: usize) -> [usize; 4] {
        let o = self.u()[29] as usize + 4 * i;
        let b = self.bytes();
        [
            b[o] as usize,
            b[o + 1] as usize,
            b[o + 2] as usize,
            b[o + 3] as usize,
        ]
    }
    fn soa(&mut self) {
        let v = self.u()[25] as usize;
        let f = self.u()[30] as usize;
        let nv = (v + 3) & !3;
        let nf = (f + 3) & !3;
        let vo = self.u()[33] as usize / 4;
        let no = self.u()[34] as usize / 4;
        for i in 0..nv {
            let p = self.point(if i < v { i } else { 0 });
            let a = self.f_mut();
            a[vo + i] = p.x;
            a[vo + nv + i] = p.y;
            a[vo + 2 * nv + i] = p.z;
        }
        for i in 0..f {
            let p = self.plane(i).normal;
            let a = self.f_mut();
            a[no + i] = p.x;
            a[no + nf + i] = p.y;
            a[no + 2 * nf + i] = p.z;
        }
    }
    fn bulk(&mut self) -> bool {
        let mut area = 0.0;
        let mut volume = 0.0;
        let mut center = Vec3::ZERO;
        let origin = self.point(0);
        let (mut xx, mut xy, mut yy, mut xz, mut zz, mut yz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let faces = self.u()[30] as usize;
        let fo = self.u()[32] as usize;
        for i in 0..faces {
            let e1 = self.bytes()[fo + i] as usize;
            let mut e2 = self.edge(e1)[0];
            let mut e3 = self.edge(e2)[0];
            let v1 = self.point(self.edge(e1)[2]).sub(origin);
            loop {
                let v2 = self.point(self.edge(e2)[2]).sub(origin);
                let v3 = self.point(self.edge(e3)[2]).sub(origin);
                area += v2.sub(v1).cross(v3.sub(v1)).length();
                let det = scalar_triple_product(v1, v2, v3);
                volume += det;
                let v4 = v1.add(v2.add(v3));
                center = center.add(v4.scale(det));
                xx += det * (v1.x * v1.x + v2.x * v2.x + v3.x * v3.x + v4.x * v4.x);
                yy += det * (v1.y * v1.y + v2.y * v2.y + v3.y * v3.y + v4.y * v4.y);
                zz += det * (v1.z * v1.z + v2.z * v2.z + v3.z * v3.z + v4.z * v4.z);
                xy += det * (v1.x * v1.y + v2.x * v2.y + v3.x * v3.y + v4.x * v4.y);
                xz += det * (v1.x * v1.z + v2.x * v2.z + v3.x * v3.z + v4.x * v4.z);
                yz += det * (v1.y * v1.z + v2.y * v2.z + v3.y * v3.z + v4.y * v4.z);
                e2 = e3;
                e3 = self.edge(e3)[0];
                if e1 == e3 {
                    break;
                }
            }
        }
        let local = if volume > 0.0 {
            center.scale(0.25 / volume)
        } else {
            Vec3::ZERO
        };
        center = local.add(origin);
        let mut radius = FLT_MAX;
        for i in 0..faces {
            radius = minf(radius, -separation(self.plane(i), center));
        }
        let mass = volume / 6.0;
        let i = Mat3 {
            cx: Vec3::new(yy + zz, -xy, -xz),
            cy: Vec3::new(-xy, xx + zz, -yz),
            cz: Vec3::new(-xz, -yz, xx + yy),
        };
        let s = steiner(mass, local);
        self.vector(13, center);
        self.vector(16, i.cx.scale(1.0 / 120.0).sub(s.cx));
        self.vector(19, i.cy.scale(1.0 / 120.0).sub(s.cy));
        self.vector(22, i.cz.scale(1.0 / 120.0).sub(s.cz));
        self.f_mut()[10] = 0.5 * area;
        self.f_mut()[11] = mass;
        self.f_mut()[12] = radius;
        !(mass <= 0.0 || volume <= 0.0 || area <= 0.0 || radius <= 0.0)
    }
    fn finish(&mut self) {
        self.soa();
        self.0[1] = 0;
        self.0[1] = hash(self.bytes());
    }
}
fn steiner(m: f32, p: Vec3) -> Mat3 {
    let xx = m * (p.y * p.y + p.z * p.z);
    let yy = m * (p.x * p.x + p.z * p.z);
    let zz = m * (p.x * p.x + p.y * p.y);
    let xy = -m * p.x * p.y;
    let xz = -m * p.x * p.z;
    let yz = -m * p.y * p.z;
    Mat3 {
        cx: Vec3::new(xx, xy, xz),
        cy: Vec3::new(xy, yy, yz),
        cz: Vec3::new(xz, yz, zz),
    }
}
/// b3CreateHull, including native count limits and twin-paired emission.
pub fn create_hull(points: &[Vec3], max_vertices: i32) -> Option<HullImage> {
    if points.len() < 4 {
        return None;
    }
    let max = max_vertices.clamp(4, MAX_VERTICES as i32) as usize;
    let mut b = Builder::new(points.len(), max);
    let origin = points[0];
    let shifted: Vec<_> = points.iter().map(|p| p.sub(origin)).collect();
    b.tolerance(&shifted);
    unsafe {
        if !b.initial(&shifted) {
            return None;
        }
        let mut budget = max - 4;
        let mut v = b.next_conflict();
        while !v.is_null() && budget > 0 {
            b.add(v);
            v = b.next_conflict();
            budget -= 1;
        }
        let mut faces = Vec::new();
        let mut edges = Vec::new();
        let mut vertices = Vec::new();
        let mut n = (*b.faces).next;
        let fh = b.faces;
        while n != fh {
            let f = n as *mut Face;
            let mut e = (*f).edge;
            loop {
                (*(*e).origin).reachable = true;
                if (*e).index < 0 {
                    (*e).index = edges.len() as i32;
                    edges.push(e);
                    (*(*e).twin).index = edges.len() as i32;
                    edges.push((*e).twin);
                }
                e = (*e).next;
                if e == (*f).edge {
                    break;
                }
            }
            (*f).plane.offset += (*f).plane.normal.dot(origin);
            (*f).centroid = (*f).centroid.add(origin);
            (*f).index = faces.len() as i32;
            faces.push(f);
            n = (*n).next;
        }
        let mut n = (*b.vertices).next;
        let vh = b.vertices;
        while n != vh {
            let v = n as *mut Vertex;
            n = (*n).next;
            if !(*v).reachable {
                remove(&raw mut (*v).link);
            } else {
                (*v).position = (*v).position.add(origin);
                (*v).index = vertices.len() as i32;
                vertices.push(v);
            }
        }
        b.interior = b.interior.add(origin);
        if vertices.len() as isize - edges.len() as isize / 2 + faces.len() as isize != 2
            || faces.len() < 4
        {
            return None;
        }
        if vertices.len() > MAX_VERTICES || faces.len() > MAX_FACES || edges.len() > MAX_HALF_EDGES
        {
            return None;
        }
        let mut h = HullImage::new(vertices.len(), edges.len(), faces.len());
        let vo = h.u()[26] as usize;
        let po = h.u()[27] as usize / 4;
        let eo = h.u()[29] as usize;
        let pl = h.u()[31] as usize / 4;
        let fo = h.u()[32] as usize;
        for (i, &v) in vertices.iter().enumerate() {
            h.vector(po + 3 * i, (*v).position);
        }
        for (i, &e) in edges.iter().enumerate() {
            h.bytes_mut()[eo + 4 * i..eo + 4 * i + 4].copy_from_slice(&[
                (*(*e).next).index as u8,
                (*(*e).twin).index as u8,
                (*(*e).origin).index as u8,
                (*(*e).face).index as u8,
            ]);
            h.bytes_mut()[vo + (*(*e).origin).index as usize] = i as u8;
        }
        for (i, &f) in faces.iter().enumerate() {
            h.vector(pl + 4 * i, (*f).plane.normal);
            h.f_mut()[pl + 4 * i + 3] = (*f).plane.offset;
            h.bytes_mut()[fo + i] = (*(*f).edge).index as u8;
        }
        let pts: Vec<_> = vertices.iter().map(|&v| (*v).position).collect();
        let [lo, hi] = bounds(&pts);
        h.vector(4, lo);
        h.vector(7, hi);
        if !h.bulk() {
            return None;
        }
        h.finish();
        Some(h)
    }
}
/// b3CreateCylinder.
pub fn create_cylinder(height: f32, radius: f32, offset: f32, sides: usize) -> Option<HullImage> {
    let mut points = Vec::with_capacity(2 * sides);
    let mut alpha = 0.0;
    let delta = 2.0 * PI / sides as f32;
    for _ in 0..sides {
        let cs = crate::math::compute_cos_sin(alpha);
        points.push(Vec3::new(radius * cs.cosine, offset, radius * cs.sine));
        points.push(Vec3::new(
            radius * cs.cosine,
            offset + height,
            radius * cs.sine,
        ));
        alpha += delta;
    }
    create_hull(&points, (2 * sides) as i32)
}
/// b3CreateCone.
pub fn create_cone(height: f32, r1: f32, r2: f32, sides: usize) -> Option<HullImage> {
    let mut points = Vec::with_capacity(2 * sides);
    let mut alpha = 0.0;
    let delta = 2.0 * PI / sides as f32;
    for _ in 0..sides {
        let cs = crate::math::compute_cos_sin(alpha);
        points.push(Vec3::new(r1 * cs.cosine, 0.0, r1 * cs.sine));
        points.push(Vec3::new(r2 * cs.cosine, height, r2 * cs.sine));
        alpha += delta;
    }
    create_hull(&points, (2 * sides) as i32)
}
/// b3CreateRock.
pub fn create_rock(radius: f32) -> Option<HullImage> {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let theta = 2.0 * PI / phi;
    let delta = crate::math::compute_cos_sin(theta);
    let mut cosine = 1.0;
    let mut sine = 0.0;
    let mut points = Vec::with_capacity(10);
    for i in 0..10 {
        let z = 1.0 - (2.0 * i as f32 + 1.0) / 10.0;
        let r = (1.0 - z * z).sqrt();
        points.push(Vec3::new(
            radius * r * cosine,
            radius * r * sine,
            radius * z,
        ));
        let c = cosine;
        let s = sine;
        cosine = delta.cosine * c - delta.sine * s;
        sine = delta.sine * c + delta.cosine * s;
    }
    create_hull(&points, 10)
}
/// b3MakeTransformedBoxHull; the other box constructors supply identity rotation/translation.
pub fn make_transformed_box(hx: f32, hy: f32, hz: f32, xf: Transform) -> HullImage {
    let minimum = 0.2 * 0.005;
    let h = Vec3::new(maxf(minimum, hx), maxf(minimum, hy), maxf(minimum, hz));
    let mut out = HullImage::new(8, 24, 6);
    let rotation = Mat3::from_quat(xf.q);
    let center = h.neg().add(h).scale(0.5);
    let extents = h.sub(h.neg()).scale(0.5);
    let center = xf.point(center);
    let extents = rotation.abs().mul_v(extents);
    out.vector(4, center.sub(extents));
    out.vector(7, center.add(extents));
    let volume = 8.0 * h.x * h.y * h.z;
    out.f_mut()[10] = 8.0 * (h.x * h.y + h.x * h.z + h.y * h.z);
    out.f_mut()[11] = volume;
    out.f_mut()[12] = minf(h.x, minf(h.y, h.z));
    out.vector(13, xf.p);
    let d = h.sub(h.neg());
    let inertia = Mat3 {
        cx: Vec3::new(volume * (d.y * d.y + d.z * d.z) / 12.0, 0.0, 0.0),
        cy: Vec3::new(0.0, volume * (d.x * d.x + d.z * d.z) / 12.0, 0.0),
        cz: Vec3::new(0.0, 0.0, volume * (d.x * d.x + d.y * d.y) / 12.0),
    };
    let inertia = rotation.mul(inertia.mul(rotation.transpose()));
    out.vector(16, inertia.cx);
    out.vector(19, inertia.cy);
    out.vector(22, inertia.cz);
    let vo = out.u()[26] as usize;
    let po = out.u()[27] as usize / 4;
    let eo = out.u()[29] as usize;
    let pl = out.u()[31] as usize / 4;
    let fo = out.u()[32] as usize;
    out.bytes_mut()[vo..vo + 8].copy_from_slice(&[8, 1, 0, 9, 13, 3, 5, 11]);
    out.bytes_mut()[fo..fo + 6].copy_from_slice(&[0, 8, 16, 20, 19, 21]);
    let edges: [[u8; 4]; 24] = [
        [2, 1, 2, 0],
        [17, 0, 1, 5],
        [4, 3, 1, 0],
        [20, 2, 5, 3],
        [6, 5, 5, 0],
        [23, 4, 6, 4],
        [0, 7, 6, 0],
        [18, 6, 2, 2],
        [10, 9, 0, 1],
        [21, 8, 3, 5],
        [12, 11, 3, 1],
        [16, 10, 7, 2],
        [14, 13, 7, 1],
        [19, 12, 4, 4],
        [8, 15, 4, 1],
        [22, 14, 0, 3],
        [7, 17, 3, 2],
        [9, 16, 2, 5],
        [11, 19, 6, 2],
        [5, 18, 7, 4],
        [15, 21, 1, 3],
        [1, 20, 0, 5],
        [3, 23, 4, 3],
        [13, 22, 5, 4],
    ];
    for (i, e) in edges.iter().enumerate() {
        out.bytes_mut()[eo + 4 * i..eo + 4 * i + 4].copy_from_slice(e);
    }
    for (i, p) in [
        Vec3::new(h.x, h.y, h.z),
        Vec3::new(-h.x, h.y, h.z),
        Vec3::new(-h.x, -h.y, h.z),
        Vec3::new(h.x, -h.y, h.z),
        Vec3::new(h.x, h.y, -h.z),
        Vec3::new(-h.x, h.y, -h.z),
        Vec3::new(-h.x, -h.y, -h.z),
        Vec3::new(h.x, -h.y, -h.z),
    ]
    .into_iter()
    .enumerate()
    {
        out.vector(po + 3 * i, xf.point(p));
    }
    for (i, n) in [
        Vec3::new(-1.0, -0.0, -0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-0.0, -1.0, -0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-0.0, -0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let offset = n.dot(if i % 2 == 0 { h.neg() } else { h });
        let n = xf.q.rotate(n);
        out.vector(pl + 4 * i, n);
        out.f_mut()[pl + 4 * i + 3] = offset + n.dot(xf.p);
    }
    out.finish();
    out
}
fn hash(bytes: &[u8]) -> u64 {
    const S: [u64; 8] = [
        0x2d358dccaa6c78a5,
        0x8bb84b93962eacc9,
        0x4b33a62ed433d4a3,
        0x4d5a2da51de1aa47,
        0xa0761d6478bd642f,
        0xe7037ed1a0b428db,
        0x90ed1765281c388c,
        0xaaaaaaaaaaaaaaaa,
    ];
    fn mix(a: u64, b: u64) -> u64 {
        let p = a as u128 * b as u128;
        p as u64 ^ (p >> 64) as u64
    }
    let read = |p: usize| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    let mut seed = mix(S[2], S[1]);
    let mut p = 0;
    let mut i = bytes.len();
    // Every b3HullData image is larger than 112 bytes.
    if i > 112 {
        let mut seeds = [seed; 7];
        loop {
            for lane in 0..7 {
                seeds[lane] = mix(
                    read(p + 16 * lane) ^ S[lane],
                    read(p + 16 * lane + 8) ^ seeds[lane],
                );
            }
            p += 112;
            i -= 112;
            if i <= 112 {
                break;
            }
        }
        seed = seeds.into_iter().fold(0, |a, b| a ^ b);
    }
    for (lane, s) in [2, 2, 1, 1, 2, 1].into_iter().enumerate() {
        if i <= 16 * (lane + 1) {
            break;
        }
        seed = mix(read(p + 16 * lane) ^ S[s], read(p + 16 * lane + 8) ^ seed);
    }
    let a = read(p + i - 16) ^ i as u64;
    let b = read(p + i - 8);
    let product = (a ^ S[1]) as u128 * (b ^ seed) as u128;
    let result = mix(
        product as u64 ^ S[7],
        (product >> 64) as u64 ^ S[1] ^ i as u64,
    );
    if result == 0 {
        1
    } else {
        result
    }
}
