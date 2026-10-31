//! Triangle-only port of Mikkelsen's mikktspace.c at 3e895b49.
//! Altered source: Rust, flat Shallot vertex streams, triangle faces only.
//! See ../licenses/LICENSE-ZLIB for Mikkelsen's original notice.

// Keep the reference's phase boundaries and arithmetic order visible.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_clamp)]

type V = [f32; 3];
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(s: f32, v: V) -> V {
    [s * v[0], s * v[1], s * v[2]]
}
fn dot(a: V, b: V) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn length(v: V) -> f32 {
    dot(v, v).sqrt()
}
fn not_zero(x: f32) -> bool {
    x.abs() > f32::MIN_POSITIVE
}
fn vnot_zero(v: V) -> bool {
    v.into_iter().any(not_zero)
}
fn normalize(v: V) -> V {
    scale(1.0 / length(v), v)
}
fn project(v: V, n: V) -> V {
    let v = sub(v, scale(dot(n, v), n));
    if vnot_zero(v) {
        normalize(v)
    } else {
        v
    }
}

pub struct Mesh<'a> {
    pub vertices: &'a [u32],
    pub uv: Option<&'a [u32]>,
    pub indices: &'a [u32],
}
impl Mesh<'_> {
    fn vertex(&self, index: usize) -> usize {
        self.indices[(index >> 2) * 3 + (index & 3)] as usize
    }
    fn position(&self, index: usize) -> V {
        let i = self.vertex(index) * 8;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn normal(&self, index: usize) -> V {
        let i = self.vertex(index) * 8 + 4;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn tex_coord(&self, index: usize) -> V {
        let i = self.vertex(index);
        let (x, y) = if let Some(uv) = self.uv {
            (uv[i * 2], uv[i * 2 + 1])
        } else {
            (self.vertices[i * 8 + 3], self.vertices[i * 8 + 7])
        };
        [f32::from_bits(x), f32::from_bits(y), 1.0]
    }
}
const DEGENERATE: u8 = 1;
const ANY: u8 = 4;
const ORIENT: u8 = 8;
const NONE: usize = usize::MAX;
#[derive(Clone, Copy)]
struct Tri {
    neighbors: [usize; 3],
    groups: [usize; 3],
    os: V,
    ot: V,
    mag_s: f32,
    mag_t: f32,
    face: usize,
    flag: u8,
}
#[derive(Clone, Copy, Default)]
struct Space {
    os: V,
    ot: V,
    mag_s: f32,
    mag_t: f32,
    orient: bool,
}
struct Group {
    vertex: usize,
    orient: bool,
    start: usize,
    count: usize,
}

#[inline(never)]
fn grid_cell(min: f32, max: f32, value: f32) -> usize {
    let index = (2048.0 * ((value - min) / (max - min))) as i32;
    index.clamp(0, 2047) as usize
}
fn channel(min: V, max: V) -> usize {
    let d = sub(max, min);
    if d[1] > d[0] && d[1] > d[2] {
        1
    } else if d[2] > d[0] {
        2
    } else {
        0
    }
}
#[derive(Clone, Copy, Default)]
struct TmpVert {
    position: V,
    index: usize,
}
fn merge_verts(list: &mut [usize], tmp: &mut [TmpVert], mesh: &Mesh<'_>) {
    let mut min = tmp[0].position;
    let mut max = min;
    for v in &tmp[1..] {
        for c in 0..3 {
            if min[c] > v.position[c] {
                min[c] = v.position[c];
            }
            if max[c] < v.position[c] {
                max[c] = v.position[c];
            }
        }
    }
    let c = channel(min, max);
    let sep = 0.5 * (max[c] + min[c]);
    if !sep.is_finite() {
        return;
    }
    if sep >= max[c] || sep <= min[c] {
        for l in 0..tmp.len() {
            let i = tmp[l].index;
            let index = list[i];
            let p = mesh.position(index);
            let n = mesh.normal(index);
            let uv = mesh.tex_coord(index);
            for v in &tmp[..l] {
                let other = list[v.index];
                if p == mesh.position(other)
                    && n == mesh.normal(other)
                    && uv == mesh.tex_coord(other)
                {
                    list[i] = other;
                    break;
                }
            }
        }
    } else {
        let mut l = 0;
        let mut r = tmp.len() - 1;
        while l < r {
            while l < r && tmp[l].position[c] < sep {
                l += 1;
            }
            while l < r && !(tmp[r].position[c] < sep) {
                r -= 1;
            }
            if l < r {
                tmp.swap(l, r);
                l += 1;
                r -= 1;
            }
        }
        if l == r && tmp[r].position[c] < sep {
            l += 1;
        }
        let (left, right) = tmp.split_at_mut(l);
        if left.len() > 1 {
            merge_verts(list, left, mesh);
        }
        if right.len() > 1 {
            merge_verts(list, right, mesh);
        }
    }
}
fn shared_vertices(list: &mut [usize], mesh: &Mesh<'_>) {
    let mut min = mesh.position(0);
    let mut max = min;
    for &index in &list[1..] {
        let p = mesh.position(index);
        for c in 0..3 {
            if min[c] > p[c] {
                min[c] = p[c];
            } else if max[c] < p[c] {
                max[c] = p[c];
            }
        }
    }
    let c = channel(min, max);
    // Sparse cells avoid clearing/scanning 2048 entries for a handful of corners.
    // Emscripten's emmalloc likewise uses small-input paths to avoid fixed allocator work.
    // Stable cell order preserves the reference's MergeVertsFast input exactly.
    if list.len() <= 16 {
        let mut entries: Vec<_> = list
            .iter()
            .enumerate()
            .map(|(i, &index)| {
                let p = mesh.position(index);
                (
                    grid_cell(min[c], max[c], p[c]),
                    TmpVert {
                        position: p,
                        index: i,
                    },
                )
            })
            .collect();
        entries.sort_by_key(|e| e.0);
        let mut tmp: Vec<_> = entries.iter().map(|e| e.1).collect();
        let mut start = 0;
        while start < entries.len() {
            let mut end = start + 1;
            while end < entries.len() && entries[end].0 == entries[start].0 {
                end += 1;
            }
            if end - start > 1 {
                merge_verts(list, &mut tmp[start..end], mesh);
            }
            start = end;
        }
        return;
    }
    let mut counts = [0usize; 2048];
    for &index in list.iter() {
        counts[grid_cell(min[c], max[c], mesh.position(index)[c])] += 1;
    }
    let mut offsets = [0usize; 2048];
    for k in 1..2048 {
        offsets[k] = offsets[k - 1] + counts[k - 1];
    }
    let mut cursors = offsets;
    let mut table = vec![0; list.len()];
    for (i, &index) in list.iter().enumerate() {
        let k = grid_cell(min[c], max[c], mesh.position(index)[c]);
        table[cursors[k]] = i;
        cursors[k] += 1;
    }
    let mut tmp = vec![TmpVert::default(); *counts.iter().max().unwrap()];
    for k in 0..2048 {
        let count = counts[k];
        if count < 2 {
            continue;
        }
        for e in 0..count {
            let i = table[offsets[k] + e];
            tmp[e] = TmpVert {
                position: mesh.position(list[i]),
                index: i,
            };
        }
        merge_verts(list, &mut tmp[..count], mesh);
    }
}

// The last run is intentionally not sub-sorted: this is the reference's edge order,
// including its non-manifold tie breaking.
fn sort_edges(edges: &mut [[usize; 3]], c: usize, seed: u32) {
    if edges.len() < 2 {
        return;
    }
    if edges.len() == 2 {
        if edges[0][c] > edges[1][c] {
            edges.swap(0, 1);
        }
        return;
    }
    let seed = seed
        .wrapping_add(seed.rotate_left(seed & 31))
        .wrapping_add(3);
    let mid = edges[seed as usize % edges.len()][c];
    let mut l = 0isize;
    let mut r = edges.len() as isize - 1;
    loop {
        while edges[l as usize][c] < mid {
            l += 1;
        }
        while edges[r as usize][c] > mid {
            r -= 1;
        }
        if l <= r {
            edges.swap(l as usize, r as usize);
            l += 1;
            r -= 1;
        }
        if l > r {
            break;
        }
    }
    if r > 0 {
        sort_edges(&mut edges[..=r as usize], c, seed);
    }
    if (l as usize) < edges.len() - 1 {
        sort_edges(&mut edges[l as usize..], c, seed);
    }
}
fn edge(list: &[usize], face: usize, a: usize, b: usize) -> (usize, usize, usize) {
    let v = &list[face * 3..face * 3 + 3];
    let e = if v[0] == a || v[0] == b {
        if v[1] == a || v[1] == b {
            0
        } else {
            2
        }
    } else {
        1
    };
    (v[e], v[(e + 1) % 3], e)
}
fn neighbors(tris: &mut [Tri], list: &[usize]) {
    let mut edges = Vec::with_capacity(tris.len() * 3);
    for f in 0..tris.len() {
        for i in 0..3 {
            let a = list[f * 3 + i];
            let b = list[f * 3 + (i + 1) % 3];
            edges.push([a.min(b), a.max(b), f]);
        }
    }
    sort_edges(&mut edges, 0, 39871946);
    let mut start = 0;
    for i in 1..edges.len() {
        if edges[start][0] != edges[i][0] {
            sort_edges(&mut edges[start..i], 1, 39871946);
            start = i;
        }
    }
    start = 0;
    for i in 1..edges.len() {
        if edges[start][..2] != edges[i][..2] {
            sort_edges(&mut edges[start..i], 2, 39871946);
            start = i;
        }
    }
    for i in 0..edges.len() {
        let [a, b, f] = edges[i];
        let (a0, a1, ea) = edge(list, f, a, b);
        if tris[f].neighbors[ea] != NONE {
            continue;
        }
        for &[c, d, t] in &edges[i + 1..] {
            if a != c || b != d {
                break;
            }
            let (b1, b0, eb) = edge(list, t, c, d);
            if a0 == b0 && a1 == b1 && tris[t].neighbors[eb] == NONE {
                tris[f].neighbors[ea] = t;
                tris[t].neighbors[eb] = f;
                break;
            }
        }
    }
}
fn init_tri_info(tris: &mut [Tri], list: &[usize], mesh: &Mesh<'_>) {
    for (f, tri) in tris.iter_mut().enumerate() {
        let [v1, v2, v3] = std::array::from_fn(|i| mesh.position(list[f * 3 + i]));
        let [t1, t2, t3] = std::array::from_fn(|i| mesh.tex_coord(list[f * 3 + i]));
        let tx = t2[0] - t1[0];
        let ty = t2[1] - t1[1];
        let sx = t3[0] - t1[0];
        let sy = t3[1] - t1[1];
        let d1 = sub(v2, v1);
        let d2 = sub(v3, v1);
        let area = tx * sy - ty * sx;
        let os = sub(scale(sy, d1), scale(ty, d2));
        let ot = add(scale(-sx, d1), scale(tx, d2));
        tri.flag |= ANY;
        if area > 0.0 {
            tri.flag |= ORIENT;
        }
        if not_zero(area) {
            let ls = length(os);
            let lt = length(ot);
            let sign = if tri.flag & ORIENT == 0 { -1.0 } else { 1.0 };
            if not_zero(ls) {
                tri.os = scale(sign / ls, os);
            }
            if not_zero(lt) {
                tri.ot = scale(sign / lt, ot);
            }
            tri.mag_s = ls / area.abs();
            tri.mag_t = lt / area.abs();
            if not_zero(tri.mag_s) && not_zero(tri.mag_t) {
                tri.flag &= !ANY;
            }
        }
    }
    neighbors(tris, list);
}
fn assign_recur(
    list: &[usize],
    tris: &mut [Tri],
    f: usize,
    group: &mut Group,
    id: usize,
    members: &mut Vec<usize>,
) {
    let i = (0..3).find(|&i| list[f * 3 + i] == group.vertex).unwrap();
    if tris[f].groups[i] != NONE {
        return;
    }
    if tris[f].flag & ANY != 0 && tris[f].groups == [NONE; 3] {
        tris[f].flag = (tris[f].flag & !ORIENT) | if group.orient { ORIENT } else { 0 };
    }
    if (tris[f].flag & ORIENT != 0) != group.orient {
        return;
    }
    members.push(f);
    group.count += 1;
    tris[f].groups[i] = id;
    let l = tris[f].neighbors[i];
    let r = tris[f].neighbors[(i + 2) % 3];
    if l != NONE {
        assign_recur(list, tris, l, group, id, members);
    }
    if r != NONE {
        assign_recur(list, tris, r, group, id, members);
    }
}
fn build_groups(tris: &mut [Tri], list: &[usize]) -> (Vec<Group>, Vec<usize>) {
    let mut groups = Vec::new();
    let mut members = Vec::with_capacity(tris.len() * 3);
    for f in 0..tris.len() {
        for i in 0..3 {
            if tris[f].flag & ANY != 0 || tris[f].groups[i] != NONE {
                continue;
            }
            let id = groups.len();
            let mut group = Group {
                vertex: list[f * 3 + i],
                orient: tris[f].flag & ORIENT != 0,
                start: members.len(),
                count: 1,
            };
            tris[f].groups[i] = id;
            members.push(f);
            let l = tris[f].neighbors[i];
            let r = tris[f].neighbors[(i + 2) % 3];
            if l != NONE {
                assign_recur(list, tris, l, &mut group, id, &mut members);
            }
            if r != NONE {
                assign_recur(list, tris, r, &mut group, id, &mut members);
            }
            groups.push(group);
        }
    }
    (groups, members)
}
fn eval_tspace(
    faces: &[usize],
    list: &[usize],
    tris: &[Tri],
    mesh: &Mesh<'_>,
    vertex: usize,
) -> Space {
    let mut res = Space::default();
    let mut sum = 0.0;
    for &f in faces {
        let tri = tris[f];
        if tri.flag & ANY != 0 {
            continue;
        }
        let i = (0..3).find(|&i| list[f * 3 + i] == vertex).unwrap();
        let n = mesh.normal(list[f * 3 + i]);
        let os = project(tri.os, n);
        let ot = project(tri.ot, n);
        let p0 = mesh.position(list[f * 3 + (i + 2) % 3]);
        let p1 = mesh.position(list[f * 3 + i]);
        let p2 = mesh.position(list[f * 3 + (i + 1) % 3]);
        let v1 = project(sub(p0, p1), n);
        let v2 = project(sub(p2, p1), n);
        let cos = dot(v1, v2);
        let cos = if cos > 1.0 {
            1.0
        } else if cos < -1.0 {
            -1.0
        } else {
            cos
        };
        // C calls double acos, then rounds to float.
        let angle = (cos as f64).acos() as f32;
        res.os = add(res.os, scale(angle, os));
        res.ot = add(res.ot, scale(angle, ot));
        res.mag_s += angle * tri.mag_s;
        res.mag_t += angle * tri.mag_t;
        sum += angle;
    }
    if vnot_zero(res.os) {
        res.os = normalize(res.os);
    }
    if vnot_zero(res.ot) {
        res.ot = normalize(res.ot);
    }
    if sum > 0.0 {
        res.mag_s /= sum;
        res.mag_t /= sum;
    }
    res
}
fn generate_tspaces(
    spaces: &mut [Space],
    tris: &[Tri],
    list: &[usize],
    groups: &[Group],
    members: &[usize],
    mesh: &Mesh<'_>,
) {
    let mut subgroups: Vec<(Vec<usize>, Space)> = Vec::new();
    let mut tmp = Vec::new();
    let mut projections = Vec::new();
    for (id, g) in groups.iter().enumerate() {
        subgroups.clear();
        let faces = &members[g.start..g.start + g.count];
        // All members share a welded normal. Hoist the invariant projections out
        // of the quadratic membership loop (Binaryen's loop-invariant-code-motion
        // pass is the WebAssembly precedent); arithmetic and member order stay fixed.
        let n = mesh.normal(g.vertex);
        projections.clear();
        projections.extend(
            faces
                .iter()
                .map(|&f| (project(tris[f].os, n), project(tris[f].ot, n))),
        );
        for (a, &f) in faces.iter().enumerate() {
            let i = tris[f].groups.iter().position(|&v| v == id).unwrap();
            let (os, ot) = projections[a];
            tmp.clear();
            for (b, &t) in faces.iter().enumerate() {
                let (os2, ot2) = projections[b];
                if (tris[f].flag | tris[t].flag) & ANY != 0
                    || tris[f].face == tris[t].face
                    || (dot(os, os2) > -1.0 && dot(ot, ot2) > -1.0)
                {
                    tmp.push(t);
                }
            }
            // Unique integer members: sorting has no observable tie order.
            tmp.sort_unstable();
            let space = if let Some((_, s)) = subgroups.iter().find(|(m, _)| *m == tmp) {
                *s
            } else {
                let s = eval_tspace(&tmp, list, tris, mesh, g.vertex);
                subgroups.push((tmp.clone(), s));
                s
            };
            spaces[tris[f].face * 3 + i] = Space {
                orient: g.orient,
                ..space
            };
        }
    }
}

pub fn generate(mesh: &Mesh<'_>) -> Option<Vec<[f32; 4]>> {
    let count = mesh.indices.len() / 3;
    if count == 0 {
        return None;
    }
    let mut list: Vec<_> = (0..count)
        .flat_map(|f| (0..3).map(move |i| (f << 2) | i))
        .collect();
    let mut tris: Vec<_> = (0..count)
        .map(|f| Tri {
            neighbors: [NONE; 3],
            groups: [NONE; 3],
            os: [0.0; 3],
            ot: [0.0; 3],
            mag_s: 0.0,
            mag_t: 0.0,
            face: f,
            flag: 0,
        })
        .collect();
    shared_vertices(&mut list, mesh);
    let mut good = count;
    for t in 0..count {
        let p = [
            mesh.position(list[t * 3]),
            mesh.position(list[t * 3 + 1]),
            mesh.position(list[t * 3 + 2]),
        ];
        if p[0] == p[1] || p[0] == p[2] || p[1] == p[2] {
            tris[t].flag |= DEGENERATE;
            good -= 1;
        }
    }
    // DegenPrologue keeps the good triangles' order by swapping in the next good one.
    let mut next = 1;
    for t in 0..good {
        if tris[t].flag & DEGENERATE == 0 {
            next = next.max(t + 2);
        } else {
            while next < count && tris[next].flag & DEGENERATE != 0 {
                next += 1;
            }
            assert!(next < count);
            tris.swap(t, next);
            for i in 0..3 {
                list.swap(t * 3 + i, next * 3 + i);
            }
            next += 1;
        }
    }
    if good == 0 {
        return Some(vec![[1.0, 0.0, 0.0, 1.0]; count * 3]);
    }
    init_tri_info(&mut tris[..good], &list, mesh);
    let (groups, members) = build_groups(&mut tris[..good], &list);
    let mut spaces = vec![
        Space {
            os: [1.0, 0.0, 0.0],
            ot: [0.0, 1.0, 0.0],
            mag_s: 1.0,
            mag_t: 1.0,
            orient: false
        };
        count * 3
    ];
    generate_tspaces(&mut spaces, &tris, &list, &groups, &members, mesh);
    for t in good..count {
        for i in 0..3 {
            if let Some(j) = list[..good * 3].iter().position(|&v| v == list[t * 3 + i]) {
                spaces[tris[t].face * 3 + i] = spaces[tris[j / 3].face * 3 + j % 3];
            }
        }
    }
    Some(
        spaces
            .into_iter()
            .map(|s| [s.os[0], s.os[1], s.os[2], if s.orient { -1.0 } else { 1.0 }])
            .collect(),
    )
}
