//! Box3D's adjacent-sibling dynamic tree. Proxy identity is independent of node placement.
use crate::math::Vec3;
use crate::mesh_query::{aabb_overlaps, bounds_ray_overlap, max, min};

pub const STACK_SIZE: usize = 512;
pub const MOVED: u32 = 1 << 30;
pub const LEAF: u32 = 1 << 31;
pub const INDEX: u32 = MOVED - 1;
pub const EMPTY: u32 = INDEX | LEAF;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Node {
    pub lo: Vec3,
    pub hi: Vec3,
    pub flags: u32,
    pub height: i32,
}
impl Node {
    pub fn empty() -> Self {
        Self {
            lo: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            hi: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            flags: EMPTY,
            height: 0,
        }
    }
    pub fn leaf(self) -> bool {
        self.flags & LEAF != 0
    }
    pub fn index(self) -> usize {
        (self.flags & INDEX) as usize
    }
    pub fn height(self) -> i32 {
        if self.leaf() {
            0
        } else {
            self.height
        }
    }
    fn center(self) -> Vec3 {
        self.lo.add(self.hi).scale(0.5)
    }
    pub fn area(self) -> f32 {
        let d = self.hi.sub(self.lo);
        2.0 * ((d.x * d.z + d.y * d.x) + d.z * d.y)
    }
    fn union(self, other: Self) -> Self {
        Self {
            lo: min(self.lo, other.lo),
            hi: max(self.hi, other.hi),
            flags: 0,
            height: 0,
        }
    }
    fn distance(self, p: Vec3) -> f32 {
        let d = p.sub(max(self.lo, min(p, self.hi)));
        d.dot(d)
    }
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Proxy {
    pub user: u64,
    pub category: u64,
    pub node: i32,
    pub next: i32,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub parents: Vec<i32>,
    pub proxies: Vec<Proxy>,
    pub end: usize,
    pub pair_free: i32,
    pub proxy_free: i32,
    pub count: usize,
    pub ordered: bool,
    swap: Vec<Node>,
    leaves: Vec<Node>,
    indices: Vec<usize>,
    centers: Vec<Vec3>,
}
impl Tree {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(16);
        Self {
            nodes: vec![Node::empty(); 2 * cap],
            parents: vec![-1; 2 * cap],
            proxies: (0..cap)
                .map(|i| Proxy {
                    user: 0,
                    category: 0,
                    node: -1,
                    next: if i + 1 < cap { (i + 1) as i32 } else { -1 },
                })
                .collect(),
            end: 2,
            pair_free: -1,
            proxy_free: 0,
            count: 0,
            ordered: true,
            swap: Vec::new(),
            leaves: Vec::new(),
            indices: Vec::new(),
            centers: Vec::new(),
        }
    }
    fn pair(&mut self) -> usize {
        if self.pair_free != -1 {
            let p = self.pair_free as usize;
            self.pair_free = self.parents[p];
            return p;
        }
        if self.end + 2 > self.nodes.len() {
            let mut cap = self.nodes.len() + self.nodes.len() / 2;
            cap += cap & 1;
            self.nodes.resize(cap, Node::empty());
            self.parents.resize(cap, -1);
            self.swap.clear();
        }
        let p = self.end;
        self.end += 2;
        p
    }
    fn free_pair(&mut self, p: usize) {
        self.nodes[p] = Node::empty();
        self.nodes[p + 1] = Node::empty();
        self.parents[p] = self.pair_free;
        self.parents[p + 1] = -1;
        self.pair_free = p as i32;
    }
    fn internal(nodes: &[Node], p: usize) -> Node {
        let a = nodes[p];
        let b = nodes[p + 1];
        let mut n = a.union(b);
        n.flags = p as u32 | ((a.flags | b.flags) & MOVED);
        n.height = 1 + a.height().max(b.height());
        n
    }
    fn link(&mut self, i: usize) {
        let n = self.nodes[i];
        if n.leaf() {
            self.proxies[n.index()].node = i as i32;
        } else {
            self.parents[n.index()] = i as i32;
            self.parents[n.index() + 1] = i as i32;
        }
    }
    fn node_ordered(&self, i: usize) -> bool {
        let n = self.nodes[i];
        n.leaf() || n.index() > i
    }
    fn sibling(&self, box_d: Node) -> usize {
        let area_d = box_d.area();
        let center_d = box_d.center();
        let mut index = 0;
        let mut area_base = self.nodes[0].area();
        let mut direct = self.nodes[0].union(box_d).area();
        let mut inherited = 0.0;
        let mut best = 0;
        let mut best_cost = direct;
        while !self.nodes[index].leaf() {
            let pair = self.nodes[index].index();
            let children = [self.nodes[pair], self.nodes[pair + 1]];
            let cost = direct + inherited;
            if cost < best_cost {
                best = index;
                best_cost = cost;
            }
            inherited += direct - area_base;
            let mut lower = [f32::MAX; 2];
            let mut areas = [0.0; 2];
            let mut costs = [0.0; 2];
            for k in 0..2 {
                costs[k] = children[k].union(box_d).area();
                if children[k].leaf() {
                    let cost = costs[k] + inherited;
                    if cost < best_cost {
                        best = pair + k;
                        best_cost = cost;
                    }
                } else {
                    areas[k] = children[k].area();
                    let delta = area_d - areas[k];
                    lower[k] = (inherited + costs[k]) + if delta < 0.0 { delta } else { 0.0 };
                }
            }
            if children[0].leaf() && children[1].leaf()
                || best_cost <= lower[0] && best_cost <= lower[1]
            {
                break;
            }
            if lower[0] == lower[1] && !children[0].leaf() {
                for k in 0..2 {
                    let d = children[k].center().sub(center_d);
                    lower[k] = d.dot(d);
                }
            }
            let k = if lower[0] < lower[1] && !children[0].leaf() {
                0
            } else {
                1
            };
            index = pair + k;
            area_base = areas[k];
            direct = costs[k];
        }
        best
    }
    fn rotate(&mut self, i: usize) {
        let b = self.nodes[i].index();
        let c = b + 1;
        let mut down = usize::MAX;
        let mut up = 0;
        let mut best = 0.0;
        for (d, s) in [(b, c), (c, b)] {
            if self.nodes[s].leaf() {
                continue;
            }
            let p = self.nodes[s].index();
            let area = self.nodes[s].area();
            for (u, other) in [(p, p + 1), (p + 1, p)] {
                let delta = self.nodes[d].union(self.nodes[other]).area() - area;
                if delta < best {
                    down = d;
                    up = u;
                    best = delta;
                }
            }
        }
        if down != usize::MAX {
            self.nodes.swap(down, up);
            self.link(down);
            self.link(up);
            if !self.node_ordered(down) || !self.node_ordered(up) {
                self.ordered = false;
            }
            let sibling = down ^ 1;
            self.nodes[sibling] = Self::internal(&self.nodes, self.nodes[sibling].index());
        }
    }
    fn insert(&mut self, id: usize, lo: Vec3, hi: Vec3, moved: bool, rotate: bool) {
        let leaf = Node {
            lo,
            hi,
            flags: id as u32 | LEAF | if moved { MOVED } else { 0 },
            height: self.proxies[id].user as u32 as i32,
        };
        if self.nodes[0].flags == EMPTY {
            self.nodes[0] = leaf;
            self.proxies[id].node = 0;
            return;
        }
        let sibling = self.sibling(leaf);
        let p = self.pair();
        self.nodes[p] = self.nodes[sibling];
        self.nodes[p + 1] = leaf;
        self.parents[p] = sibling as i32;
        self.parents[p + 1] = sibling as i32;
        self.link(p);
        self.proxies[id].node = (p + 1) as i32;
        self.nodes[sibling] = Self::internal(&self.nodes, p);
        if !self.node_ordered(sibling) || !self.node_ordered(p) {
            self.ordered = false;
        }
        let mut index = sibling as i32;
        while index != -1 {
            let i = index as usize;
            if rotate {
                self.rotate(i);
            }
            self.nodes[i] = Self::internal(&self.nodes, self.nodes[i].index());
            index = self.parents[i];
        }
    }
    pub fn create(&mut self, lo: Vec3, hi: Vec3, category: u64, user: u64, moved: bool) -> usize {
        if self.proxy_free == -1 {
            let old = self.proxies.len();
            let cap = old + (old / 2).max(1);
            self.proxies.extend((old..cap).map(|i| Proxy {
                user: 0,
                category: 0,
                node: -1,
                next: if i + 1 < cap { (i + 1) as i32 } else { -1 },
            }));
            self.proxy_free = old as i32;
        }
        let id = self.proxy_free as usize;
        self.proxy_free = self.proxies[id].next;
        self.proxies[id] = Proxy {
            user,
            category,
            node: -1,
            next: -1,
        };
        self.count += 1;
        self.insert(id, lo, hi, moved, true);
        id
    }
    fn remove(&mut self, id: usize) {
        let leaf = self.proxies[id].node as usize;
        if leaf == 0 {
            self.nodes[0] = Node::empty();
            return;
        }
        let parent = self.parents[leaf] as usize;
        self.nodes[parent] = self.nodes[leaf ^ 1];
        self.link(parent);
        self.free_pair(leaf & !1);
        let mut i = self.parents[parent];
        while i != -1 {
            let n = i as usize;
            self.nodes[n] = Self::internal(&self.nodes, self.nodes[n].index());
            i = self.parents[n];
        }
    }
    pub fn destroy(&mut self, id: usize) {
        self.remove(id);
        self.proxies[id].node = -1;
        self.proxies[id].next = self.proxy_free;
        self.proxy_free = id as i32;
        self.count -= 1;
    }
    pub fn move_proxy(&mut self, id: usize, lo: Vec3, hi: Vec3, moved: bool) {
        self.remove(id);
        self.insert(id, lo, hi, moved, false);
    }
    pub fn mark_moved(&mut self, id: usize) {
        let mut i = self.proxies[id].node;
        while i != -1 {
            let n = &mut self.nodes[i as usize];
            if n.flags & MOVED != 0 {
                break;
            }
            n.flags |= MOVED;
            i = self.parents[i as usize];
        }
    }
    pub fn enlarge(&mut self, id: usize, lo: Vec3, hi: Vec3) {
        let i = self.proxies[id].node as usize;
        self.nodes[i].lo = lo;
        self.nodes[i].hi = hi;
        self.nodes[i].flags |= MOVED;
        let mut p = self.parents[i];
        while p != -1 {
            let n = p as usize;
            self.nodes[n].lo = min(self.nodes[n].lo, lo);
            self.nodes[n].hi = max(self.nodes[n].hi, hi);
            self.nodes[n].flags |= MOVED;
            p = self.parents[n];
        }
    }
    pub fn clear_moved(&mut self) {
        if self.nodes[0].flags & MOVED == 0 {
            return;
        }
        self.nodes[0].flags &= !MOVED;
        if self.nodes[0].leaf() {
            return;
        }
        let mut stack = [0; STACK_SIZE];
        stack[0] = self.nodes[0].index();
        let mut count = 1;
        while count > 0 {
            count -= 1;
            let p = stack[count];
            for i in p..p + 2 {
                let n = &mut self.nodes[i];
                if n.flags & MOVED != 0 {
                    n.flags &= !MOVED;
                    if !n.leaf() {
                        assert!(count < STACK_SIZE);
                        stack[count] = n.index();
                        count += 1;
                    }
                }
            }
        }
    }
    pub fn refit(&mut self) {
        if self.nodes[0].flags & MOVED == 0 {
            return;
        }
        if self.ordered {
            for i in (0..self.end).rev() {
                let n = self.nodes[i];
                if n.flags & (LEAF | MOVED) == MOVED {
                    let union = self.nodes[n.index()].union(self.nodes[n.index() + 1]);
                    self.nodes[i].lo = union.lo;
                    self.nodes[i].hi = union.hi;
                }
            }
        } else {
            let mut stack = [0i32; STACK_SIZE];
            stack[0] = 0;
            let mut count = 1;
            while count > 0 {
                count -= 1;
                let item = stack[count];
                if item < 0 {
                    let i = (!item) as usize;
                    let p = self.nodes[i].index();
                    let u = self.nodes[p].union(self.nodes[p + 1]);
                    self.nodes[i].lo = u.lo;
                    self.nodes[i].hi = u.hi;
                    continue;
                }
                let n = self.nodes[item as usize];
                if n.leaf() || n.flags & MOVED == 0 {
                    continue;
                }
                assert!(count + 3 <= STACK_SIZE);
                stack[count] = !item;
                stack[count + 1] = n.index() as i32;
                stack[count + 2] = (n.index() + 1) as i32;
                count += 3;
            }
        }
    }
    pub fn view(&self) -> TreeQuery<'_> {
        TreeQuery {
            nodes: &self.nodes,
            proxies: &self.proxies,
            count: self.count,
            end: self.end,
        }
    }
    pub fn area_ratio(&self) -> f32 {
        self.view().area_ratio()
    }
    pub fn query<F: FnMut(usize, u64) -> bool>(
        &self,
        lo: Vec3,
        hi: Vec3,
        mask: u64,
        all: bool,
        cb: F,
    ) -> (u32, u32) {
        self.view().query(lo, hi, mask, all, cb)
    }
    pub fn closest<F: FnMut(f32, usize, u64) -> f32>(
        &self,
        point: Vec3,
        mask: u64,
        all: bool,
        limit: &mut f32,
        cb: F,
    ) -> (u32, u32) {
        self.view().closest(point, mask, all, limit, cb)
    }
    pub fn cast<F: FnMut(f32, usize, u64) -> f32>(
        &self,
        lo: Vec3,
        hi: Vec3,
        translation: Vec3,
        fraction: f32,
        mask: u64,
        all: bool,
        box_cast: bool,
        cb: F,
    ) -> (u32, u32) {
        self.view()
            .cast(lo, hi, translation, fraction, mask, all, box_cast, cb)
    }
}
pub struct TreeQuery<'a> {
    pub nodes: &'a [Node],
    pub proxies: &'a [Proxy],
    pub count: usize,
    pub end: usize,
}
impl TreeQuery<'_> {
    fn root_pair(&self) -> usize {
        if self.nodes[0].leaf() {
            0
        } else {
            self.nodes[0].index()
        }
    }
    fn matches(&self, n: Node, mask: u64, all: bool) -> bool {
        let bits = self.proxies[n.index()].category & mask;
        if all {
            bits == mask
        } else {
            bits != 0
        }
    }
    pub fn area_ratio(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for n in &self.nodes[1..self.end] {
            if !n.leaf() {
                sum += n.area();
            }
        }
        sum / self.nodes[0].area()
    }
    pub fn query<F: FnMut(usize, u64) -> bool>(
        &self,
        lo: Vec3,
        hi: Vec3,
        mask: u64,
        all: bool,
        mut cb: F,
    ) -> (u32, u32) {
        if self.count == 0 {
            return (0, 0);
        }
        let mut stack = [0; STACK_SIZE];
        stack[0] = self.root_pair();
        let mut count = 1;
        let mut stats = (0, 0);
        while count > 0 {
            count -= 1;
            let p = stack[count];
            stats.0 += 1;
            for i in p..p + 2 {
                let n = self.nodes[i];
                if !aabb_overlaps(n.lo, n.hi, lo, hi) {
                    continue;
                }
                if n.leaf() {
                    if self.matches(n, mask, all) {
                        stats.1 += 1;
                        if !cb(n.index(), self.proxies[n.index()].user) {
                            return stats;
                        }
                    }
                } else if count < STACK_SIZE - 1 {
                    stack[count] = n.index();
                    count += 1;
                }
            }
        }
        stats
    }
    pub fn closest<F: FnMut(f32, usize, u64) -> f32>(
        &self,
        point: Vec3,
        mask: u64,
        all: bool,
        limit: &mut f32,
        mut cb: F,
    ) -> (u32, u32) {
        if self.count == 0 {
            return (0, 0);
        }
        let mut stack = [(0, 0.0); STACK_SIZE];
        let p = self.root_pair();
        let mut count = 0;
        let push = |stack: &mut [(usize, f32)], count: &mut usize, p: usize| {
            let a = (p, self.nodes[p].distance(point));
            let b = (p + 1, self.nodes[p + 1].distance(point));
            if b.1 < a.1 {
                stack[*count] = a;
                stack[*count + 1] = b;
            } else {
                stack[*count] = b;
                stack[*count + 1] = a;
            }
            *count += 2;
        };
        if p == 0 {
            stack[0] = (0, self.nodes[0].distance(point));
            count = 1;
        } else {
            push(&mut stack, &mut count, p);
        }
        let mut stats = (0, 0);
        while count > 0 {
            count -= 1;
            let (i, d) = stack[count];
            let n = self.nodes[i];
            stats.0 += 1;
            if d >= *limit {
                continue;
            }
            if n.leaf() {
                if self.matches(n, mask, all) {
                    let dd = cb(*limit, n.index(), self.proxies[n.index()].user);
                    if dd < *limit {
                        *limit = dd;
                    }
                    stats.1 += 1;
                }
            } else if count + 2 <= STACK_SIZE {
                push(&mut stack, &mut count, n.index());
            }
        }
        stats
    }
    pub fn cast<F: FnMut(f32, usize, u64) -> f32>(
        &self,
        lo: Vec3,
        hi: Vec3,
        translation: Vec3,
        mut fraction: f32,
        mask: u64,
        all: bool,
        box_cast: bool,
        mut cb: F,
    ) -> (u32, u32) {
        if self.count == 0 {
            return (0, 0);
        }
        let origin = if box_cast { lo.add(hi).scale(0.5) } else { lo };
        let extent = if box_cast {
            hi.sub(lo).scale(0.5)
        } else {
            Vec3::ZERO
        };
        let mut total_lo = min(lo, lo.mul_add(fraction, translation));
        let mut total_hi = max(hi, hi.mul_add(fraction, translation));
        let mut stack = [0; STACK_SIZE];
        stack[0] = self.root_pair();
        let mut count = 1;
        let mut stats = (0, 0);
        while count > 0 {
            count -= 1;
            let p = stack[count];
            stats.0 += 1;
            let mut hit = [Node::empty(); 2];
            let mut hits = 0;
            for i in p..p + 2 {
                let n = self.nodes[i];
                if aabb_overlaps(n.lo, n.hi, total_lo, total_hi)
                    && bounds_ray_overlap(n.lo.sub(extent), n.hi.add(extent), origin, translation)
                {
                    hit[hits] = n;
                    hits += 1;
                }
            }
            if hits == 2 && !hit[0].leaf() && !hit[1].leaf() {
                let d1 = hit[0].center().sub(origin);
                let d2 = hit[1].center().sub(origin);
                if d1.dot(d1) < d2.dot(d2) {
                    hit.swap(0, 1);
                }
            }
            for n in &hit[..hits] {
                if n.leaf() {
                    if !self.matches(*n, mask, all) {
                        continue;
                    }
                    let value = cb(fraction, n.index(), self.proxies[n.index()].user);
                    stats.1 += 1;
                    if value == 0.0 {
                        return stats;
                    }
                    if value > 0.0
                        && if box_cast {
                            value < fraction
                        } else {
                            value <= fraction
                        }
                    {
                        fraction = value;
                        total_lo = min(lo, lo.mul_add(fraction, translation));
                        total_hi = max(hi, hi.mul_add(fraction, translation));
                    }
                } else if count < STACK_SIZE - 1 {
                    stack[count] = n.index();
                    count += 1;
                }
            }
        }
        stats
    }
}
impl Tree {
    fn partition(indices: &mut [usize], centers: &mut [Vec3]) -> usize {
        let count = indices.len();
        if count <= 2 {
            return count / 2;
        }
        let mut lo = centers[0];
        let mut hi = lo;
        for &c in &centers[1..] {
            lo = min(lo, c);
            hi = max(hi, c);
        }
        let d = hi.sub(lo);
        let c = lo.add(hi).scale(0.5);
        let axis = if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        };
        let component = |v: Vec3| [v.x, v.y, v.z][axis];
        let pivot = component(c);
        let mut a = 0;
        let mut b = count;
        while a < b {
            while a < b && component(centers[a]) < pivot {
                a += 1;
            }
            while a < b && component(centers[b - 1]) >= pivot {
                b -= 1;
            }
            if a < b {
                indices.swap(a, b - 1);
                centers.swap(a, b - 1);
                a += 1;
                b -= 1;
            }
        }
        if a == 0 || a == count {
            count / 2
        } else {
            a
        }
    }
    fn bump_pair(&mut self, parent: usize) -> usize {
        let p = self.end;
        self.end += 2;
        self.parents[p] = parent as i32;
        self.parents[p + 1] = parent as i32;
        p
    }
    fn copy_subtree(&mut self, mut node: Node, index: usize) {
        if node.leaf() {
            self.swap[index] = node;
            self.proxies[node.index()].node = index as i32;
            return;
        }
        let mut stack = [(0, 0); STACK_SIZE];
        let mut count = 0;
        let mut old = node.index();
        let mut new = self.bump_pair(index);
        node.flags = (node.flags & !INDEX) | new as u32;
        self.swap[index] = node;
        loop {
            self.swap[new] = self.nodes[old];
            self.swap[new + 1] = self.nodes[old + 1];
            let right = self.swap[new + 1];
            if right.leaf() {
                self.proxies[right.index()].node = (new + 1) as i32;
            } else {
                assert!(count < STACK_SIZE);
                stack[count] = (right.index(), new + 1);
                count += 1;
            }
            let left = self.swap[new];
            if !left.leaf() {
                let index = new;
                old = left.index();
                new = self.bump_pair(index);
                self.swap[index].flags = (self.swap[index].flags & !INDEX) | new as u32;
                continue;
            }
            self.proxies[left.index()].node = new as i32;
            if count == 0 {
                break;
            }
            count -= 1;
            let item = stack[count];
            old = item.0;
            new = self.bump_pair(item.1);
            self.swap[item.1].flags = (self.swap[item.1].flags & !INDEX) | new as u32;
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn snapshot(&self, out: &mut Vec<u8>) {
        for value in [
            self.nodes.len(),
            self.proxies.len(),
            self.end,
            self.pair_free as u32 as usize,
            self.proxy_free as u32 as usize,
            self.count,
            usize::from(self.ordered),
        ] {
            crate::regions::write_word(out, value);
        }
        for n in &self.nodes {
            for value in [
                n.lo.x.to_bits(),
                n.lo.y.to_bits(),
                n.lo.z.to_bits(),
                n.hi.x.to_bits(),
                n.hi.y.to_bits(),
                n.hi.z.to_bits(),
                n.flags,
                n.height as u32,
            ] {
                crate::regions::write_word(out, value as usize);
            }
        }
        for &p in &self.parents {
            crate::regions::write_word(out, p as u32 as usize);
        }
        for p in &self.proxies {
            for value in [
                p.user as u32,
                (p.user >> 32) as u32,
                p.category as u32,
                (p.category >> 32) as u32,
                p.node as u32,
                p.next as u32,
            ] {
                crate::regions::write_word(out, value as usize);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn restore(input: &mut &[u8]) -> Self {
        let cap = crate::regions::read_word(input);
        let pc = crate::regions::read_word(input);
        let mut t = Self::new(16);
        t.end = crate::regions::read_word(input);
        t.pair_free = crate::regions::read_word(input) as i32;
        t.proxy_free = crate::regions::read_word(input) as i32;
        t.count = crate::regions::read_word(input);
        t.ordered = crate::regions::read_word(input) != 0;
        t.nodes.resize(cap, Node::empty());
        t.parents.resize(cap, -1);
        t.proxies.resize(
            pc,
            Proxy {
                user: 0,
                category: 0,
                node: -1,
                next: -1,
            },
        );
        for n in &mut t.nodes {
            let mut v = [0; 8];
            for x in &mut v {
                *x = crate::regions::read_word(input) as u32;
            }
            *n = Node {
                lo: Vec3::new(
                    f32::from_bits(v[0]),
                    f32::from_bits(v[1]),
                    f32::from_bits(v[2]),
                ),
                hi: Vec3::new(
                    f32::from_bits(v[3]),
                    f32::from_bits(v[4]),
                    f32::from_bits(v[5]),
                ),
                flags: v[6],
                height: v[7] as i32,
            };
        }
        for p in &mut t.parents {
            *p = crate::regions::read_word(input) as i32;
        }
        for p in &mut t.proxies {
            let mut v = [0; 6];
            for x in &mut v {
                *x = crate::regions::read_word(input) as u32;
            }
            *p = Proxy {
                user: v[0] as u64 | ((v[1] as u64) << 32),
                category: v[2] as u64 | ((v[3] as u64) << 32),
                node: v[4] as i32,
                next: v[5] as i32,
            };
        }
        t
    }
    pub fn valid(&self) -> bool {
        if self.end < 2
            || self.end > self.nodes.len()
            || self.end & 1 != 0
            || self.parents.len() != self.nodes.len()
            || self.parents[0] != -1
            || self.nodes[1].flags != EMPTY
        {
            return false;
        }
        let mut seen = vec![false; self.end];
        let mut proxies = vec![false; self.proxies.len()];
        let mut stack = vec![(0, -1)];
        let mut leaves = 0;
        while let Some((i, parent)) = stack.pop() {
            if i >= self.end || seen[i] || self.parents[i] != parent {
                return false;
            }
            seen[i] = true;
            let n = self.nodes[i];
            if n.flags == EMPTY {
                if i != 0 || self.count != 0 {
                    return false;
                }
                continue;
            }
            for (lo, hi) in [(n.lo.x, n.hi.x), (n.lo.y, n.hi.y), (n.lo.z, n.hi.z)] {
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return false;
                }
            }
            if n.leaf() {
                let p = n.index();
                if p >= proxies.len() || proxies[p] || self.proxies[p].node != i as i32 {
                    return false;
                }
                proxies[p] = true;
                leaves += 1;
            } else {
                let p = n.index();
                if p < 2 || p & 1 != 0 || p + 1 >= self.end {
                    return false;
                }
                if n.height != 1 + self.nodes[p].height().max(self.nodes[p + 1].height()) {
                    return false;
                }
                for c in p..p + 2 {
                    let child = self.nodes[c];
                    if child.flags == EMPTY
                        || n.lo.x > child.lo.x
                        || n.lo.y > child.lo.y
                        || n.lo.z > child.lo.z
                        || n.hi.x < child.hi.x
                        || n.hi.y < child.hi.y
                        || n.hi.z < child.hi.z
                    {
                        return false;
                    }
                    stack.push((c, i as i32));
                }
            }
        }
        if leaves != self.count {
            return false;
        }
        seen[1] = true;
        let mut free = self.pair_free;
        while free != -1 {
            if free < 2 || free & 1 != 0 || free as usize + 1 >= self.end {
                return false;
            }
            let p = free as usize;
            if seen[p]
                || seen[p + 1]
                || self.nodes[p].flags != EMPTY
                || self.nodes[p + 1].flags != EMPTY
            {
                return false;
            }
            seen[p] = true;
            seen[p + 1] = true;
            free = self.parents[p];
        }
        if seen.iter().any(|s| !*s) {
            return false;
        }
        let mut free = self.proxy_free;
        while free != -1 {
            if free < 0 || free as usize >= proxies.len() {
                return false;
            }
            let p = free as usize;
            if proxies[p] || self.proxies[p].node != -1 {
                return false;
            }
            proxies[p] = true;
            free = self.proxies[p].next;
        }
        proxies.iter().all(|s| *s)
    }
    // Reserve before scheduling a world rebuild: workers may not grow shared linear memory.
    pub fn reserve_rebuild(&mut self) {
        self.swap.resize(self.nodes.len(), Node::empty());
        self.leaves
            .reserve(self.count.saturating_sub(self.leaves.len()));
        self.indices
            .reserve(self.count.saturating_sub(self.indices.len()));
        self.centers
            .reserve(self.count.saturating_sub(self.centers.len()));
    }
    pub fn rebuild(&mut self, full: bool) -> usize {
        if self.count == 0 || !full && self.nodes[0].flags & MOVED == 0 && self.ordered {
            return 0;
        }
        self.reserve_rebuild();
        self.leaves.clear();
        self.indices.clear();
        self.centers.clear();
        let mut stack = [0; STACK_SIZE];
        let mut count = 0;
        let root = self.nodes[0];
        if !root.leaf() && (full || root.flags & MOVED != 0) {
            stack[0] = root.index();
            count = 1;
        } else {
            let mut n = root;
            if !full {
                n.flags &= !MOVED;
            }
            self.leaves.push(n);
            self.indices.push(0);
            self.centers.push(n.center());
        }
        while count > 0 {
            count -= 1;
            let p = stack[count];
            for i in p..p + 2 {
                let mut n = self.nodes[i];
                if !n.leaf() && (full || n.flags & MOVED != 0) {
                    assert!(count < STACK_SIZE);
                    stack[count] = n.index();
                    count += 1;
                } else {
                    if !full {
                        n.flags &= !MOVED;
                    }
                    self.indices.push(self.leaves.len());
                    self.leaves.push(n);
                    self.centers.push(n.center());
                }
            }
        }
        let leaves = self.leaves.len();
        self.end = 2;
        self.swap[1] = Node::empty();
        self.parents[0] = -1;
        self.parents[1] = -1;
        if leaves == 1 {
            self.copy_subtree(self.leaves[self.indices[0]], 0);
        } else {
            // node, pair, next child, start, split, end
            let mut build = [[0; 6]; STACK_SIZE];
            let p = self.bump_pair(0);
            let split = Self::partition(&mut self.indices, &mut self.centers);
            build[0] = [0, p, 0, 0, split, leaves];
            let mut top = 0;
            loop {
                let item = build[top];
                if item[2] == 2 {
                    self.swap[item[0]] = Self::internal(&self.swap, item[1]);
                    if top == 0 {
                        break;
                    }
                    top -= 1;
                    continue;
                }
                let slot = item[2];
                build[top][2] += 1;
                let start = if slot == 0 { item[3] } else { item[4] };
                let end = if slot == 0 { item[4] } else { item[5] };
                let index = item[1] + slot;
                if end - start == 1 {
                    self.copy_subtree(self.leaves[self.indices[start]], index);
                } else {
                    assert!(top + 1 < STACK_SIZE);
                    top += 1;
                    let p = self.bump_pair(index);
                    let split = start
                        + Self::partition(
                            &mut self.indices[start..end],
                            &mut self.centers[start..end],
                        );
                    build[top] = [index, p, 0, start, split, end];
                }
            }
        }
        std::mem::swap(&mut self.nodes, &mut self.swap);
        self.pair_free = -1;
        self.ordered = true;
        leaves
    }
}
