// Port of Draco 55462882: mesh_edgebreaker_decoder_impl.cc,
// mesh_edgebreaker_traversal{,_valence}_decoder.h and mesh_attribute_corner_table.cc.
use crate::{
    allocation::{filled, push},
    buffer::{Bits, Reader},
    entropy::{self, Binary},
    Result,
};
use std::collections::HashMap;

pub(crate) const INVALID: usize = usize::MAX;
const C: u32 = 0;
const S: u32 = 1;
const L: u32 = 3;
const R: u32 = 5;
const E: u32 = 7;
const TOPOLOGY_ERROR: &str = "invalid Draco Edgebreaker topology";
const ALLOCATION_ERROR: &str = "Draco Edgebreaker allocation failed";

pub(crate) fn next(c: usize) -> usize {
    if c == INVALID {
        INVALID
    } else if c % 3 == 2 {
        c - 2
    } else {
        c + 1
    }
}

pub(crate) fn previous(c: usize) -> usize {
    if c == INVALID {
        INVALID
    } else if c.is_multiple_of(3) {
        c + 2
    } else {
        c - 1
    }
}

pub(crate) struct CornerTable {
    pub vertices: Vec<usize>,
    pub opposites: Vec<usize>,
    pub leftmost: Vec<usize>,
    holes: Vec<bool>,
    max_faces: usize,
    max_vertices: usize,
}

impl CornerTable {
    fn new(max_faces: usize, max_vertices: usize) -> Self {
        Self {
            vertices: Vec::new(),
            opposites: Vec::new(),
            leftmost: Vec::new(),
            holes: Vec::new(),
            max_faces,
            max_vertices,
        }
    }

    // Grow only when a face/vertex is decoded, not from untrusted header counts.
    fn add_face(&mut self) -> Result<usize> {
        let corner = self.vertices.len();
        if corner / 3 >= self.max_faces {
            return Err("too many Draco Edgebreaker faces");
        }
        self.vertices.try_reserve(3).map_err(|_| ALLOCATION_ERROR)?;
        self.opposites
            .try_reserve(3)
            .map_err(|_| ALLOCATION_ERROR)?;
        self.vertices.extend_from_slice(&[INVALID; 3]);
        self.opposites.extend_from_slice(&[INVALID; 3]);
        Ok(corner)
    }

    fn add_vertex(&mut self) -> Result<usize> {
        let v = self.leftmost.len();
        if v >= self.max_vertices {
            return Err("too many Draco Edgebreaker vertices");
        }
        push(&mut self.leftmost, INVALID)?;
        push(&mut self.holes, true)?;
        Ok(v)
    }

    fn vertex(&self, c: usize) -> Result<usize> {
        self.vertices
            .get(c)
            .copied()
            .filter(|&v| v < self.leftmost.len())
            .ok_or(TOPOLOGY_ERROR)
    }

    pub fn opposite(&self, c: usize) -> usize {
        self.opposites.get(c).copied().unwrap_or(INVALID)
    }

    fn leftmost(&self, v: usize) -> Result<usize> {
        self.leftmost.get(v).copied().ok_or(TOPOLOGY_ERROR)
    }

    fn set_leftmost(&mut self, v: usize, c: usize) -> Result<()> {
        if c != INVALID && c >= self.vertices.len() {
            return Err(TOPOLOGY_ERROR);
        }
        *self.leftmost.get_mut(v).ok_or(TOPOLOGY_ERROR)? = c;
        Ok(())
    }

    // MapCornerToVertex does not change the leftmost corner in the library.
    fn map(&mut self, c: usize, v: usize) -> Result<()> {
        if v >= self.leftmost.len() {
            return Err(TOPOLOGY_ERROR);
        }
        *self.vertices.get_mut(c).ok_or(TOPOLOGY_ERROR)? = v;
        Ok(())
    }

    fn link(&mut self, a: usize, b: usize) -> Result<()> {
        if a == b || self.opposite(a) != INVALID || self.opposite(b) != INVALID {
            return Err(TOPOLOGY_ERROR);
        }
        *self.opposites.get_mut(a).ok_or(TOPOLOGY_ERROR)? = b;
        *self.opposites.get_mut(b).ok_or(TOPOLOGY_ERROR)? = a;
        Ok(())
    }

    fn interior(&mut self, v: usize) -> Result<()> {
        *self.holes.get_mut(v).ok_or(TOPOLOGY_ERROR)? = false;
        Ok(())
    }

    pub fn swing_left(&self, c: usize) -> usize {
        next(self.opposite(next(c)))
    }

    pub fn swing_right(&self, c: usize) -> usize {
        previous(self.opposite(previous(c)))
    }

    // A vertex walk cannot visit more corners than the table contains, even
    // when malformed opposite links form a cycle not containing its start.
    fn walk_step(&self, steps: &mut usize) -> Result<()> {
        if *steps >= self.vertices.len() {
            return Err("cyclic Draco Edgebreaker vertex fan");
        }
        *steps += 1;
        Ok(())
    }

    fn compact(&mut self, invalid: &[usize]) -> Result<usize> {
        let mut count = self.leftmost.len();
        for &dst in invalid {
            while count > 0 && self.leftmost[count - 1] == INVALID {
                count -= 1;
            }
            let src = count.checked_sub(1).ok_or(TOPOLOGY_ERROR)?;
            if src < dst {
                continue;
            }
            // VertexCornersIterator first swings left, then right from the
            // original corner if it reaches a boundary.
            let start = self.leftmost(src)?;
            let mut c = start;
            let mut left = true;
            let mut steps = 0;
            while c != INVALID {
                self.walk_step(&mut steps)?;
                if self.vertex(c)? != src {
                    return Err(TOPOLOGY_ERROR);
                }
                self.map(c, dst)?;
                if left {
                    c = self.swing_left(c);
                    if c == INVALID {
                        c = self.swing_right(start);
                        left = false;
                    } else if c == start {
                        c = INVALID;
                    }
                } else {
                    c = self.swing_right(c);
                }
            }
            self.set_leftmost(dst, start)?;
            self.set_leftmost(src, INVALID)?;
            self.holes[dst] = self.holes[src];
            self.holes[src] = false;
            count -= 1;
        }
        Ok(count)
    }
}

struct SplitEvent {
    source: u32,
    split: u32,
    right: bool,
}

fn split_events(r: &mut Reader<'_>, faces: usize) -> Result<Vec<SplitEvent>> {
    let count = r.count()?;
    if count > faces || count > r.remaining() / 2 {
        return Err("invalid Draco Edgebreaker split event count");
    }
    let mut events = Vec::new();
    let mut source = 0u32;
    for _ in 0..count {
        source = source
            .checked_add(r.varint(32)? as u32)
            .ok_or("Draco Edgebreaker split source overflow")?;
        let split = source
            .checked_sub(r.varint(32)? as u32)
            .ok_or("invalid Draco Edgebreaker split delta")?;
        push(
            &mut events,
            SplitEvent {
                source,
                split,
                right: false,
            },
        )?;
    }
    let mut bits = Bits::new(&r.bytes[r.offset..]);
    for event in &mut events {
        event.right = bits.read(1)? != 0;
    }
    r.take(bits.consumed())?;
    // Hole events are not written in bitstream 2.2.
    Ok(events)
}

enum Symbols<'a> {
    Standard(Bits<'a>),
    Valence {
        contexts: [Vec<u32>; 6],
        valences: Vec<u32>,
        active: Option<usize>,
    },
}

struct Traversal<'a> {
    symbols: Symbols<'a>,
    starts: Binary<'a>,
    seams: Vec<Binary<'a>>,
}

impl<'a> Traversal<'a> {
    fn new(r: &mut Reader<'a>, valence: bool, attributes: usize, faces: usize) -> Result<Self> {
        let standard = if valence {
            None
        } else {
            let size = usize::try_from(r.varint(64)?)
                .map_err(|_| "Draco Edgebreaker traversal size overflow")?;
            Some(Bits::new(r.take(size)?))
        };
        let starts = Binary::new(r)?;
        let mut seams = Vec::new();
        for _ in 0..attributes {
            push(&mut seams, Binary::new(r)?)?;
        }
        let symbols = if let Some(bits) = standard {
            Symbols::Standard(bits)
        } else {
            let mut contexts = std::array::from_fn(|_| Vec::new());
            for context in &mut contexts {
                let count = r.count()?;
                if count > faces {
                    return Err("invalid Draco Edgebreaker valence context size");
                }
                if count != 0 {
                    *context = entropy::symbols(r, count, 1)?;
                }
            }
            Symbols::Valence {
                contexts,
                valences: Vec::new(),
                active: None,
            }
        };
        Ok(Self {
            symbols,
            starts,
            seams,
        })
    }

    fn symbol(&mut self) -> Result<u32> {
        match &mut self.symbols {
            Symbols::Standard(bits) => {
                let first = bits.read(1)?;
                Ok(if first == C {
                    C
                } else {
                    first | bits.read(2)? << 1
                })
            }
            Symbols::Valence {
                contexts, active, ..
            } => {
                let Some(context) = *active else {
                    return Ok(E);
                };
                let id = contexts[context]
                    .pop()
                    .ok_or("exhausted Draco Edgebreaker valence context")?;
                [C, S, L, R, E]
                    .get(id as usize)
                    .copied()
                    .ok_or("invalid Draco Edgebreaker valence symbol")
            }
        }
    }

    fn merge(&mut self, dest: usize, source: usize) -> Result<()> {
        if let Symbols::Valence { valences, .. } = &mut self.symbols {
            let source = *valences.get(source).ok_or(TOPOLOGY_ERROR)?;
            let dest = valences.get_mut(dest).ok_or(TOPOLOGY_ERROR)?;
            *dest = dest.checked_add(source).ok_or(TOPOLOGY_ERROR)?;
        }
        Ok(())
    }

    fn new_active(&mut self, table: &CornerTable, c: usize, symbol: u32) -> Result<()> {
        if let Symbols::Valence {
            valences, active, ..
        } = &mut self.symbols
        {
            if valences.len() < table.leftmost.len() {
                valences
                    .try_reserve(table.leftmost.len() - valences.len())
                    .map_err(|_| ALLOCATION_ERROR)?;
                valences.resize(table.leftmost.len(), 0);
            }
            let vertices = [
                table.vertex(c)?,
                table.vertex(next(c))?,
                table.vertex(previous(c))?,
            ];
            let increments = match symbol {
                C | S => [0, 1, 1],
                R => [1, 1, 2],
                L => [1, 2, 1],
                E => [2, 2, 2],
                _ => return Err(TOPOLOGY_ERROR),
            };
            for (v, increment) in vertices.into_iter().zip(increments) {
                valences[v] = valences[v].checked_add(increment).ok_or(TOPOLOGY_ERROR)?;
            }
            *active = Some((valences[vertices[1]].clamp(2, 7) - 2) as usize);
        }
        Ok(())
    }
}

fn connectivity(
    table: &mut CornerTable,
    traversal: &mut Traversal<'_>,
    num_symbols: usize,
    events: &mut Vec<SplitEvent>,
    compact: bool,
) -> Result<()> {
    let mut active = Vec::new();
    let mut split_active = HashMap::new();
    let mut invalid = Vec::new();
    for symbol_id in 0..num_symbols {
        let symbol = traversal.symbol()?;
        let corner = table.add_face()?;
        let mut check_split = false;
        match symbol {
            C => {
                let a = *active.last().ok_or(TOPOLOGY_ERROR)?;
                let x = table.vertex(next(a))?;
                let b = next(table.leftmost(x)?);
                if a == b || table.opposite(a) != INVALID || table.opposite(b) != INVALID {
                    return Err(TOPOLOGY_ERROR);
                }
                let a_prev = table.vertex(previous(a))?;
                let b_next = table.vertex(next(b))?;
                if x == a_prev || x == b_next {
                    return Err(TOPOLOGY_ERROR);
                }
                table.link(a, corner + 1)?;
                table.link(b, corner + 2)?;
                table.map(corner, x)?;
                table.map(corner + 1, b_next)?;
                table.map(corner + 2, a_prev)?;
                table.set_leftmost(a_prev, corner + 2)?;
                table.interior(x)?;
                *active.last_mut().ok_or(TOPOLOGY_ERROR)? = corner;
            }
            R | L => {
                let a = *active.last().ok_or(TOPOLOGY_ERROR)?;
                let (opp, left, right) = if symbol == R {
                    (corner + 2, corner + 1, corner)
                } else {
                    (corner + 1, corner, corner + 2)
                };
                table.link(opp, a)?;
                let v = table.add_vertex()?;
                table.map(opp, v)?;
                table.set_leftmost(v, opp)?;
                let v_right = table.vertex(previous(a))?;
                table.map(right, v_right)?;
                table.set_leftmost(v_right, right)?;
                table.map(left, table.vertex(next(a))?)?;
                *active.last_mut().ok_or(TOPOLOGY_ERROR)? = corner;
                check_split = true;
            }
            S => {
                let b = active.pop().ok_or(TOPOLOGY_ERROR)?;
                if let Some(&a) = split_active.get(&symbol_id) {
                    push(&mut active, a)?;
                }
                let a = *active.last().ok_or(TOPOLOGY_ERROR)?;
                if a == b || table.opposite(a) != INVALID || table.opposite(b) != INVALID {
                    return Err(TOPOLOGY_ERROR);
                }
                table.link(a, corner + 2)?;
                table.link(b, corner + 1)?;
                let p = table.vertex(previous(a))?;
                table.map(corner, p)?;
                table.map(corner + 1, table.vertex(next(a))?)?;
                let b_prev = table.vertex(previous(b))?;
                table.map(corner + 2, b_prev)?;
                table.set_leftmost(b_prev, corner + 2)?;
                let mut n_corner = next(b);
                let n = table.vertex(n_corner)?;
                traversal.merge(p, n)?;
                table.set_leftmost(p, table.leftmost(n)?)?;
                let first = n_corner;
                let mut steps = 0;
                while n_corner != INVALID {
                    table.walk_step(&mut steps)?;
                    table.map(n_corner, p)?;
                    n_corner = table.swing_left(n_corner);
                    if n_corner == first {
                        return Err(TOPOLOGY_ERROR);
                    }
                }
                table.set_leftmost(n, INVALID)?;
                if compact {
                    push(&mut invalid, n)?;
                }
                *active.last_mut().ok_or(TOPOLOGY_ERROR)? = corner;
            }
            E => {
                for local in 0..3 {
                    let v = table.add_vertex()?;
                    table.map(corner + local, v)?;
                    table.set_leftmost(v, corner + local)?;
                }
                push(&mut active, corner)?;
                check_split = true;
            }
            _ => return Err("invalid Draco Edgebreaker topology symbol"),
        }
        let top = *active.last().ok_or(TOPOLOGY_ERROR)?;
        traversal.new_active(table, top, symbol)?;
        if check_split {
            let encoder_id = (num_symbols - symbol_id - 1) as u32;
            while let Some(event) = events.last() {
                if event.source > encoder_id {
                    return Err("missed Draco Edgebreaker topology split event");
                }
                if event.source != encoder_id {
                    break;
                }
                let split_id = num_symbols
                    .checked_sub(event.split as usize + 1)
                    .ok_or(TOPOLOGY_ERROR)?;
                let split_corner = if event.right {
                    next(top)
                } else {
                    previous(top)
                };
                split_active.try_reserve(1).map_err(|_| ALLOCATION_ERROR)?;
                split_active.insert(split_id, split_corner);
                events.pop();
            }
        }
    }
    while let Some(a) = active.pop() {
        if !traversal.starts.read()? {
            continue;
        }
        let n = table.vertex(next(a))?;
        let b = next(table.leftmost(n)?);
        let x = table.vertex(next(b))?;
        let c = next(table.leftmost(x)?);
        if a == b || a == c || b == c {
            return Err(TOPOLOGY_ERROR);
        }
        let p = table.vertex(next(c))?;
        let new = table.add_face()?;
        table.link(new, a)?;
        table.link(new + 1, b)?;
        table.link(new + 2, c)?;
        for (local, v) in [x, p, n].into_iter().enumerate() {
            table.map(new + local, v)?;
            table.interior(v)?;
        }
    }
    if table.vertices.len() / 3 != table.max_faces {
        return Err("unexpected Draco Edgebreaker face count");
    }
    if compact {
        table.compact(&invalid)?;
    }
    Ok(())
}

/// MeshAttributeCornerTable: base connectivity cut along one attribute's seams.
pub(crate) struct SeamTable {
    pub edges: Vec<bool>,
    on_seam: Vec<bool>,
    pub vertices: Vec<usize>,
    pub leftmost: Vec<usize>,
}

impl SeamTable {
    fn new(table: &CornerTable) -> Result<Self> {
        Ok(Self {
            edges: filled(table.vertices.len(), false)?,
            on_seam: filled(table.leftmost.len(), false)?,
            vertices: filled(table.vertices.len(), INVALID)?,
            leftmost: Vec::new(),
        })
    }

    fn add_seam(&mut self, table: &CornerTable, c: usize) -> Result<()> {
        let opp = table.opposite(c);
        for edge in [c, opp] {
            if edge == INVALID {
                continue;
            }
            *self.edges.get_mut(edge).ok_or(TOPOLOGY_ERROR)? = true;
            self.on_seam[table.vertex(next(edge))?] = true;
            self.on_seam[table.vertex(previous(edge))?] = true;
        }
        Ok(())
    }

    fn swing_left(&self, table: &CornerTable, c: usize) -> usize {
        if self.edges[next(c)] {
            INVALID
        } else {
            table.swing_left(c)
        }
    }

    fn recompute_vertices(&mut self, table: &CornerTable) -> Result<()> {
        let mut count = 0;
        for v in 0..table.leftmost.len() {
            let c = table.leftmost(v)?;
            if c == INVALID {
                continue;
            }
            let mut id = count;
            count += 1;
            let mut first = c;
            let leftmost = self.leftmost.len();
            push(&mut self.leftmost, c)?;
            if self.on_seam[v] {
                let mut act = self.swing_left(table, first);
                let mut steps = 0;
                while act != INVALID {
                    table.walk_step(&mut steps)?;
                    first = act;
                    act = self.swing_left(table, act);
                    if act == c {
                        return Err(TOPOLOGY_ERROR);
                    }
                }
            }
            if table.vertex(first)? != v || self.vertices[first] != INVALID {
                return Err(TOPOLOGY_ERROR);
            }
            self.vertices[first] = id;
            self.leftmost[leftmost] = first;
            let mut act = table.swing_right(first);
            let mut steps = 0;
            while act != INVALID && act != first {
                table.walk_step(&mut steps)?;
                if table.vertex(act)? != v || self.vertices[act] != INVALID {
                    return Err(TOPOLOGY_ERROR);
                }
                if self.edges[next(act)] {
                    id = count;
                    count += 1;
                    push(&mut self.leftmost, act)?;
                }
                self.vertices[act] = id;
                act = table.swing_right(act);
            }
        }
        Ok(())
    }
}

fn assign_points(
    table: &CornerTable,
    attributes: &[SeamTable],
    expected_points: usize,
) -> Result<Vec<[u32; 3]>> {
    let mut points = filled(table.vertices.len(), u32::MAX)?;
    if attributes.is_empty() {
        if table
            .leftmost
            .iter()
            .filter(|&&corner| corner != INVALID)
            .count()
            != expected_points
        {
            return Err("Draco point count mismatch");
        }
        for (c, point) in points.iter_mut().enumerate() {
            *point = table.vertex(c)? as u32;
        }
    } else {
        let mut count = 0u32;
        for v in 0..table.leftmost.len() {
            let c = table.leftmost(v)?;
            if c == INVALID {
                continue;
            }
            let mut first = c;
            if !table.holes[v] {
                for attribute in attributes {
                    if !attribute.on_seam[v] {
                        continue;
                    }
                    let id = attribute.vertices[c];
                    let mut act = table.swing_right(c);
                    let mut steps = 0;
                    let mut found = false;
                    while act != c {
                        table.walk_step(&mut steps)?;
                        if act == INVALID {
                            return Err(TOPOLOGY_ERROR);
                        }
                        if attribute.vertices[act] != id {
                            first = act;
                            found = true;
                            break;
                        }
                        act = table.swing_right(act);
                    }
                    if found {
                        break;
                    }
                }
            }
            if table.vertex(first)? != v || points[first] != u32::MAX {
                return Err(TOPOLOGY_ERROR);
            }
            points[first] = count;
            count += 1;
            let mut prev = first;
            let mut act = table.swing_right(first);
            let mut steps = 0;
            while act != INVALID && act != first {
                table.walk_step(&mut steps)?;
                if table.vertex(act)? != v || points[act] != u32::MAX {
                    return Err(TOPOLOGY_ERROR);
                }
                if attributes
                    .iter()
                    .any(|a| a.vertices[act] != a.vertices[prev])
                {
                    points[act] = count;
                    count += 1;
                } else {
                    points[act] = points[prev];
                }
                prev = act;
                act = table.swing_right(act);
            }
        }
        if count as usize != expected_points {
            return Err("Draco point count mismatch");
        }
        if points.contains(&u32::MAX) {
            return Err(TOPOLOGY_ERROR);
        }
    }
    let mut faces = Vec::new();
    faces
        .try_reserve_exact(table.max_faces)
        .map_err(|_| ALLOCATION_ERROR)?;
    faces.extend_from_slice(points.as_chunks::<3>().0);
    Ok(faces)
}

/// Decoded connectivity and the tables its attribute decoders traverse.
pub(crate) struct Connectivity {
    pub faces: Vec<[u32; 3]>,
    pub table: CornerTable,
    pub seams: Vec<SeamTable>,
}

/// Decode bitstream 2.2 connectivity, leaving the reader at attribute decoders.
pub(crate) fn decode(
    r: &mut Reader<'_>,
    expected_faces: usize,
    expected_points: usize,
) -> Result<Connectivity> {
    let valence = match r.byte()? {
        0 => false,
        1 => return Err("deprecated Draco predictive Edgebreaker is not supported"),
        2 => true,
        _ => return Err("unsupported Draco Edgebreaker traversal decoder"),
    };
    let vertices = r.count()?;
    let faces = r.count()?;
    if faces != expected_faces {
        return Err("Draco face count mismatch");
    }
    if vertices > expected_points {
        return Err("Draco point count mismatch");
    }
    if faces > i32::MAX as usize / 3 || vertices > faces * 3 {
        return Err("invalid Draco Edgebreaker mesh size");
    }
    let min_edges = faces * 3 / 2;
    let max_edges = (vertices as u64) * (vertices.saturating_sub(1) as u64) / 2;
    if max_edges < min_edges as u64 {
        return Err("impossible Draco Edgebreaker vertex and face counts");
    }
    let attributes = usize::from(r.byte()?);
    let symbols = r.count()?;
    if symbols > faces || faces > symbols + symbols / 3 {
        return Err("invalid Draco Edgebreaker symbol count");
    }
    let splits = r.count()?;
    if splits > symbols {
        return Err("invalid Draco Edgebreaker split symbol count");
    }
    let max_vertices = vertices.checked_add(splits).ok_or(TOPOLOGY_ERROR)?;
    if max_vertices > i32::MAX as usize {
        return Err("invalid Draco Edgebreaker vertex count");
    }
    let mut events = split_events(r, faces)?;
    let mut traversal = Traversal::new(r, valence, attributes, faces)?;
    let mut table = CornerTable::new(faces, max_vertices);
    connectivity(
        &mut table,
        &mut traversal,
        symbols,
        &mut events,
        attributes == 0,
    )?;
    let mut attribute_tables = Vec::new();
    for _ in 0..attributes {
        push(&mut attribute_tables, SeamTable::new(&table)?)?;
    }
    for c in 0..table.vertices.len() {
        let opp = table.opposite(c);
        if opp != INVALID && opp / 3 < c / 3 {
            continue;
        }
        for (attribute, decoder) in attribute_tables.iter_mut().zip(&mut traversal.seams) {
            if opp == INVALID || decoder.read()? {
                attribute.add_seam(&table, c)?;
            }
        }
    }
    for attribute in &mut attribute_tables {
        attribute.recompute_vertices(&table)?;
    }
    let faces = assign_points(&table, &attribute_tables, expected_points)?;
    Ok(Connectivity {
        faces,
        table,
        seams: attribute_tables,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Connectivity prefixes and faces from the pinned encoder/decoder API.
    const STANDARD_CLOSED: &[u8] = &[0, 4, 4, 0, 3, 0, 0, 1, 47, 1, 1, 16];
    const VALENCE_CLOSED: &[u8] = &[
        2, 4, 4, 0, 3, 0, 0, 1, 1, 16, 1, 1, 1, 4, 11, 1, 64, 1, 0, 1, 1, 1, 1, 1, 64, 1, 0, 0, 0,
        0, 0,
    ];
    const STANDARD_SEAMS: &[u8] = &[0, 4, 2, 1, 2, 0, 0, 1, 31, 255, 1, 17, 1, 1, 16];
    const CLOSED_FACES: [[u32; 3]; 4] = [[0, 1, 2], [2, 1, 3], [1, 0, 3], [2, 3, 0]];

    #[test]
    fn standard_and_valence_initial_faces_match_pin_and_stop_before_attributes() {
        for encoded in [STANDARD_CLOSED, VALENCE_CLOSED] {
            let mut bytes = encoded.to_vec();
            bytes.extend_from_slice(&[0x87, 0x43]);
            let mut r = Reader::new(&bytes);
            assert_eq!(decode(&mut r, 4, 4).unwrap().faces, CLOSED_FACES);
            assert_eq!(r.offset, encoded.len());
            assert_eq!(r.take(2).unwrap(), &[0x87, 0x43]);
        }
    }

    #[test]
    fn caller_counts_bound_header_and_require_final_point_count() {
        assert_eq!(
            decode(&mut Reader::new(STANDARD_CLOSED), 5, 4)
                .err()
                .unwrap(),
            "Draco face count mismatch"
        );
        assert_eq!(
            decode(&mut Reader::new(STANDARD_CLOSED), 4, 3)
                .err()
                .unwrap(),
            "Draco point count mismatch"
        );
        assert_eq!(
            decode(&mut Reader::new(STANDARD_CLOSED), 4, 5)
                .err()
                .unwrap(),
            "Draco point count mismatch"
        );
        for points in [4, 5, 7] {
            assert_eq!(
                decode(&mut Reader::new(STANDARD_SEAMS), 2, points)
                    .err()
                    .unwrap(),
                "Draco point count mismatch"
            );
        }
    }

    #[test]
    fn attribute_seams_split_points_in_pinned_vertex_order() {
        let mut r = Reader::new(STANDARD_SEAMS);
        assert_eq!(decode(&mut r, 2, 6).unwrap().faces, [[0, 1, 4], [2, 5, 3]]);
        assert_eq!(r.offset, STANDARD_SEAMS.len());
    }

    #[test]
    fn connectivity_prefix_truncations_are_errors() {
        for encoded in [STANDARD_CLOSED, VALENCE_CLOSED, STANDARD_SEAMS] {
            for end in 0..encoded.len() {
                assert!(decode(
                    &mut Reader::new(&encoded[..end]),
                    if encoded == STANDARD_SEAMS { 2 } else { 4 },
                    if encoded == STANDARD_SEAMS { 6 } else { 4 }
                )
                .is_err());
            }
        }
    }

    #[test]
    fn small_connectivity_bit_mutations_never_panic() {
        for encoded in [STANDARD_CLOSED, VALENCE_CLOSED, STANDARD_SEAMS] {
            for offset in 0..encoded.len() {
                for bit in 0..8 {
                    let mut bytes = encoded.to_vec();
                    bytes[offset] ^= 1 << bit;
                    assert!(std::panic::catch_unwind(|| decode(
                        &mut Reader::new(&bytes),
                        if encoded == STANDARD_SEAMS { 2 } else { 4 },
                        if encoded == STANDARD_SEAMS { 6 } else { 4 }
                    ))
                    .is_ok());
                }
            }
        }
    }

    #[test]
    fn predictive_traversal_is_refused_by_name() {
        assert!(decode(&mut Reader::new(&[1]), 0, 0)
            .err()
            .unwrap()
            .contains("predictive"));
    }

    #[test]
    fn split_event_deltas_share_one_byte_packed_edge_stream() {
        let bytes = [3, 5, 4, 0, 2, 4, 8, 0b101, 0x87];
        let mut r = Reader::new(&bytes);
        let events = split_events(&mut r, 3).unwrap();
        assert_eq!(r.offset, 8);
        let decoded: Vec<_> = events
            .iter()
            .map(|e| (e.source, e.split, e.right))
            .collect();
        assert_eq!(decoded, [(5, 1, true), (5, 3, false), (9, 1, true)]);
        assert!(split_events(&mut Reader::new(&[1, 1, 2, 0]), 1).is_err());
    }

    fn tetrahedron() -> CornerTable {
        let mut table = CornerTable::new(4, 4);
        for _ in 0..4 {
            table.add_vertex().unwrap();
        }
        for face in CLOSED_FACES {
            let c = table.add_face().unwrap();
            for (local, v) in face.into_iter().enumerate() {
                table.map(c + local, v as usize).unwrap();
            }
        }
        for (a, b) in [(0, 5), (1, 10), (2, 8), (3, 7), (4, 11), (6, 9)] {
            table.link(a, b).unwrap();
        }
        for (v, c) in [0, 1, 3, 8].into_iter().enumerate() {
            table.set_leftmost(v, c).unwrap();
            table.interior(v).unwrap();
        }
        table
    }

    #[test]
    fn interior_seam_rotation_uses_first_attribute_with_a_discontinuity() {
        let table = tetrahedron();
        let mut a = SeamTable::new(&table).unwrap();
        let mut b = SeamTable::new(&table).unwrap();
        // Each seam loop encloses one face. At vertex zero the clockwise fan
        // is [0, 7, 11], with a discontinuity at 7 for A and at 11 for B.
        for c in [6, 7, 8] {
            a.add_seam(&table, c).unwrap();
        }
        for c in [9, 10, 11] {
            b.add_seam(&table, c).unwrap();
        }
        a.recompute_vertices(&table).unwrap();
        b.recompute_vertices(&table).unwrap();
        let mut attributes = [a, b];
        let faces = assign_points(&table, &attributes, 10).unwrap();
        assert_eq!([faces[2][1], faces[3][2], faces[0][0]], [0, 1, 2]);
        attributes.swap(0, 1);
        let faces = assign_points(&table, &attributes, 10).unwrap();
        assert_eq!([faces[3][2], faces[0][0], faces[2][1]], [0, 1, 2]);
    }

    #[test]
    fn invalid_vertices_are_filled_from_last_valid_not_stably_renumbered() {
        let mut table = CornerTable::new(1, 5);
        for _ in 0..5 {
            table.add_vertex().unwrap();
        }
        let c = table.add_face().unwrap();
        for (local, v) in [4, 2, 3].into_iter().enumerate() {
            table.map(c + local, v).unwrap();
            table.set_leftmost(v, c + local).unwrap();
        }
        assert_eq!(table.compact(&[0, 1]).unwrap(), 3);
        assert_eq!(assign_points(&table, &[], 3).unwrap(), [[0, 2, 1]]);
    }

    #[test]
    fn valence_symbols_are_reversed_and_invalid_context_ids_are_errors() {
        let mut r = Reader::new(&[255, 1, 0]);
        let mut traversal = Traversal {
            symbols: Symbols::Valence {
                contexts: [vec![5, 0], vec![], vec![], vec![], vec![], vec![]],
                valences: vec![],
                active: None,
            },
            starts: Binary::new(&mut r).unwrap(),
            seams: vec![],
        };
        assert_eq!(traversal.symbol().unwrap(), E);
        let mut table = CornerTable::new(1, 3);
        table.add_face().unwrap();
        for v in 0..3 {
            table.add_vertex().unwrap();
            table.map(v, v).unwrap();
        }
        traversal.new_active(&table, 0, E).unwrap();
        assert_eq!(traversal.symbol().unwrap(), C);
        assert!(traversal.symbol().is_err());
        traversal.merge(0, 1).unwrap();
        if let Symbols::Valence {
            valences, active, ..
        } = &traversal.symbols
        {
            assert_eq!(valences, &[4, 2, 2]);
            assert_eq!(*active, Some(0));
        }
    }
}
