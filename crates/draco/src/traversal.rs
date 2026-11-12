// Port of Draco 55462882: mesh/corner_table.h, mesh/mesh_attribute_corner_table.h,
// compression/mesh/traverser/*, linear_sequencer.h and
// mesh_attribute_indices_encoding_data.h.
use crate::{
    allocation::{filled, push},
    edgebreaker::{face, neighbours, next, previous, CornerTable, SeamTable, INVALID},
    Result,
};

const TRAVERSAL_ERROR: &str = "invalid Draco attribute traversal";

/// The corner-table operations Draco's traversers and mesh predictors use.
pub(crate) trait Corners {
    fn num_vertices(&self) -> usize;
    fn num_corners(&self) -> usize;
    fn opposite(&self, c: usize) -> usize;
    /// `INVALID` for an invalid corner or a corner no vertex claims.
    fn vertex(&self, c: usize) -> usize;
    fn leftmost(&self, v: usize) -> usize;
    fn is_on_boundary(&self, v: usize) -> bool;
    fn swing_left(&self, c: usize) -> usize {
        next(self.opposite(next(c)))
    }
    fn swing_right(&self, c: usize) -> usize {
        previous(self.opposite(previous(c)))
    }
    fn right_corner(&self, c: usize) -> usize {
        if c == INVALID {
            INVALID
        } else {
            self.opposite(next(c))
        }
    }
    fn num_faces(&self) -> usize {
        self.num_corners() / 3
    }
}

impl Corners for CornerTable {
    fn num_vertices(&self) -> usize {
        self.leftmost.len()
    }
    fn num_corners(&self) -> usize {
        self.vertices.len()
    }
    fn opposite(&self, c: usize) -> usize {
        CornerTable::opposite(self, c)
    }
    fn vertex(&self, c: usize) -> usize {
        self.vertices.get(c).copied().unwrap_or(INVALID)
    }
    fn leftmost(&self, v: usize) -> usize {
        self.leftmost.get(v).copied().unwrap_or(INVALID)
    }
    fn is_on_boundary(&self, v: usize) -> bool {
        self.swing_left(self.leftmost(v)) == INVALID
    }
}

pub(crate) struct AttributeCorners<'a> {
    pub base: &'a CornerTable,
    pub seams: &'a SeamTable,
}

impl Corners for AttributeCorners<'_> {
    fn num_vertices(&self) -> usize {
        self.seams.leftmost.len()
    }
    fn num_corners(&self) -> usize {
        self.base.vertices.len()
    }
    fn opposite(&self, c: usize) -> usize {
        if c == INVALID || self.seams.edges.get(c).copied().unwrap_or(true) {
            INVALID
        } else {
            self.base.opposite(c)
        }
    }
    fn vertex(&self, c: usize) -> usize {
        self.seams.vertices.get(c).copied().unwrap_or(INVALID)
    }
    fn leftmost(&self, v: usize) -> usize {
        self.seams.leftmost.get(v).copied().unwrap_or(INVALID)
    }
    fn is_on_boundary(&self, v: usize) -> bool {
        let corner = self.leftmost(v);
        corner == INVALID || self.swing_left(corner) == INVALID
    }
}

/// The table an attribute traverses, dispatched statically in its hot loops.
#[derive(Clone, Copy)]
pub(crate) enum Table<'a> {
    Base(&'a CornerTable),
    Seams(&'a AttributeCorners<'a>),
}

macro_rules! delegate {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $out:ty;)*) => {
        impl Corners for Table<'_> {
            $(
                #[inline]
                fn $name(&self, $($arg: $ty),*) -> $out {
                    match self {
                        Table::Base(table) => table.$name($($arg),*),
                        Table::Seams(table) => table.$name($($arg),*),
                    }
                }
            )*
        }
    };
}
delegate! {
    num_vertices() -> usize;
    num_corners() -> usize;
    opposite(c: usize) -> usize;
    vertex(c: usize) -> usize;
    leftmost(v: usize) -> usize;
    is_on_boundary(v: usize) -> bool;
}

/// MeshAttributeIndicesEncodingData. Unvisited vertices map to value zero, as
/// `std::vector::resize` leaves them in the pinned decoder.
#[derive(Default)]
pub(crate) struct Encoding {
    pub vertex_to_value: Vec<i32>,
    pub value_to_corner: Vec<usize>,
}

impl Encoding {
    pub fn new(vertices: usize) -> Result<Self> {
        Ok(Self {
            vertex_to_value: filled(vertices, 0)?,
            value_to_corner: Vec::new(),
        })
    }

    #[inline(always)]
    /// Records value `count`; `points` and `value_to_corner` are pre-sized
    /// to the vertex count, which bounds the values one traversal visits.
    fn visit(
        &mut self,
        faces: &[[u32; 3]],
        points: &mut [u32],
        count: usize,
        vertex: usize,
        corner: usize,
    ) -> Result<()> {
        let point = faces.get(face(corner)).ok_or(TRAVERSAL_ERROR)?[corner - 3 * face(corner)];
        let value = i32::try_from(count).map_err(|_| TRAVERSAL_ERROR)?;
        *self
            .vertex_to_value
            .get_mut(vertex)
            .ok_or(TRAVERSAL_ERROR)? = value;
        *points.get_mut(count).ok_or(TRAVERSAL_ERROR)? = point;
        *self.value_to_corner.get_mut(count).ok_or(TRAVERSAL_ERROR)? = corner;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Order {
    DepthFirst,
    PredictionDegree,
}

struct Visits {
    faces: Vec<bool>,
    vertices: Vec<bool>,
}

impl Visits {
    fn face(&self, c: usize) -> bool {
        c == INVALID || self.faces.get(face(c)).copied().unwrap_or(true)
    }
    fn vertex(&self, v: usize) -> Result<bool> {
        self.vertices.get(v).copied().ok_or(TRAVERSAL_ERROR)
    }
    // Marks `v` and reports it to the observer if it was unvisited.
    #[inline(always)]
    fn visit(&mut self, context: &mut Observer<'_>, v: usize, corner: usize) -> Result<()> {
        if !self.vertex(v)? {
            self.vertices[v] = true;
            context
                .encoding
                .visit(context.faces, context.points, context.count, v, corner)?;
            context.count += 1;
        }
        Ok(())
    }
}

struct Observer<'a> {
    faces: &'a [[u32; 3]],
    encoding: &'a mut Encoding,
    points: &'a mut [u32],
    count: usize,
}

/// MeshTraversalSequencer::GenerateSequence: the point order in which the
/// attribute values were encoded, recording each vertex's value in `encoding`.
pub(crate) fn sequence(
    table: &(impl Corners + ?Sized),
    order: Order,
    faces: &[[u32; 3]],
    encoding: &mut Encoding,
) -> Result<Vec<u32>> {
    let mut points = filled(table.num_vertices(), 0)?;
    encoding.value_to_corner = filled(table.num_vertices(), 0)?;
    let mut visits = Visits {
        faces: filled(table.num_faces(), false)?,
        vertices: filled(table.num_vertices(), false)?,
    };
    let mut observer = Observer {
        faces,
        encoding,
        points: &mut points,
        count: 0,
    };
    match order {
        Order::DepthFirst => {
            let mut stack = Vec::new();
            for face in 0..table.num_faces() {
                depth_first(table, &mut visits, &mut observer, &mut stack, 3 * face)?;
            }
        }
        Order::PredictionDegree => {
            let mut degree = filled(table.num_vertices(), 0u32)?;
            let mut stacks: [Vec<usize>; 3] = Default::default();
            for face in 0..table.num_faces() {
                if degree.is_empty() {
                    continue;
                }
                prediction_degree(
                    table,
                    &mut visits,
                    &mut observer,
                    &mut degree,
                    &mut stacks,
                    3 * face,
                )?;
            }
        }
    }
    let count = observer.count;
    encoding.value_to_corner.truncate(count);
    points.truncate(count);
    Ok(points)
}

fn depth_first(
    table: &(impl Corners + ?Sized),
    visits: &mut Visits,
    observer: &mut Observer<'_>,
    stack: &mut Vec<usize>,
    start: usize,
) -> Result<()> {
    if visits.face(start) {
        return Ok(());
    }
    stack.clear();
    push(stack, start)?;
    let next_vertex = table.vertex(next(start));
    let previous_vertex = table.vertex(previous(start));
    if next_vertex == INVALID || previous_vertex == INVALID {
        return Err(TRAVERSAL_ERROR);
    }
    visits.visit(observer, next_vertex, next(start))?;
    visits.visit(observer, previous_vertex, previous(start))?;
    while let Some(&top) = stack.last() {
        let mut corner = top;
        if corner == INVALID || visits.face(corner) {
            stack.pop();
            continue;
        }
        loop {
            visits.faces[face(corner)] = true;
            let vertex = table.vertex(corner);
            if vertex == INVALID {
                return Err(TRAVERSAL_ERROR);
            }
            if !visits.vertex(vertex)? {
                let boundary = table.is_on_boundary(vertex);
                visits.visit(observer, vertex, corner)?;
                if !boundary {
                    corner = table.right_corner(corner);
                    if corner == INVALID {
                        return Err(TRAVERSAL_ERROR);
                    }
                    continue;
                }
            }
            let (next, previous) = neighbours(corner);
            let (right, left) = (table.opposite(next), table.opposite(previous));
            match (visits.face(right), visits.face(left)) {
                (true, true) => {
                    stack.pop();
                    break;
                }
                (true, false) => corner = left,
                (false, true) => corner = right,
                (false, false) => {
                    *stack.last_mut().unwrap() = left;
                    push(stack, right)?;
                    break;
                }
            }
        }
    }
    Ok(())
}

fn prediction_degree(
    table: &(impl Corners + ?Sized),
    visits: &mut Visits,
    observer: &mut Observer<'_>,
    degree: &mut [u32],
    stacks: &mut [Vec<usize>; 3],
    start: usize,
) -> Result<()> {
    push(&mut stacks[0], start)?;
    let mut best = 0;
    for corner in [next(start), previous(start), start] {
        visits.visit(observer, table.vertex(corner), corner)?;
    }
    loop {
        let Some(priority) = (best..3).find(|&i| !stacks[i].is_empty()) else {
            return Ok(());
        };
        best = priority;
        let mut corner = stacks[priority].pop().unwrap();
        if visits.face(corner) {
            continue;
        }
        loop {
            visits.faces[face(corner)] = true;
            visits.visit(observer, table.vertex(corner), corner)?;
            let (next, previous) = neighbours(corner);
            let (right, left) = (table.opposite(next), table.opposite(previous));
            let right_visited = visits.face(right);
            if !visits.face(left) {
                let priority = priority_of(table, visits, degree, left)?;
                if right_visited && priority <= best {
                    corner = left;
                    continue;
                }
                push(&mut stacks[priority], left)?;
                best = best.min(priority);
            }
            if !right_visited {
                let priority = priority_of(table, visits, degree, right)?;
                if priority <= best {
                    corner = right;
                    continue;
                }
                push(&mut stacks[priority], right)?;
                best = best.min(priority);
            }
            break;
        }
    }
}

fn priority_of(
    table: &(impl Corners + ?Sized),
    visits: &Visits,
    degree: &mut [u32],
    corner: usize,
) -> Result<usize> {
    let tip = table.vertex(corner);
    if visits.vertex(tip)? {
        return Ok(0);
    }
    let degree = degree.get_mut(tip).ok_or(TRAVERSAL_ERROR)?;
    *degree = degree.saturating_add(1);
    Ok(if *degree > 1 { 1 } else { 2 })
}
