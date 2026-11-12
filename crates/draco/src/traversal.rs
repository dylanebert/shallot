// Port of Draco 55462882: mesh/corner_table.h, mesh/mesh_attribute_corner_table.h,
// compression/mesh/traverser/*, linear_sequencer.h and
// mesh_attribute_indices_encoding_data.h.
use crate::{
    allocation::{filled, push},
    edgebreaker::{next, previous, CornerTable, SeamTable, INVALID},
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
    fn left_corner(&self, c: usize) -> usize {
        if c == INVALID {
            INVALID
        } else {
            self.opposite(previous(c))
        }
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

    fn visit(
        &mut self,
        faces: &[[u32; 3]],
        points: &mut Vec<u32>,
        vertex: usize,
        corner: usize,
    ) -> Result<()> {
        let point = faces.get(corner / 3).ok_or(TRAVERSAL_ERROR)?[corner % 3];
        let value = i32::try_from(self.value_to_corner.len()).map_err(|_| TRAVERSAL_ERROR)?;
        *self
            .vertex_to_value
            .get_mut(vertex)
            .ok_or(TRAVERSAL_ERROR)? = value;
        push(points, point)?;
        push(&mut self.value_to_corner, corner)
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
        c == INVALID || self.faces.get(c / 3).copied().unwrap_or(true)
    }
    fn vertex(&self, v: usize) -> Result<bool> {
        self.vertices.get(v).copied().ok_or(TRAVERSAL_ERROR)
    }
    // Marks `v` and reports it to the observer if it was unvisited.
    fn visit(&mut self, context: &mut Observer<'_>, v: usize, corner: usize) -> Result<()> {
        if !self.vertex(v)? {
            self.vertices[v] = true;
            context
                .encoding
                .visit(context.faces, context.points, v, corner)?;
        }
        Ok(())
    }
}

struct Observer<'a> {
    faces: &'a [[u32; 3]],
    encoding: &'a mut Encoding,
    points: &'a mut Vec<u32>,
}

/// MeshTraversalSequencer::GenerateSequence: the point order in which the
/// attribute values were encoded, recording each vertex's value in `encoding`.
pub(crate) fn sequence(
    table: &(impl Corners + ?Sized),
    order: Order,
    faces: &[[u32; 3]],
    encoding: &mut Encoding,
) -> Result<Vec<u32>> {
    let mut points = Vec::new();
    points
        .try_reserve(table.num_vertices())
        .map_err(|_| "Draco allocation failed")?;
    let mut visits = Visits {
        faces: filled(table.num_faces(), false)?,
        vertices: filled(table.num_vertices(), false)?,
    };
    let mut observer = Observer {
        faces,
        encoding,
        points: &mut points,
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
            visits.faces[corner / 3] = true;
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
            let right = table.right_corner(corner);
            let left = table.left_corner(corner);
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
            visits.faces[corner / 3] = true;
            visits.visit(observer, table.vertex(corner), corner)?;
            let right = table.right_corner(corner);
            let left = table.left_corner(corner);
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
