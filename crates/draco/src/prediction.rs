// Port of Draco 55462882: compression/attributes/prediction_schemes/*_decoder.h,
// their predictors and decoding transforms, normal_compression_utils.h and
// core/math_utils.h's IntSqrt. C++ signed overflow is undefined; these ports
// use two's-complement wrapping, which is what the pinned build computes.
use crate::{
    allocation::filled,
    buffer::Reader,
    edgebreaker::{next, previous, INVALID},
    entropy::Binary,
    traversal::{Corners, Encoding, Table},
    Result,
};

const PREDICTION_ERROR: &str = "invalid Draco attribute prediction";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Method {
    Delta,
    Parallelogram,
    MultiParallelogram,
    ConstrainedMultiParallelogram,
    TexCoordsDeprecated,
    TexCoordsPortable,
    GeometricNormal,
}

/// OctahedronToolBox. Unset fields keep the pinned constructor's -1 values.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Octahedron {
    bits: i32,
    max_quantized: i32,
    max_value: i32,
    scale: f32,
    center: i32,
}

impl Default for Octahedron {
    fn default() -> Self {
        Self {
            bits: -1,
            max_quantized: -1,
            max_value: -1,
            scale: 1.0,
            center: -1,
        }
    }
}

impl Octahedron {
    pub fn new(bits: i32) -> Option<Self> {
        let mut tool = Self::default();
        tool.set_bits(bits).then_some(tool)
    }

    fn set_bits(&mut self, bits: i32) -> bool {
        if !(2..=30).contains(&bits) {
            return false;
        }
        self.bits = bits;
        self.max_quantized = ((1u32 << bits) - 1) as i32;
        self.max_value = self.max_quantized - 1;
        self.scale = 2.0 / self.max_value as f32;
        self.center = self.max_value / 2;
        true
    }

    fn canonicalize_coords(&self, mut s: i32, mut t: i32) -> (i32, i32) {
        let (max, center) = (self.max_value, self.center);
        if (s == 0 && t == 0) || (s == 0 && t == max) || (s == max && t == 0) {
            s = max;
            t = max;
        } else if s == 0 && t > center {
            t = center.wrapping_sub(t.wrapping_sub(center));
        } else if s == max && t < center {
            t = center.wrapping_add(center.wrapping_sub(t));
        } else if t == max && s < center {
            s = center.wrapping_add(center.wrapping_sub(s));
        } else if t == 0 && s > center {
            s = center.wrapping_sub(s.wrapping_sub(center));
        }
        (s, t)
    }

    fn vector_to_coords(&self, v: [i32; 3]) -> (i32, i32) {
        let (s, t) = if v[0] >= 0 {
            (
                v[1].wrapping_add(self.center),
                v[2].wrapping_add(self.center),
            )
        } else {
            (
                if v[1] < 0 {
                    v[2].wrapping_abs()
                } else {
                    self.max_value.wrapping_sub(v[2].wrapping_abs())
                },
                if v[2] < 0 {
                    v[1].wrapping_abs()
                } else {
                    self.max_value.wrapping_sub(v[1].wrapping_abs())
                },
            )
        };
        self.canonicalize_coords(s, t)
    }

    fn canonicalize_vector(&self, v: &mut [i32; 3]) {
        let sum = v.iter().map(|&v| i64::from(v.wrapping_abs())).sum::<i64>();
        if sum == 0 {
            v[0] = self.center;
        } else {
            let center = i64::from(self.center);
            v[0] = (i64::from(v[0]) * center).wrapping_div(sum) as i32;
            v[1] = (i64::from(v[1]) * center).wrapping_div(sum) as i32;
            let rest = self
                .center
                .wrapping_sub(v[0].wrapping_abs())
                .wrapping_sub(v[1].wrapping_abs());
            v[2] = if v[2] >= 0 { rest } else { rest.wrapping_neg() };
        }
    }

    /// QuantizedOctahedralCoordsToUnitVector without fused multiply-adds.
    pub fn unit_vector(&self, s: i32, t: i32) -> [f32; 3] {
        let mut y = s as f32 * self.scale - 1.0;
        let mut z = t as f32 * self.scale - 1.0;
        let x = 1.0 - y.abs() - z.abs();
        let mut offset = -x;
        if offset < 0.0 {
            offset = 0.0;
        }
        y += if y < 0.0 { offset } else { -offset };
        z += if z < 0.0 { offset } else { -offset };
        let norm = x * x + y * y + z * z;
        if f64::from(norm) < 1e-6 {
            [0.0; 3]
        } else {
            let d = 1.0 / norm.sqrt();
            [x * d, y * d, z * d]
        }
    }

    fn in_diamond(&self, s: i32, t: i32) -> bool {
        s.unsigned_abs().wrapping_add(t.unsigned_abs()) <= self.center as u32
    }

    fn invert_diamond(&self, s: &mut i32, t: &mut i32) {
        let (sign_s, sign_t): (i32, i32) = if *s >= 0 && *t >= 0 {
            (1, 1)
        } else if *s <= 0 && *t <= 0 {
            (-1, -1)
        } else {
            (if *s > 0 { 1 } else { -1 }, if *t > 0 { 1 } else { -1 })
        };
        let corner_s = sign_s.wrapping_mul(self.center) as u32;
        let corner_t = sign_t.wrapping_mul(self.center) as u32;
        let mut us = *s as u32;
        let mut ut = *t as u32;
        us = us.wrapping_add(us).wrapping_sub(corner_s);
        ut = ut.wrapping_add(ut).wrapping_sub(corner_t);
        if sign_s * sign_t >= 0 {
            let temp = us;
            us = ut.wrapping_neg();
            ut = temp.wrapping_neg();
        } else {
            std::mem::swap(&mut us, &mut ut);
        }
        us = us.wrapping_add(corner_s);
        ut = ut.wrapping_add(corner_t);
        *s = (us as i32).wrapping_div(2);
        *t = (ut as i32).wrapping_div(2);
    }

    fn mod_max(&self, x: i32) -> i32 {
        if x > self.center {
            x.wrapping_sub(self.max_quantized)
        } else if x < self.center.wrapping_neg() {
            x.wrapping_add(self.max_quantized)
        } else {
            x
        }
    }

    // PredictionSchemeNormalOctahedronTransformBase::set_max_quantized_value.
    fn from_max_quantized(max_quantized: i32) -> Result<Self> {
        if max_quantized % 2 == 0 {
            return Err("invalid Draco octahedron transform");
        }
        let bits = 32 - (max_quantized as u32).leading_zeros() as i32;
        Self::new(bits).ok_or("invalid Draco octahedron transform")
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Transform {
    /// PredictionSchemeWrapDecodingTransform; `components` is set only by Init.
    Wrap {
        components: usize,
        min: i32,
        max: i32,
        max_dif: i32,
    },
    Octahedron(Octahedron),
    Canonical(Octahedron),
}

impl Transform {
    pub fn wrap() -> Self {
        Transform::Wrap {
            components: 0,
            min: 0,
            max: 0,
            max_dif: 0,
        }
    }

    fn init(&mut self, count: usize) {
        if let Transform::Wrap { components, .. } = self {
            *components = count;
        }
    }

    pub fn corrections_positive(&self) -> bool {
        !matches!(self, Transform::Wrap { .. })
    }

    fn bits(&self) -> i32 {
        match self {
            Transform::Wrap { .. } => -1,
            Transform::Octahedron(tool) | Transform::Canonical(tool) => tool.bits,
        }
    }

    fn decode(&mut self, r: &mut Reader<'_>) -> Result<()> {
        match self {
            Transform::Wrap {
                min, max, max_dif, ..
            } => {
                let (low, high) = (r.i32()?, r.i32()?);
                let dif = i64::from(high) - i64::from(low);
                if low > high || dif >= i64::from(i32::MAX) {
                    return Err("invalid Draco wrap transform");
                }
                (*min, *max, *max_dif) = (low, high, 1 + dif as i32);
            }
            Transform::Octahedron(tool) => *tool = Octahedron::from_max_quantized(r.i32()?)?,
            Transform::Canonical(tool) => {
                let max_quantized = r.i32()?;
                r.i32()?;
                *tool = Octahedron::from_max_quantized(max_quantized)?;
            }
        }
        Ok(())
    }

    /// ComputeOriginalValue(pred, data[at..] as corrections, data[at..] as output).
    fn apply(&self, pred: &[i32], data: &mut [i32], at: usize) -> Result<()> {
        self.apply_to(pred, data.get_mut(at..).ok_or(PREDICTION_ERROR)?)
    }

    /// `apply` predicting from the `components` values before `at`.
    fn apply_prior(&self, data: &mut [i32], at: usize, components: usize) -> Result<()> {
        let (head, tail) = data.split_at_mut(at);
        let pred = &head[at - components..];
        if pred.len() >= 2 {
            return self.apply_to(pred, tail);
        }
        // The octahedron transforms read two components, zero past the prior.
        let mut padded = [0; 2];
        padded[..pred.len()].copy_from_slice(pred);
        self.apply_to(&padded, tail)
    }

    fn apply_to(&self, pred: &[i32], data: &mut [i32]) -> Result<()> {
        match self {
            &Transform::Wrap {
                components,
                min,
                max,
                max_dif,
            } => {
                let out = data.get_mut(..components).ok_or(PREDICTION_ERROR)?;
                for (out, &pred) in out
                    .iter_mut()
                    .zip(pred.get(..components).ok_or(PREDICTION_ERROR)?)
                {
                    let mut value = i64::from(pred.clamp(min, max)) + i64::from(*out);
                    if value > i64::from(max) {
                        value -= i64::from(max_dif);
                    } else if value < i64::from(min) {
                        value += i64::from(max_dif);
                    }
                    *out = value as i32;
                }
            }
            Transform::Octahedron(tool) => {
                let out = data.get_mut(..2).ok_or(PREDICTION_ERROR)?;
                let center = tool.center as u32;
                let mut p = [
                    (pred[0] as u32).wrapping_sub(center) as i32,
                    (pred[1] as u32).wrapping_sub(center) as i32,
                ];
                let in_diamond = tool.in_diamond(p[0], p[1]);
                if !in_diamond {
                    let [s, t] = &mut p;
                    tool.invert_diamond(s, t);
                }
                let mut o = [
                    tool.mod_max((p[0] as u32).wrapping_add(out[0] as u32) as i32),
                    tool.mod_max((p[1] as u32).wrapping_add(out[1] as u32) as i32),
                ];
                if !in_diamond {
                    let [s, t] = &mut o;
                    tool.invert_diamond(s, t);
                }
                out[0] = (o[0] as u32).wrapping_add(center) as i32;
                out[1] = (o[1] as u32).wrapping_add(center) as i32;
            }
            Transform::Canonical(tool) => {
                let out = data.get_mut(..2).ok_or(PREDICTION_ERROR)?;
                let mut p = [
                    pred[0].wrapping_sub(tool.center),
                    pred[1].wrapping_sub(tool.center),
                ];
                let in_diamond = tool.in_diamond(p[0], p[1]);
                if !in_diamond {
                    let [s, t] = &mut p;
                    tool.invert_diamond(s, t);
                }
                let bottom_left = (p[0] == 0 && p[1] == 0) || (p[0] < 0 && p[1] <= 0);
                let rotation = rotation_count(p);
                if !bottom_left {
                    p = rotate(p, rotation);
                }
                let mut o = [
                    tool.mod_max(p[0].wrapping_add(out[0])),
                    tool.mod_max(p[1].wrapping_add(out[1])),
                ];
                if !bottom_left {
                    o = rotate(o, (4 - rotation) % 4);
                }
                if !in_diamond {
                    let [s, t] = &mut o;
                    tool.invert_diamond(s, t);
                }
                out[0] = o[0].wrapping_add(tool.center);
                out[1] = o[1].wrapping_add(tool.center);
            }
        }
        Ok(())
    }
}

fn rotation_count([x, y]: [i32; 2]) -> i32 {
    match (x.signum(), y) {
        (0, 0) => 0,
        (0, y) if y > 0 => 3,
        (0, _) => 1,
        (1, y) if y >= 0 => 2,
        (1, _) => 1,
        (_, y) if y <= 0 => 0,
        _ => 3,
    }
}

fn rotate([x, y]: [i32; 2], count: i32) -> [i32; 2] {
    match count {
        1 => [y, x.wrapping_neg()],
        2 => [x.wrapping_neg(), y.wrapping_neg()],
        3 => [y.wrapping_neg(), x],
        _ => [x, y],
    }
}

/// The parent POSITION attribute's portable values, read through its point map.
pub(crate) struct Positions<'a> {
    pub values: &'a [i32],
    pub components: usize,
    pub map: Option<&'a [u32]>,
}

impl Positions<'_> {
    // ConvertValue stops at the first component outside the buffer, leaving zeros.
    fn get(&self, point: u32) -> [i64; 3] {
        let index = self
            .map
            .map_or(Some(point), |map| map.get(point as usize).copied());
        let mut out = [0; 3];
        if let Some(index) = index {
            for (c, out) in out.iter_mut().enumerate().take(self.components) {
                match (index as usize)
                    .checked_mul(self.components)
                    .and_then(|base| self.values.get(base + c))
                {
                    Some(&value) => *out = i64::from(value),
                    None => break,
                }
            }
        }
        out
    }
}

pub(crate) struct MeshData<'a> {
    pub table: Table<'a>,
    pub encoding: &'a Encoding,
}

impl MeshData<'_> {
    fn value(&self, vertex: usize) -> Result<usize> {
        self.encoding
            .vertex_to_value
            .get(vertex)
            .map(|&value| value as usize)
            .ok_or(PREDICTION_ERROR)
    }
    fn vertex_value(&self, corner: usize) -> Result<usize> {
        self.value(self.table.vertex(corner))
    }
}

pub(crate) struct Scheme<'a> {
    pub method: Method,
    pub transform: Transform,
    creases: [Vec<bool>; 4],
    orientations: Vec<bool>,
    flips: Option<Binary<'a>>,
    max_values: usize,
}

impl<'a> Scheme<'a> {
    pub fn new(method: Method, transform: Transform) -> Self {
        Self {
            method,
            transform,
            creases: Default::default(),
            orientations: Vec::new(),
            flips: None,
            max_values: 0,
        }
    }

    pub fn parents(&self) -> bool {
        matches!(
            self.method,
            Method::TexCoordsDeprecated | Method::TexCoordsPortable | Method::GeometricNormal
        )
    }

    pub fn decode(&mut self, r: &mut Reader<'a>, corners: usize, max_values: usize) -> Result<()> {
        self.max_values = max_values;
        match self.method {
            Method::ConstrainedMultiParallelogram => {
                for creases in &mut self.creases {
                    let count = r.count()?;
                    if count > corners {
                        return Err("invalid Draco crease edge count");
                    }
                    if count > 0 {
                        *creases = filled(count, false)?;
                        let mut bits = Binary::new(r)?;
                        for crease in creases.iter_mut() {
                            *crease = bits.read()?;
                        }
                    }
                }
            }
            Method::TexCoordsPortable | Method::TexCoordsDeprecated => {
                let count = if self.method == Method::TexCoordsPortable {
                    let count = r.i32()?;
                    if count < 0 || count as usize > self.max_values {
                        return Err("invalid Draco texture orientation count");
                    }
                    count as usize
                } else {
                    let count = r.count()?;
                    if count == 0 || count > corners {
                        return Err("invalid Draco texture orientation count");
                    }
                    count
                };
                self.orientations = filled(count, false)?;
                let mut bits = Binary::new(r)?;
                let mut last = true;
                for orientation in &mut self.orientations {
                    if !bits.read()? {
                        last = !last;
                    }
                    *orientation = last;
                }
            }
            Method::GeometricNormal => {
                self.transform.decode(r)?;
                self.flips = Some(Binary::new(r)?);
                return Ok(());
            }
            _ => (),
        }
        self.transform.decode(r)
    }

    /// ComputeOriginalValues in place: `data` holds corrections on entry.
    pub fn compute(
        &mut self,
        data: &mut [i32],
        components: usize,
        points: &[u32],
        mesh: Option<&MeshData<'_>>,
        parent: Option<&Positions<'_>>,
    ) -> Result<()> {
        let Some(mesh) = mesh.filter(|_| self.method != Method::Delta) else {
            return self.delta(data, components);
        };
        let values = mesh.encoding.value_to_corner.len();
        match self.method {
            Method::TexCoordsPortable | Method::TexCoordsDeprecated if components != 2 => {
                return Err(PREDICTION_ERROR);
            }
            _ => (),
        }
        if self.method != Method::GeometricNormal {
            self.transform.init(components);
        }
        // Components fit in a u8, so predictions live on the stack.
        let mut pred = [0i32; 256];
        if matches!(
            self.method,
            Method::Parallelogram
                | Method::MultiParallelogram
                | Method::ConstrainedMultiParallelogram
        ) {
            self.transform.apply(&pred, data, 0)?;
        }
        if values
            .checked_mul(components)
            .is_none_or(|n| n > data.len())
        {
            return Err(PREDICTION_ERROR);
        }
        match self.method {
            Method::Delta => unreachable!(),
            Method::Parallelogram => {
                for p in 1..values {
                    let corner = mesh.encoding.value_to_corner[p];
                    let at = p * components;
                    if parallelogram(p, corner, mesh, data, components, &mut pred)? {
                        self.transform.apply(&pred, data, at)?;
                    } else {
                        self.transform.apply_prior(data, at, components)?;
                    }
                }
            }
            Method::MultiParallelogram => {
                let mut single = [0i32; 256];
                for p in 1..values {
                    let start = mesh.encoding.value_to_corner[p];
                    let mut corner = start;
                    let mut count = 0;
                    pred[..components].fill(0);
                    let mut steps = 0;
                    while corner != INVALID {
                        step(&mut steps, mesh)?;
                        if parallelogram(p, corner, mesh, data, components, &mut single)? {
                            for (pred, single) in pred[..components].iter_mut().zip(&single) {
                                *pred = pred.wrapping_add(*single);
                            }
                            count += 1;
                        }
                        corner = mesh.table.swing_right(corner);
                        if corner == start {
                            corner = INVALID;
                        }
                    }
                    self.average_or_previous(&mut pred, count, data, p, components)?;
                }
            }
            Method::ConstrainedMultiParallelogram => {
                let mut candidates = [[0i32; 256]; 4];
                let mut positions = [0; 4];
                for p in 1..values {
                    let start = mesh.encoding.value_to_corner[p];
                    let mut corner = start;
                    let mut count = 0;
                    let mut first_pass = true;
                    let mut steps = 0;
                    while corner != INVALID {
                        step(&mut steps, mesh)?;
                        if parallelogram(p, corner, mesh, data, components, &mut candidates[count])?
                        {
                            count += 1;
                            if count == 4 {
                                break;
                            }
                        }
                        corner = if first_pass {
                            mesh.table.swing_left(corner)
                        } else {
                            mesh.table.swing_right(corner)
                        };
                        if corner == start {
                            break;
                        }
                        if corner == INVALID && first_pass {
                            first_pass = false;
                            corner = mesh.table.swing_right(start);
                        }
                    }
                    let mut used = 0;
                    if count > 0 {
                        pred[..components].fill(0);
                        for candidate in &candidates[..count] {
                            let context = count - 1;
                            let position = positions[context];
                            positions[context] += 1;
                            if !*self.creases[context]
                                .get(position)
                                .ok_or(PREDICTION_ERROR)?
                            {
                                used += 1;
                                for (pred, value) in pred[..components].iter_mut().zip(candidate) {
                                    *pred = pred.wrapping_add(*value);
                                }
                            }
                        }
                    }
                    self.average_or_previous(&mut pred, used, data, p, components)?;
                }
            }
            Method::TexCoordsPortable | Method::TexCoordsDeprecated => {
                let parent = parent.ok_or(PREDICTION_ERROR)?;
                for p in 0..values {
                    let corner = mesh.encoding.value_to_corner[p];
                    let prediction = if self.method == Method::TexCoordsPortable {
                        self.portable_uv(corner, data, p, mesh, points, parent)?
                    } else {
                        self.deprecated_uv(corner, data, p, mesh, points, parent)?
                    };
                    self.transform.apply(&prediction, data, p * 2)?;
                }
            }
            Method::GeometricNormal => {
                let parent = parent.ok_or(PREDICTION_ERROR)?;
                let mut tool = Octahedron::default();
                tool.set_bits(self.transform.bits());
                let flips = self.flips.as_mut().ok_or(PREDICTION_ERROR)?;
                for p in 0..values {
                    let corner = mesh.encoding.value_to_corner[p];
                    let mut normal = area_normal(corner, mesh, points, parent)?;
                    tool.canonicalize_vector(&mut normal);
                    if flips.read()? {
                        normal = normal.map(i32::wrapping_neg);
                    }
                    let (s, t) = tool.vector_to_coords(normal);
                    self.transform.apply(&[s, t], data, p * 2)?;
                }
            }
        }
        Ok(())
    }

    fn delta(&mut self, data: &mut [i32], components: usize) -> Result<()> {
        self.transform.init(components);
        let zero = [0i32; 256];
        self.transform.apply(&zero, data, 0)?;
        let mut at = components;
        while at < data.len() {
            self.transform.apply_prior(data, at, components)?;
            at += components;
        }
        Ok(())
    }

    fn average_or_previous(
        &self,
        pred: &mut [i32],
        count: i32,
        data: &mut [i32],
        p: usize,
        components: usize,
    ) -> Result<()> {
        let at = p * components;
        if count == 0 {
            self.transform.apply_prior(data, at, components)
        } else {
            for value in &mut pred[..components] {
                *value /= count;
            }
            self.transform.apply(pred, data, at)
        }
    }

    fn pop_orientation(&mut self) -> Result<bool> {
        self.orientations.pop().ok_or(PREDICTION_ERROR)
    }

    fn neighbours(&self, corner: usize, mesh: &MeshData<'_>) -> Result<(usize, usize)> {
        Ok((
            mesh.vertex_value(next(corner))?,
            mesh.vertex_value(previous(corner))?,
        ))
    }

    // Shared fallback when the triangle cannot predict this texture coordinate.
    // The pinned predictor's previous-vertex offset is always overwritten.
    fn previous_uv(&self, data: &[i32], p: usize, next: usize) -> Result<[i32; 2]> {
        let offset = if next < p {
            next * 2
        } else if p > 0 {
            (p - 1) * 2
        } else {
            return Ok([0, 0]);
        };
        Ok([
            *data.get(offset).ok_or(PREDICTION_ERROR)?,
            *data.get(offset + 1).ok_or(PREDICTION_ERROR)?,
        ])
    }

    fn portable_uv(
        &mut self,
        corner: usize,
        data: &[i32],
        p: usize,
        mesh: &MeshData<'_>,
        points: &[u32],
        parent: &Positions<'_>,
    ) -> Result<[i32; 2]> {
        let (n, pv) = self.neighbours(corner, mesh)?;
        if pv < p && n < p {
            let uv = |id: usize| [i64::from(data[id * 2]), i64::from(data[id * 2 + 1])];
            let (n_uv, p_uv) = (uv(n), uv(pv));
            if p_uv == n_uv {
                return Ok([p_uv[0] as i32, p_uv[1] as i32]);
            }
            let position = |id: usize| -> Result<[i64; 3]> {
                Ok(parent.get(*points.get(id).ok_or(PREDICTION_ERROR)?))
            };
            let (tip, next, prev) = (position(p)?, position(n)?, position(pv)?);
            let pn = sub3(prev, next);
            let pn_norm = dot3(pn, pn) as u64;
            if let Some(uv_limit) = (i64::MAX as u64).checked_div(pn_norm) {
                let cn = sub3(tip, next);
                let cn_dot_pn = dot3(pn, cn);
                let pn_uv = [p_uv[0] - n_uv[0], p_uv[1] - n_uv[1]];
                let n_uv_max = n_uv[0].wrapping_abs().max(n_uv[1].wrapping_abs());
                if (n_uv_max as u64) > uv_limit {
                    return Err(PREDICTION_ERROR);
                }
                let pn_uv_max = pn_uv[0].wrapping_abs().max(pn_uv[1].wrapping_abs());
                if cn_dot_pn.wrapping_abs()
                    > i64::MAX.checked_div(pn_uv_max).ok_or(PREDICTION_ERROR)?
                {
                    return Err(PREDICTION_ERROR);
                }
                let x_uv = [0, 1].map(|i| {
                    (n_uv[i] as u64)
                        .wrapping_mul(pn_norm)
                        .wrapping_add((pn_uv[i] as u64).wrapping_mul(cn_dot_pn as u64))
                        as i64
                });
                let pn_max = pn[0]
                    .wrapping_abs()
                    .max(pn[1].wrapping_abs())
                    .max(pn[2].wrapping_abs());
                if cn_dot_pn.wrapping_abs()
                    > i64::MAX.checked_div(pn_max).ok_or(PREDICTION_ERROR)?
                {
                    return Err(PREDICTION_ERROR);
                }
                let norm = pn_norm as i64;
                let x_pos = [0, 1, 2].map(|i| {
                    next[i].wrapping_add(cn_dot_pn.wrapping_mul(pn[i]).wrapping_div(norm))
                });
                let cx_norm = dot3(sub3(tip, x_pos), sub3(tip, x_pos)) as u64;
                let root = int_sqrt(cx_norm.wrapping_mul(pn_norm)) as i64;
                let cx_uv = [
                    pn_uv[1].wrapping_mul(root),
                    pn_uv[0].wrapping_neg().wrapping_mul(root),
                ];
                let orientation = self.pop_orientation()?;
                let uv = [0, 1].map(|i| {
                    let (x, c) = (x_uv[i] as u64, cx_uv[i] as u64);
                    let sum = if orientation {
                        x.wrapping_add(c)
                    } else {
                        x.wrapping_sub(c)
                    };
                    (sum as i64).wrapping_div(norm) as i32
                });
                return Ok(uv);
            }
        }
        self.previous_uv(data, p, n)
    }

    fn deprecated_uv(
        &mut self,
        corner: usize,
        data: &[i32],
        p: usize,
        mesh: &MeshData<'_>,
        points: &[u32],
        parent: &Positions<'_>,
    ) -> Result<[i32; 2]> {
        fn to_int(value: f64) -> i32 {
            if value.is_nan() || value > f64::from(i32::MAX) || value < f64::from(i32::MIN) {
                i32::MIN
            } else {
                value as i32
            }
        }
        let (n, pv) = self.neighbours(corner, mesh)?;
        if pv < p && n < p {
            let uv = |id: usize| [data[id * 2] as f32, data[id * 2 + 1] as f32];
            let (n_uv, p_uv) = (uv(n), uv(pv));
            if p_uv == n_uv {
                return Ok(p_uv.map(|v| to_int(f64::from(v))));
            }
            let position = |id: usize| -> Result<[f32; 3]> {
                Ok(parent
                    .get(*points.get(id).ok_or(PREDICTION_ERROR)?)
                    .map(|v| v as i32 as f32))
            };
            let (tip, next, prev) = (position(p)?, position(n)?, position(pv)?);
            let pn = [0, 1, 2].map(|i| prev[i] - next[i]);
            let cn = [0, 1, 2].map(|i| tip[i] - next[i]);
            let dot = |a: [f32; 3], b: [f32; 3]| {
                let mut sum = 0.0f32;
                for i in 0..3 {
                    sum += a[i] * b[i];
                }
                sum
            };
            let pn_norm = dot(pn, pn);
            let (s, t) = if pn_norm > 0.0 {
                let s = dot(pn, cn) / pn_norm;
                let d = [0, 1, 2].map(|i| cn[i] - pn[i] * s);
                (s, (dot(d, d) / pn_norm).sqrt())
            } else {
                (0.0, 0.0)
            };
            let pn_uv = [p_uv[0] - n_uv[0], p_uv[1] - n_uv[1]];
            let pnus = pn_uv[0] * s + n_uv[0];
            let pnut = pn_uv[0] * t;
            let pnvs = pn_uv[1] * s + n_uv[1];
            let pnvt = pn_uv[1] * t;
            let uv = if self.pop_orientation()? {
                [pnus - pnvt, pnvs + pnut]
            } else {
                [pnus + pnvt, pnvs - pnut]
            };
            return Ok(uv.map(|v| to_int((f64::from(v) + 0.5).floor())));
        }
        self.previous_uv(data, p, n)
    }
}

fn step(steps: &mut usize, mesh: &MeshData<'_>) -> Result<()> {
    // The pinned decoder loops forever on a fan that never returns to its start.
    if *steps > mesh.table.num_corners() {
        return Err("cyclic Draco prediction fan");
    }
    *steps += 1;
    Ok(())
}

/// ComputeParallelogramPrediction.
fn parallelogram(
    entry: usize,
    corner: usize,
    mesh: &MeshData<'_>,
    data: &[i32],
    components: usize,
    out: &mut [i32],
) -> Result<bool> {
    let opposite = mesh.table.opposite(corner);
    if opposite == INVALID {
        return Ok(false);
    }
    let opp = mesh.vertex_value(opposite)?;
    let next = mesh.vertex_value(next(opposite))?;
    let prev = mesh.vertex_value(previous(opposite))?;
    if opp >= entry || next >= entry || prev >= entry {
        return Ok(false);
    }
    for (c, out) in out.iter_mut().enumerate().take(components) {
        let value = |id: usize| i64::from(data[id * components + c]);
        *out = (value(next) + value(prev) - value(opp)) as i32;
    }
    Ok(true)
}

/// MeshPredictionSchemeGeometricNormalPredictorArea::ComputePredictedValue.
fn area_normal(
    corner: usize,
    mesh: &MeshData<'_>,
    points: &[u32],
    parent: &Positions<'_>,
) -> Result<[i32; 3]> {
    let position = |c: usize| -> Result<[i64; 3]> {
        let id = mesh.vertex_value(c)?;
        Ok(parent.get(*points.get(id).ok_or(PREDICTION_ERROR)?))
    };
    let center = position(corner)?;
    let mut normal = [0u64; 3];
    let mut c = corner;
    let mut left = true;
    let mut steps = 0;
    while c != INVALID {
        step(&mut steps, mesh)?;
        let d_next = sub3(position(next(c))?, center).map(|v| v as u64);
        let d_prev = sub3(position(previous(c))?, center).map(|v| v as u64);
        let cross = [
            d_next[1]
                .wrapping_mul(d_prev[2])
                .wrapping_sub(d_next[2].wrapping_mul(d_prev[1])),
            d_next[2]
                .wrapping_mul(d_prev[0])
                .wrapping_sub(d_next[0].wrapping_mul(d_prev[2])),
            d_next[0]
                .wrapping_mul(d_prev[1])
                .wrapping_sub(d_next[1].wrapping_mul(d_prev[0])),
        ];
        for (normal, cross) in normal.iter_mut().zip(cross) {
            *normal = normal.wrapping_add(cross);
        }
        // VertexCornersIterator: swing left, then right from the start.
        if left {
            c = mesh.table.swing_left(c);
            if c == INVALID {
                c = mesh.table.swing_right(corner);
                left = false;
            } else if c == corner {
                c = INVALID;
            }
        } else {
            c = mesh.table.swing_right(c);
        }
    }
    let mut normal = normal.map(|v| v as i64);
    let mut sum = 0i64;
    for value in normal {
        let value = value.wrapping_abs();
        if sum > i64::MAX.wrapping_sub(value) {
            sum = i64::MAX;
            break;
        }
        sum += value;
    }
    const UPPER: i64 = 1 << 29;
    if sum > UPPER {
        let quotient = sum / UPPER;
        normal = normal.map(|v| v / quotient);
    }
    Ok(normal.map(|v| v as i32))
}

fn sub3(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [0, 1, 2].map(|i| a[i].wrapping_sub(b[i]))
}

fn dot3(a: [i64; 3], b: [i64; 3]) -> i64 {
    (0..3).fold(0i64, |sum, i| sum.wrapping_add(a[i].wrapping_mul(b[i])))
}

fn int_sqrt(number: u64) -> u64 {
    if number == 0 {
        return 0;
    }
    let mut act = number;
    let mut root = 1u64;
    while act >= 2 {
        root = root.wrapping_mul(2);
        act /= 4;
    }
    loop {
        root = root.wrapping_add(number / root) / 2;
        if root.wrapping_mul(root) <= number {
            return root;
        }
    }
}
