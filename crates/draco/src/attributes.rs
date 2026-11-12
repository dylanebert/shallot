// Port of Draco 55462882: point_cloud_decoder.cc's DecodePointAttributes,
// attributes_decoder.cc, sequential_attribute_decoders_controller.cc,
// sequential_{,integer_,quantization_,normal_}attribute_decoder.cc,
// attribute_{quantization,octahedron}_transform.cc, the attribute decoder
// selection in mesh_edgebreaker_decoder_impl.cc and prediction_scheme_decoder_factory.h.
use crate::{
    allocation::{filled, push},
    buffer::Reader,
    edgebreaker::Connectivity,
    entropy,
    prediction::{MeshData, Method, Octahedron, Positions, Scheme, Transform},
    traversal::{sequence, AttributeCorners, Corners, Encoding, Order},
    Result,
};

const ATTRIBUTE_ERROR: &str = "invalid Draco attribute data";

/// Draco's `DataType`, numbered as in the bitstream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Int8 = 1,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float32,
    Float64,
    Bool,
}

impl DataType {
    pub fn from_id(id: u8) -> Option<Self> {
        use DataType::*;
        [
            Int8, Uint8, Int16, Uint16, Int32, Uint32, Int64, Uint64, Float32, Float64, Bool,
        ]
        .get(usize::from(id).checked_sub(1)?)
        .copied()
    }

    pub fn size(self) -> usize {
        use DataType::*;
        match self {
            Int8 | Uint8 | Bool => 1,
            Int16 | Uint16 => 2,
            Int32 | Uint32 | Float32 => 4,
            Int64 | Uint64 | Float64 => 8,
        }
    }

    fn integer(self) -> Option<(u32, bool)> {
        use DataType::*;
        Some(match self {
            Int8 => (8, true),
            Uint8 => (8, false),
            Int16 => (16, true),
            Uint16 => (16, false),
            Int32 => (32, true),
            Uint32 => (32, false),
            Int64 => (64, true),
            Uint64 => (64, false),
            Float32 | Float64 | Bool => return None,
        })
    }
}

/// Draco's named `GeometryAttribute::Type`, numbered as in the bitstream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeType {
    Position,
    Normal,
    Color,
    TexCoord,
    Generic,
}

/// A decoded attribute: values in its Draco data type and a point-to-value map.
pub struct Attribute {
    pub kind: AttributeType,
    pub data_type: DataType,
    pub components: u8,
    pub normalized: bool,
    pub unique_id: u32,
    values: Vec<u8>,
    map: Option<Vec<u32>>,
    points: usize,
}

impl Attribute {
    fn stride(&self) -> usize {
        self.data_type.size() * usize::from(self.components)
    }

    fn index(&self, point: usize) -> usize {
        match &self.map {
            None => point,
            Some(map) => map[point] as usize,
        }
    }

    /// Values for every point in `data_type`, little-endian, as the pinned
    /// `Decoder::GetAttributeDataArrayForAllPoints` gives them.
    pub fn points(&self, data_type: DataType) -> Result<Vec<u8>> {
        use DataType::*;
        // The pinned wrapper takes only these, even when the attribute matches.
        if !matches!(
            data_type,
            Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Float32
        ) {
            return Err("unsupported Draco output data type: 64-bit and bool are refused");
        }
        let stride = self.stride();
        let components = usize::from(self.components);
        let size = data_type.size();
        let mut out = Vec::new();
        out.try_reserve_exact(
            self.points
                .checked_mul(components * size)
                .ok_or("Draco allocation failed")?,
        )
        .map_err(|_| "Draco allocation failed")?;
        if data_type == self.data_type {
            for point in 0..self.points {
                let start = self
                    .index(point)
                    .checked_mul(stride)
                    .ok_or(ATTRIBUTE_ERROR)?;
                out.extend_from_slice(
                    self.values
                        .get(start..start + stride)
                        .ok_or("Draco point has no attribute value")?,
                );
            }
            return Ok(out);
        }
        let width = self.data_type.size();
        for point in 0..self.points {
            let start = self.index(point).saturating_mul(stride);
            for c in 0..components {
                let at = start.saturating_add(c * width);
                let bytes = self
                    .values
                    .get(at..at.saturating_add(width))
                    .ok_or("Draco point has no attribute value")?;
                convert(bytes, self.data_type, data_type, self.normalized, &mut out)
                    .ok_or("Draco attribute value does not fit the requested type")?;
            }
        }
        Ok(out)
    }
}

#[derive(Clone, Copy)]
enum Value {
    Integer(i128),
    Float(f64, bool),
    Bool(u8),
}

fn read(bytes: &[u8], data_type: DataType) -> Value {
    let mut wide = [0u8; 8];
    wide[..bytes.len()].copy_from_slice(bytes);
    let raw = u64::from_le_bytes(wide);
    match data_type {
        DataType::Float32 => Value::Float(f64::from(f32::from_bits(raw as u32)), true),
        DataType::Float64 => Value::Float(f64::from_bits(raw), false),
        // A C++ bool load of a byte other than 0 or 1 is undefined; the pinned
        // build converts the byte it loads.
        DataType::Bool => Value::Bool(bytes[0]),
        _ => {
            let (bits, signed) = data_type.integer().unwrap();
            Value::Integer(wrap(i128::from(raw), bits, signed))
        }
    }
}

fn wrap(value: i128, bits: u32, signed: bool) -> i128 {
    let modulus = 1i128 << bits;
    let value = value.rem_euclid(modulus);
    if signed && value >= modulus / 2 {
        value - modulus
    } else {
        value
    }
}

fn integer_limits(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

// C++ usual arithmetic conversions for `a < b` with integer operands.
fn less(a: i128, a_type: (u32, bool), b: i128, b_type: (u32, bool)) -> bool {
    let promote = |(bits, signed): (u32, bool)| {
        if bits < 32 {
            (32, true)
        } else {
            (bits, signed)
        }
    };
    let (a_type, b_type) = (promote(a_type), promote(b_type));
    let common = if a_type.1 == b_type.1 {
        (a_type.0.max(b_type.0), a_type.1)
    } else {
        let (unsigned, signed) = if a_type.1 {
            (b_type, a_type)
        } else {
            (a_type, b_type)
        };
        if unsigned.0 >= signed.0 {
            unsigned
        } else {
            signed
        }
    };
    wrap(a, common.0, common.1) < wrap(b, common.0, common.1)
}

/// GeometryAttribute::ConvertComponentValue for one component.
fn convert(
    bytes: &[u8],
    from: DataType,
    to: DataType,
    normalized: bool,
    out: &mut Vec<u8>,
) -> Option<()> {
    let value = read(bytes, from);
    let target = to.integer();
    if let Some((bits, signed)) = target {
        let (min, max) = integer_limits(bits, signed);
        match value {
            Value::Integer(v) => {
                let from_type = from.integer().unwrap();
                let out_min = if from_type.1 { min } else { 0 };
                if less(v, from_type, out_min, (bits, signed))
                    || less(max, (bits, signed), v, from_type)
                {
                    return None;
                }
            }
            Value::Float(v, single) => {
                if !v.is_finite() {
                    return None;
                }
                let (low, high) = if single {
                    (f64::from(min as f32), f64::from(max as f32))
                } else {
                    (min as f64, max as f64)
                };
                if v < low || v >= high {
                    return None;
                }
            }
            Value::Bool(_) => (),
        }
    }
    let write_integer = |out: &mut Vec<u8>, v: i128| {
        let (bits, _) = target.unwrap();
        out.extend_from_slice(&(v as u64).to_le_bytes()[..bits as usize / 8]);
    };
    match (value, target) {
        (Value::Float(v, _), Some(_)) if normalized => {
            if !(0.0..=1.0).contains(&v) {
                return None;
            }
            let (_, max) = integer_limits(target.unwrap().0, target.unwrap().1);
            write_integer(out, (v * max as f64 + 0.5).floor() as i128);
        }
        (Value::Float(v, _), Some(_)) => write_integer(out, v as i128),
        (Value::Integer(v), Some(_)) => write_integer(out, v),
        (Value::Bool(v), Some(_)) => write_integer(out, i128::from(v)),
        (value, None) => {
            let mut float = match value {
                Value::Integer(v) => v as f32,
                Value::Float(v, _) => v as f32,
                Value::Bool(v) => f32::from(v),
            };
            if normalized {
                match value {
                    Value::Integer(_) => {
                        let (bits, signed) = from.integer().unwrap();
                        float /= integer_limits(bits, signed).1 as f32;
                    }
                    Value::Bool(_) => float /= 1.0,
                    Value::Float(..) => (),
                }
            }
            out.extend_from_slice(&float.to_le_bytes());
        }
    }
    Some(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Generic,
    Integer,
    Quantization,
    Normals,
}

#[derive(Clone, Copy)]
enum Sequencer {
    Linear,
    /// Traverses the base table, or `seams[data]`'s table for corner attributes.
    Traversal {
        order: Order,
        corner: bool,
        data: Option<usize>,
    },
}

struct Decoder {
    sequencer: Sequencer,
    attributes: Vec<usize>,
    kinds: Vec<Kind>,
}

struct Quantization {
    min: Vec<f32>,
    range: f32,
    bits: u32,
}

#[derive(Default)]
struct State {
    portable: Option<Vec<i32>>,
    components: usize,
    quantization: Option<Quantization>,
    octahedron_bits: u8,
}

/// Decodes the attribute decoders following connectivity.
pub(crate) fn decode(
    r: &mut Reader<'_>,
    faces: &[[u32; 3]],
    points: usize,
    connectivity: Option<&Connectivity>,
    requested: &[u32],
) -> Result<Vec<Attribute>> {
    if points > i32::MAX as usize {
        return Err("invalid Draco point count");
    }
    let count = r.byte()?;
    let mut decoders = Vec::new();
    let mut data_used = Vec::new();
    let mut position_used = false;
    if let Some(connectivity) = connectivity {
        data_used = filled(connectivity.seams.len(), false)?;
    }
    for _ in 0..count {
        let sequencer = if connectivity.is_some() {
            let data = r.byte()? as i8;
            let element = r.byte()?;
            let data = if data >= 0 {
                let used = data_used
                    .get_mut(data as usize)
                    .ok_or("unexpected Draco attribute data")?;
                if *used {
                    return Err("Draco attribute data already has a decoder");
                }
                *used = true;
                Some(data as usize)
            } else {
                if position_used {
                    return Err("Draco position data already has a decoder");
                }
                position_used = true;
                None
            };
            let order = match r.byte()? {
                0 => Order::DepthFirst,
                1 => Order::PredictionDegree,
                _ => return Err("unsupported Draco attribute traversal"),
            };
            // Every element type other than MESH_VERTEX_ATTRIBUTE is per corner.
            let corner = element != 0;
            if corner && (order != Order::DepthFirst || data.is_none()) {
                return Err("unsupported Draco corner attribute traversal");
            }
            Sequencer::Traversal {
                order,
                corner,
                data,
            }
        } else {
            Sequencer::Linear
        };
        push(
            &mut decoders,
            Decoder {
                sequencer,
                attributes: Vec::new(),
                kinds: Vec::new(),
            },
        )?;
    }
    let mut attributes: Vec<Attribute> = Vec::new();
    for decoder in &mut decoders {
        let count = r.count()?;
        if count == 0 || count > r.remaining().saturating_mul(5) {
            return Err("invalid Draco attribute count");
        }
        for _ in 0..count {
            let kind = r.byte()?;
            let data_type = r.byte()?;
            let components = r.byte()?;
            let normalized = r.byte()? > 0;
            let kind = [
                AttributeType::Position,
                AttributeType::Normal,
                AttributeType::Color,
                AttributeType::TexCoord,
                AttributeType::Generic,
            ]
            .get(usize::from(kind))
            .copied()
            .ok_or("invalid Draco attribute type")?;
            let data_type = DataType::from_id(data_type).ok_or("invalid Draco data type")?;
            if components == 0 {
                return Err("invalid Draco attribute components");
            }
            let unique_id = r.varint(32)? as u32;
            push(&mut decoder.attributes, attributes.len())?;
            push(
                &mut attributes,
                Attribute {
                    kind,
                    data_type,
                    components,
                    normalized,
                    unique_id,
                    values: Vec::new(),
                    map: None,
                    points,
                },
            )?;
        }
        for &id in &decoder.attributes {
            let attribute = &attributes[id];
            let kind = match r.byte()? {
                0 => Kind::Generic,
                1 => Kind::Integer,
                2 if attribute.data_type == DataType::Float32 => Kind::Quantization,
                3 if attribute.data_type == DataType::Float32 && attribute.components == 3 => {
                    Kind::Normals
                }
                _ => return Err("unsupported Draco attribute decoder"),
            };
            push(&mut decoder.kinds, kind)?;
        }
    }
    let mut states = Vec::new();
    for _ in 0..attributes.len() {
        push(&mut states, State::default())?;
    }
    // Only requested attributes keep values. The first POSITION is the only
    // attribute a prediction reads as a parent, so it keeps its portable
    // values and map too; every other attribute is dropped once decoded, so
    // peak memory follows the request rather than the stream's length.
    let mut wanted = filled(attributes.len(), false)?;
    for (wanted, attribute) in wanted.iter_mut().zip(&attributes) {
        *wanted = requested.contains(&attribute.unique_id);
    }
    let parent = attributes
        .iter()
        .position(|a| a.kind == AttributeType::Position);
    let keep = |id: usize| wanted[id] || Some(id) == parent;
    for decoder in &decoders {
        let base = connectivity.map(|connectivity| &connectivity.table);
        let seams = match decoder.sequencer {
            Sequencer::Traversal {
                data: Some(data), ..
            } => Some(AttributeCorners {
                base: base.unwrap(),
                seams: &connectivity.unwrap().seams[data],
            }),
            _ => None,
        };
        let mut encoding = None;
        let mut map = None;
        let point_ids = match decoder.sequencer {
            Sequencer::Linear => {
                let mut ids = Vec::new();
                ids.try_reserve_exact(points)
                    .map_err(|_| "Draco allocation failed")?;
                ids.extend(0..points as u32);
                ids
            }
            Sequencer::Traversal { order, corner, .. } => {
                let base = base.unwrap();
                let table: &dyn Corners = match &seams {
                    Some(seams) if corner => seams,
                    _ => base,
                };
                // Value maps are sized for whichever table the decoder may use.
                let vertices = seams
                    .as_ref()
                    .map_or(0, |seams| seams.num_vertices())
                    .max(base.num_vertices());
                let mut values = Encoding::new(vertices)?;
                let ids = sequence(table, order, faces, &mut values)?;
                let mut points_to_values = filled(points, u32::MAX)?;
                for (corner, &point) in faces.iter().flatten().enumerate() {
                    let value = *values
                        .vertex_to_value
                        .get(table.vertex(corner))
                        .ok_or(ATTRIBUTE_ERROR)? as u32;
                    if point as usize >= points || value as usize >= points {
                        return Err(ATTRIBUTE_ERROR);
                    }
                    points_to_values[point as usize] = value;
                }
                encoding = Some(values);
                map = Some(points_to_values);
                ids
            }
        };
        for &id in decoder.attributes.iter().filter(|&&id| keep(id)) {
            attributes[id].map = match &map {
                None => None,
                Some(map) => {
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(map.len())
                        .map_err(|_| "Draco allocation failed")?;
                    copy.extend_from_slice(map);
                    Some(copy)
                }
            };
        }
        let mesh = match (decoder.sequencer, &encoding) {
            (Sequencer::Traversal { corner, .. }, Some(encoding)) => Some(MeshData {
                table: match &seams {
                    Some(seams) if corner => seams,
                    _ => base.unwrap(),
                },
                encoding,
            }),
            _ => None,
        };
        let corners = faces.len() * 3;
        for (&id, &kind) in decoder.attributes.iter().zip(&decoder.kinds) {
            let entries = point_ids.len();
            let stride = attributes[id].stride();
            let size = entries.checked_mul(stride).ok_or(ATTRIBUTE_ERROR)?;
            let mut values = filled(if wanted[id] { size } else { 0 }, 0u8)?;
            if kind == Kind::Generic {
                let bytes = r.take(size)?;
                if wanted[id] {
                    values.copy_from_slice(bytes);
                }
            } else {
                let portable = integer_values(
                    r,
                    &attributes,
                    &states,
                    id,
                    kind,
                    &point_ids,
                    mesh.as_ref(),
                    corners,
                )?;
                states[id].components = if kind == Kind::Normals {
                    2
                } else {
                    usize::from(attributes[id].components)
                };
                if keep(id) {
                    states[id].portable = Some(portable);
                }
            }
            attributes[id].values = values;
        }
        for (&id, &kind) in decoder.attributes.iter().zip(&decoder.kinds) {
            match kind {
                Kind::Quantization => {
                    let components = usize::from(attributes[id].components);
                    let mut min = Vec::new();
                    for _ in 0..components {
                        push(&mut min, r.f32()?)?;
                    }
                    let range = r.f32()?;
                    let bits = u32::from(r.byte()?);
                    if !(1..=30).contains(&bits) {
                        return Err("invalid Draco quantization bits");
                    }
                    states[id].quantization = Some(Quantization { min, range, bits });
                }
                Kind::Normals => states[id].octahedron_bits = r.byte()?,
                Kind::Generic | Kind::Integer => (),
            }
        }
        for (&id, &kind) in decoder.attributes.iter().zip(&decoder.kinds) {
            let state = &states[id];
            let attribute = &mut attributes[id];
            let portable = state.portable.as_deref().filter(|_| wanted[id]);
            if portable.is_none() {
                // Unrequested: apply only the refusals the transform would make.
                match kind {
                    Kind::Integer if !matches!(attribute.data_type.integer(), Some((bits, _)) if bits <= 32) =>
                    {
                        return Err("unsupported Draco integer attribute type");
                    }
                    Kind::Normals
                        if Octahedron::new(i32::from(state.octahedron_bits)).is_none() =>
                    {
                        return Err("invalid Draco normal quantization bits");
                    }
                    _ => (),
                }
                continue;
            }
            match kind {
                Kind::Generic => (),
                Kind::Integer => {
                    let size = attribute.data_type.size();
                    if !matches!(
                        attribute.data_type,
                        DataType::Int8
                            | DataType::Uint8
                            | DataType::Int16
                            | DataType::Uint16
                            | DataType::Int32
                            | DataType::Uint32
                    ) {
                        return Err("unsupported Draco integer attribute type");
                    }
                    for (out, value) in attribute
                        .values
                        .chunks_exact_mut(size)
                        .zip(portable.unwrap())
                    {
                        out.copy_from_slice(&value.to_le_bytes()[..size]);
                    }
                }
                Kind::Quantization => {
                    let quantization = state.quantization.as_ref().unwrap();
                    let max = ((1u32 << quantization.bits) - 1) as i32;
                    let delta = quantization.range / max as f32;
                    let components = usize::from(attribute.components);
                    for (i, (out, value)) in attribute
                        .values
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(portable.unwrap())
                        .enumerate()
                    {
                        let value = *value as f32 * delta;
                        let value = value + quantization.min[i % components];
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                }
                Kind::Normals => {
                    let tool = Octahedron::new(i32::from(state.octahedron_bits))
                        .ok_or("invalid Draco normal quantization bits")?;
                    for (out, st) in attribute
                        .values
                        .as_chunks_mut::<12>()
                        .0
                        .iter_mut()
                        .zip(portable.unwrap().as_chunks::<2>().0)
                    {
                        let vector = tool.unit_vector(st[0], st[1]);
                        for (out, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(vector) {
                            out.copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    for &id in requested {
        if !attributes.iter().any(|a| a.unique_id == id) {
            return Err("requested Draco attribute unique id is not in the stream");
        }
    }
    let mut out = Vec::new();
    for (id, attribute) in attributes.into_iter().enumerate() {
        if wanted[id] {
            push(&mut out, attribute)?;
        }
    }
    Ok(out)
}

/// SequentialIntegerAttributeDecoder::DecodeValues: the attribute's portable values.
#[allow(clippy::too_many_arguments)]
fn integer_values<'a>(
    r: &mut Reader<'a>,
    attributes: &[Attribute],
    states: &[State],
    id: usize,
    kind: Kind,
    point_ids: &[u32],
    mesh: Option<&MeshData<'_>>,
    corners: usize,
) -> Result<Vec<i32>> {
    let method = r.byte()? as i8;
    if !(-2..7).contains(&method) {
        return Err("invalid Draco prediction scheme");
    }
    let mut scheme = None;
    if method != -2 {
        let transform = r.byte()? as i8;
        if !(-1..4).contains(&transform) {
            return Err("invalid Draco prediction transform");
        }
        scheme = create_scheme(method, transform, kind, mesh.is_some());
    }
    let mut parent = None;
    if scheme.as_ref().is_some_and(Scheme::parents) {
        let position = attributes
            .iter()
            .position(|a| a.kind == AttributeType::Position)
            .ok_or("Draco prediction requires a position attribute")?;
        let values = states[position]
            .portable
            .as_deref()
            .ok_or("Draco prediction parent is not decoded")?;
        if states[position].components != 3 {
            return Err("Draco prediction parent must have three components");
        }
        parent = Some(Positions {
            values,
            components: 3,
            map: attributes[position].map.as_deref(),
        });
    }
    let components = if kind == Kind::Normals {
        2
    } else {
        usize::from(attributes[id].components)
    };
    let entries = point_ids.len();
    let count = entries.checked_mul(components).ok_or(ATTRIBUTE_ERROR)?;
    if count > u32::MAX as usize {
        return Err(ATTRIBUTE_ERROR);
    }
    let mut values = filled(count, 0i32)?;
    if entries == 0 {
        return Err("Draco attribute has no values");
    }
    if r.byte()? > 0 {
        let symbols = entropy::symbols(r, count, components)?;
        for (value, symbol) in values.iter_mut().zip(symbols) {
            *value = symbol as i32;
        }
    } else {
        let width = usize::from(r.byte()?);
        if width > 4 || r.remaining() < width * count {
            return Err("invalid Draco raw attribute width");
        }
        for value in &mut values {
            let mut bytes = [0; 4];
            bytes[..width].copy_from_slice(r.take(width)?);
            *value = i32::from_le_bytes(bytes);
        }
    }
    if scheme
        .as_ref()
        .is_none_or(|s| !s.transform.corrections_positive())
    {
        for value in &mut values {
            let symbol = *value as u32;
            *value = if symbol & 1 == 0 {
                (symbol >> 1) as i32
            } else {
                -((symbol >> 1) as i32) - 1
            };
        }
    }
    if let Some(scheme) = &mut scheme {
        scheme.decode(r, corners, count)?;
        scheme.compute(&mut values, components, point_ids, mesh, parent.as_ref())?;
    }
    Ok(values)
}

/// CreateIntPredictionScheme and CreatePredictionSchemeForDecoder.
fn create_scheme<'a>(method: i8, transform: i8, kind: Kind, mesh: bool) -> Option<Scheme<'a>> {
    let transform = match (kind, transform) {
        (Kind::Normals, 2) => Transform::Octahedron(Octahedron::default()),
        (Kind::Normals, 3) => Transform::Canonical(Octahedron::default()),
        (Kind::Normals, _) => return None,
        (_, 1) => Transform::wrap(),
        _ => return None,
    };
    let wrap = matches!(transform, Transform::Wrap { .. });
    // Methods the factory does not build for this transform fall back to delta.
    let method = match (mesh, method, wrap) {
        (true, 6, _) => Method::GeometricNormal,
        (true, 1, true) => Method::Parallelogram,
        (true, 2, true) => Method::MultiParallelogram,
        (true, 3, true) => Method::TexCoordsDeprecated,
        (true, 4, true) => Method::ConstrainedMultiParallelogram,
        (true, 5, true) => Method::TexCoordsPortable,
        _ => Method::Delta,
    };
    Some(Scheme::new(method, transform))
}
