use bevy_mikktspace::{Geometry, TangentSpace};

struct Mesh<'a> {
    vertices: &'a [u32],
    uv: Option<&'a [u32]>,
    indices: &'a [u32],
    tangents: Vec<[f32; 4]>,
}

impl Mesh<'_> {
    fn index(&self, face: usize, corner: usize) -> usize {
        self.indices[face * 3 + corner] as usize
    }
}

impl Geometry for Mesh<'_> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }
    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }
    fn position(&self, face: usize, corner: usize) -> [f32; 3] {
        let i = self.index(face, corner) * 8;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn normal(&self, face: usize, corner: usize) -> [f32; 3] {
        let i = self.index(face, corner) * 8 + 4;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn tex_coord(&self, face: usize, corner: usize) -> [f32; 2] {
        let vertex = self.index(face, corner);
        if let Some(uv) = self.uv {
            std::array::from_fn(|j| f32::from_bits(uv[vertex * 2 + j]))
        } else {
            [3, 7].map(|lane| f32::from_bits(self.vertices[vertex * 8 + lane]))
        }
    }
    fn set_tangent(&mut self, tangent: Option<TangentSpace>, face: usize, corner: usize) {
        let mut tangent = tangent.unwrap_or_default().tangent_encoded();
        // Bevy's generate_tangents_for_mesh and glTF Sample Renderer negate MikkTSpace's w.
        tangent[3] = -tangent[3];
        self.tangents[face * 3 + corner] = tangent;
    }
}

fn boxed_words(words: Vec<u32>) -> *mut u32 {
    Box::into_raw(words.into_boxed_slice()) as *mut u32
}

#[export_name = "allocWords"]
pub extern "C" fn alloc_words(words: usize) -> *mut u32 {
    boxed_words(vec![0; words])
}

/// # Safety
/// `ptr` must be a live allocation of `words` u32s returned by this module, freed once.
#[export_name = "freeWords"]
pub unsafe extern "C" fn free_words(ptr: *mut u32, words: usize) {
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, words)) });
}

/// # Safety
/// Input is [vertex count, corner count, stream count, UV stream index, stream widths...,
/// streams..., indices...], all u32 words. Stream zero is eight f32s per vertex; UV index
/// zero selects its UV lanes (3 and 7), otherwise a two-f32 stream. Indices must be in range.
/// Returns [allocation length, vertex count, streams..., indices..., four-f32 tangents...].
/// A null result means MikkTSpace refused the geometry. The caller frees input and output.
#[no_mangle]
pub unsafe extern "C" fn generate(input: *const u32, words: usize) -> *mut u32 {
    let input = unsafe { std::slice::from_raw_parts(input, words) };
    let (nv, nc, ns, uv) = (
        input[0] as usize,
        input[1] as usize,
        input[2] as usize,
        input[3] as usize,
    );
    let widths = &input[4..4 + ns];
    let mut offset = 4 + ns;
    let streams: Vec<_> = widths
        .iter()
        .map(|&width| {
            let end = offset + nv * width as usize;
            let stream = &input[offset..end];
            offset = end;
            stream
        })
        .collect();
    let indices = &input[offset..offset + nc];
    let mut mesh = Mesh {
        vertices: streams[0],
        uv: (uv != 0).then(|| streams[uv]),
        indices,
        tangents: vec![[0.; 4]; nc],
    };
    if bevy_mikktspace::generate_tangents(&mut mesh).is_err() {
        return std::ptr::null_mut();
    }

    // MikkTSpace defines corner results. Keep each original vertex for its first tangent,
    // and append only distinct bit patterns, never overwrite a previous corner's tangent.
    let mut variants: Vec<Vec<([u32; 4], u32)>> = vec![Vec::new(); nv];
    let mut source: Vec<usize> = (0..nv).collect();
    let mut tangents = vec![[0; 4]; nv];
    let mut remapped = Vec::with_capacity(nc);
    for (&vertex, tangent) in indices.iter().zip(mesh.tangents) {
        let vertex = vertex as usize;
        let bits = tangent.map(f32::to_bits);
        let group = &mut variants[vertex];
        let index = if let Some((_, index)) = group.iter().find(|(value, _)| *value == bits) {
            *index
        } else {
            let index = if group.is_empty() {
                vertex as u32
            } else {
                source.push(vertex);
                tangents.push([0; 4]);
                (source.len() - 1) as u32
            };
            tangents[index as usize] = bits;
            group.push((bits, index));
            index
        };
        remapped.push(index);
    }
    let count = source.len();
    let size = 2 + count * (widths.iter().map(|&w| w as usize).sum::<usize>() + 4) + nc;
    let mut output = Vec::with_capacity(size);
    output.extend([size as u32, count as u32]);
    for (stream, &width) in streams.iter().zip(widths) {
        let width = width as usize;
        for &vertex in &source {
            output.extend_from_slice(&stream[vertex * width..(vertex + 1) * width]);
        }
    }
    output.extend(remapped);
    output.extend(tangents.into_iter().flatten());
    boxed_words(output)
}
