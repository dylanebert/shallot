use bevy_mikktspace::{Geometry, TangentSpace};
use std::{hint::black_box, time::Instant};
mod tangent;
struct Input {
    vertices: Vec<u32>,
    indices: Vec<u32>,
    output: Vec<[f32; 4]>,
}
impl Geometry for Input {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }
    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }
    fn position(&self, f: usize, c: usize) -> [f32; 3] {
        let i = self.indices[f * 3 + c] as usize * 8;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn normal(&self, f: usize, c: usize) -> [f32; 3] {
        let i = self.indices[f * 3 + c] as usize * 8 + 4;
        std::array::from_fn(|j| f32::from_bits(self.vertices[i + j]))
    }
    fn tex_coord(&self, f: usize, c: usize) -> [f32; 2] {
        let i = self.indices[f * 3 + c] as usize * 8;
        [
            f32::from_bits(self.vertices[i + 3]),
            f32::from_bits(self.vertices[i + 7]),
        ]
    }
    fn set_tangent(&mut self, t: Option<TangentSpace>, f: usize, c: usize) {
        let mut t = t.unwrap_or_default().tangent_encoded();
        t[3] = -t[3];
        self.output[f * 3 + c] = t;
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = words[0] as usize;
    let c = words[1] as usize;
    let mut input = Input {
        vertices: words[2..2 + n * 8].to_vec(),
        indices: words[2 + n * 8..].to_vec(),
        output: Vec::new(),
    };
    let iterations: usize = args[3].parse().unwrap();
    let mut run = || {
        if args[2] == "port" {
            black_box(
                tangent::generate(&tangent::Mesh {
                    vertices: &input.vertices,
                    uv: None,
                    indices: &input.indices,
                })
                .unwrap(),
            );
        } else {
            input.output = vec![[0.0; 4]; c];
            bevy_mikktspace::generate_tangents(&mut input).unwrap();
            black_box(&input.output);
            input.output = Vec::new();
        }
    };
    for _ in 0..5 {
        run();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    println!(
        "{}",
        start.elapsed().as_secs_f64() * 1e6 / iterations as f64
    );
}
