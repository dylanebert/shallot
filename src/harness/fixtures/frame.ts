/** The fixed pure-frustum scene used by the S1 boundary witness. */
export const FRUSTUM_FIXTURE = {
    fov: 90,
    aspect: 1,
    near: 1,
    far: 10,
    boundary: [-5, 0, -5] as const,
    outside: [-5.01, 0, -5] as const,
    radius: 0,
};

/** The fixed GPU render target and shader output used by the S1 probe witness. */
export const GPU_FIXTURE = {
    width: 2,
    height: 2,
    format: "rgba8unorm" as GPUTextureFormat,
    pixel: [224, 64, 32, 255] as const,
};

export const GPU_SHADER = /* wgsl */ `
struct VertexOut { @builtin(position) position: vec4f };

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var output: VertexOut;
    output.position = vec4f(positions[index], 0.0, 1.0);
    return output;
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(${(GPU_FIXTURE.pixel[0] / 255).toFixed(8)}, ${(GPU_FIXTURE.pixel[1] / 255).toFixed(8)}, ${(GPU_FIXTURE.pixel[2] / 255).toFixed(8)}, 1.0);
}
`;
