export const SURFACE_DATA_STRUCT_WGSL = /* wgsl */ `
struct SurfaceData {
    worldPos: vec3<f32>,
    objectPos: vec3<f32>,
    worldNormal: vec3<f32>,
    objectNormal: vec3<f32>,
    baseColor: vec3<f32>,
    emission: vec3<f32>,
    uv: vec2<f32>,
    roughness: f32,
    reflectivity: f32,
    opacity: f32,
}`;

export const SCENE_STRUCT_WGSL = /* wgsl */ `
struct Scene {
    viewProj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    cameraWorld: mat4x4<f32>,
    ambientColor: vec4<f32>,
    sunDirection: vec4<f32>,
    sunColor: vec4<f32>,
    clearColor: vec4<f32>,
    cameraMode: f32,
    cameraSize: f32,
    viewport: vec2<f32>,
    fov: f32,
    near: f32,
    far: f32,
    shadowSoftness: f32,
    shadowSamples: u32,
    reflectionEnabled: u32,
    _reserved0: u32,
    instanceCount: u32,
    time: f32,
    pointLightCount: u32,
    shadowStrength: f32,
    _pad2: f32,
    exposure: f32,
    vignetteStrength: f32,
    vignetteInner: f32,
    vignetteOuter: f32,
    posterizeBands: f32,
    ditherStrength: f32,
    tonemapMode: u32,
    fxaaEnabled: u32,
}`;

export const SKY_STRUCT_WGSL = /* wgsl */ `
struct Sky {
    hazeDensity: f32,
    horizonBand: f32,
    _pad3: f32,
    _pad4: f32,
    hazeColor: vec4<f32>,
    skyZenith: vec4<f32>,
    skyHorizon: vec4<f32>,
    moonParams: vec4<f32>,
    moonDirection: vec4<f32>,
    starParams: vec4<f32>,
    cloudParams: vec4<f32>,
    cloudColor: vec4<f32>,
    sunParams: vec4<f32>,
    sunVisualColor: vec4<f32>,
    sunDirection: vec4<f32>,
}`;

export const DATA_STRUCT_WGSL = /* wgsl */ `
struct Data {
    baseColor: vec4<f32>,
    pbr: vec4<f32>,
    emission: vec4<f32>,
    flags: u32,
    sizeX: f32,
    sizeY: f32,
    sizeZ: f32,
}`;

export const OKLAB_WGSL = /* wgsl */ `
fn toOKLab(c: vec3<f32>) -> vec3<f32> {
    let lms = vec3(
        0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
        0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
        0.0883024619 * c.r + 0.2220049174 * c.g + 0.6896926207 * c.b,
    );
    let cbrt = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
    return vec3(
        0.2104542553 * cbrt.x + 0.7936177850 * cbrt.y - 0.0040720468 * cbrt.z,
        1.9779984951 * cbrt.x - 2.4285922050 * cbrt.y + 0.4505937099 * cbrt.z,
        0.0259040371 * cbrt.x + 0.7827717662 * cbrt.y - 0.8086757660 * cbrt.z,
    );
}

fn fromOKLab(lab: vec3<f32>) -> vec3<f32> {
    let l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    return max(vec3(
         4.0767416621 * l*l*l - 3.3077115913 * m*m*m + 0.2309699292 * s*s*s,
        -1.2684380046 * l*l*l + 2.6097574011 * m*m*m - 0.3413193965 * s*s*s,
        -0.0041960863 * l*l*l - 0.7034186147 * m*m*m + 1.7076147010 * s*s*s,
    ), vec3(0.0));
}

fn darkTone(base: vec3<f32>) -> vec3<f32> {
    let lab = toOKLab(base);
    return fromOKLab(vec3(lab.x * 0.75, lab.y, lab.z - 0.02));
}

fn lightTone(base: vec3<f32>) -> vec3<f32> {
    let lab = toOKLab(base);
    return fromOKLab(vec3(lab.x * 1.12, lab.y, lab.z + 0.02));
}
`;

export const SPACE_CONVERT_WGSL = /* wgsl */ `
fn toWorldSpace(localPos: vec3<f32>, eid: u32) -> vec3<f32> {
    return (matrices[eid] * vec4(localPos, 1.0)).xyz;
}
fn toObjectSpace(wp: vec3<f32>, eid: u32) -> vec3<f32> {
    let m = matrices[eid];
    let p = wp - m[3].xyz;
    return vec3(dot(p, m[0].xyz), dot(p, m[1].xyz), dot(p, m[2].xyz));
}
`;

export const POINT_LIGHT_STRUCT_WGSL = /* wgsl */ `
struct PointLightData {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    shadowIdx: f32,
}`;

export const POINT_SHADOW_STRUCT_WGSL = /* wgsl */ `
struct PointShadow {
    viewProj: array<mat4x4<f32>, 24>,
    lightPosRadius: array<vec4<f32>, 4>,
}`;

export const SHADOW_STRUCT_WGSL = /* wgsl */ `
struct Shadow {
    cascade0ViewProj: mat4x4<f32>,
    cascade1ViewProj: mat4x4<f32>,
    cascade2ViewProj: mat4x4<f32>,
    cascade3ViewProj: mat4x4<f32>,
    cascadeSplits: vec4<f32>,
    cascadeTexelSizes: vec4<f32>,
}`;

export const VERTEX_PULL_WGSL = /* wgsl */ `
struct PulledVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
}

fn pullVertex(vertexIndex: u32, eid: u32) -> PulledVertex {
    let shapeId = shapes[eid];
    let sm = meshMeta[shapeId];
    let vtxOffset = sm.x + vertexIndex * 8u;
    var v: PulledVertex;
    v.position = vec3(meshVertexData[vtxOffset], meshVertexData[vtxOffset+1u], meshVertexData[vtxOffset+2u]);
    v.normal = vec3(meshVertexData[vtxOffset+3u], meshVertexData[vtxOffset+4u], meshVertexData[vtxOffset+5u]);
    v.uv = vec2(meshVertexData[vtxOffset+6u], meshVertexData[vtxOffset+7u]);
    return v;
}
`;

export const WGSL_STRUCTS = /* wgsl */ `
struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instance: u32,
}

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) @interpolate(flat) entityId: u32,
    @location(3) worldPos: vec3<f32>,
    @location(4) objectPos: vec3<f32>,
    @location(5) objectNormal: vec3<f32>,
    @location(6) uv: vec2<f32>,
}

${SURFACE_DATA_STRUCT_WGSL}
${OKLAB_WGSL}
${SPACE_CONVERT_WGSL}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) entityId: u32,
}

${SCENE_STRUCT_WGSL}

${DATA_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> sizes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> data: array<Data>;
@group(0) @binding(8) var<storage, read> shapes: array<u32>;
@group(0) @binding(9) var<storage, read> meshVertexData: array<f32>;
@group(0) @binding(10) var<storage, read> meshMeta: array<vec4<u32>>;

${VERTEX_PULL_WGSL}
`;
