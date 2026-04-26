import type { SurfaceData } from "./index";
import { surfaceRegistry, hasProperties } from "./index";
import { compileVertexBody, injectInstPreamble } from "./shaders";

export interface PipelineVariantConfig {
    lighting?: {
        params: string;
        body: (id: number) => string;
    };
}

export const OPACITY_GUARD_WGSL = "if (surface.opacity <= 0.0) { discard; }";

export function compileVertexVariant(id: number, data: SurfaceData): string {
    const rawVertexBody = compileVertexBody(data.vertex);
    const needsProps = data.properties && data.properties.length > 0 && hasProperties();
    const vertexNeedsProps = needsProps && data.vertex?.includes("inst.");
    const vertexBody = vertexNeedsProps ? injectInstPreamble(rawVertexBody) : rawVertexBody;
    return `
fn userVertexTransform_${id}(localPos: vec3<f32>, normal: vec3<f32>, meshUv: vec2<f32>, eid: u32) -> VertexTransformResult {
    ${vertexBody}
}`;
}

function compileSurfaceVariant(
    id: number,
    data: SurfaceData,
    config?: PipelineVariantConfig,
): string {
    const needsProps = data.properties && data.properties.length > 0 && hasProperties();
    const propsPreamble = needsProps ? "let inst = instanceData[eid];\n    " : "";
    const fragmentBody = propsPreamble + (data.fragment ?? "");

    let lightingFn = "";
    if (config?.lighting) {
        lightingFn = `
fn applyLighting_${id}(surface: SurfaceData, ${config.lighting.params}) -> vec3<f32> {
    ${config.lighting.body(id)}
}
`;
    }

    const label = surfaceRegistry.getName(id) ?? `#${id}`;
    return `
// === surface ${id}: "${label}" ===
fn userFragment_${id}(surface: ptr<function, SurfaceData>, position: vec4<f32>, eid: u32) {
    ${fragmentBody}
}
${lightingFn}`;
}

export function compileVertexDispatch(surfaceCount: number): string {
    const cases = Array.from(
        { length: surfaceCount },
        (_, i) =>
            `        case ${i}u: { return userVertexTransform_${i}(localPos, normal, uv, eid); }`,
    ).join("\n");

    return `
struct VertexTransformResult {
    position: vec3<f32>,
    uv: vec2<f32>,
}

fn dispatchVertexTransform(surfaceId: u32, localPos: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, eid: u32) -> VertexTransformResult {
    switch surfaceId {
${cases}
        default: { return userVertexTransform_0(localPos, normal, uv, eid); }
    }
}`;
}

function compileDispatchFunctions(surfaceCount: number, config?: PipelineVariantConfig): string {
    const fragmentCases = Array.from(
        { length: surfaceCount },
        (_, i) => `        case ${i}u: { userFragment_${i}(surface, position, eid); }`,
    ).join("\n");

    let lightingDispatch = "";
    if (config?.lighting) {
        const lightingCases = Array.from(
            { length: surfaceCount },
            (_, i) =>
                `        case ${i}u: { return applyLighting_${i}(surface, ${paramNames(config.lighting!.params)}); }`,
        ).join("\n");

        lightingDispatch = `
fn dispatchLighting(surfaceId: u32, surface: SurfaceData, ${config.lighting.params}) -> vec3<f32> {
    switch surfaceId {
${lightingCases}
        default: { return applyLighting_0(surface, ${paramNames(config.lighting.params)}); }
    }
}
`;
    }

    return `
${compileVertexDispatch(surfaceCount)}

fn dispatchFragment(surfaceId: u32, surface: ptr<function, SurfaceData>, position: vec4<f32>, eid: u32) {
    switch surfaceId {
${fragmentCases}
        default: { userFragment_0(surface, position, eid); }
    }
}
${lightingDispatch}`;
}

export function compileSurfaceBlock(
    surfaces: SurfaceData[],
    config?: PipelineVariantConfig,
): string {
    const vertexVariants = surfaces.map((s, i) => compileVertexVariant(i, s)).join("\n");
    const fragmentVariants = surfaces.map((s, i) => compileSurfaceVariant(i, s, config)).join("\n");
    const dispatch = compileDispatchFunctions(surfaces.length, config);
    return `${vertexVariants}\n${fragmentVariants}\n${dispatch}`;
}

function paramNames(params: string): string {
    return params
        .split(",")
        .map((p) => p.trim().split(":")[0].trim())
        .join(", ");
}
