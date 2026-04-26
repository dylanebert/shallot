import { registry, type Registry } from "../../../engine";
import { buf, type Buf } from "../../../engine";
import { createFieldProxy, type FieldProxy } from "../../../engine/ecs/core";
import { WGSL_STRUCTS } from "./structs";
import { WGSL_LIGHTING_CALC, SPECULAR_WGSL } from "./shaders";
import { validateSurface } from "./validate";
import { compileSurfaceBlock } from "./compile";

export type PropertyType = "f32" | "u32" | "i32";

export interface Property {
    name: string;
    type: PropertyType;
}

export interface SurfaceData {
    vertex?: string;
    fragment?: string;
    properties?: Property[];
}

const MAX_SURFACES = 32;

export const surfaceRegistry: Registry<SurfaceData> = registry(MAX_SURFACES);

function initBuiltIns(): void {
    surfaceRegistry.add({}, "default");
    surfaceRegistry.add(
        { fragment: `(*surface).baseColor = (*surface).worldNormal * 0.5 + 0.5;` },
        "normals",
    );
    surfaceRegistry.add(
        {
            fragment: `
    let depth = position.z;
    let remapped = pow(1.0 - depth, 0.1);
    (*surface).baseColor = vec3(remapped);`,
        },
        "depth",
    );
    surfaceRegistry.add({}, "albedo");
}

initBuiltIns();

export const SurfaceType = {
    Default: 0,
    Normals: 1,
    Depth: 2,
    Albedo: 3,
} as const;

export function surface(data: SurfaceData, name?: string): number {
    validateSurface(data);
    if (data.properties) registerProperties(data.properties);
    return surfaceRegistry.add(data, name);
}

export function clearDefaultSurfaces(): void {
    surfaceRegistry.clear();
    clearProperties();
    initBuiltIns();
}

export const SurfaceIds = buf(Uint32Array, 1, 0);

const MAX_INSTANCE_DATA_SIZE = 128;

interface FieldEntry {
    name: string;
    type: PropertyType;
    offset: number;
    data: Buf;
}

const propertyMap = new Map<string, FieldEntry>();
let propertyEntries: FieldEntry[] = [];
let propertyStride = 0;

function rebuildInstanceLayout(): void {
    propertyEntries = [...propertyMap.values()].sort((a, b) => a.offset - b.offset);
    const lastField = propertyEntries[propertyEntries.length - 1];
    const rawSize = lastField ? lastField.offset + 4 : 0;
    propertyStride = Math.ceil(rawSize / 16) * 16;
}

export function registerProperties(fields: Property[]): void {
    const prevSize = propertyMap.size;
    for (const field of fields) {
        const existing = propertyMap.get(field.name);
        if (existing) {
            if (existing.type !== field.type) {
                throw new Error(
                    `[surface] property "${field.name}" registered as ${existing.type}, cannot re-register as ${field.type}`,
                );
            }
            continue;
        }
        const offset = propertyMap.size * 4;
        if (offset + 4 > MAX_INSTANCE_DATA_SIZE) {
            console.warn(
                `[surface] property data exceeds ${MAX_INSTANCE_DATA_SIZE}-byte cap, field "${field.name}" skipped`,
            );
            continue;
        }
        const Type =
            field.type === "u32" ? Uint32Array : field.type === "i32" ? Int32Array : Float32Array;
        propertyMap.set(field.name, {
            name: field.name,
            type: field.type,
            offset,
            data: buf(Type, 1, 0),
        });
    }
    if (propertyMap.size !== prevSize) rebuildInstanceLayout();
}

export function instanceLayout(): ReadonlyMap<string, { type: PropertyType; offset: number }> {
    return propertyMap;
}

export function instanceEntries(): readonly FieldEntry[] {
    return propertyEntries;
}

export function instanceStride(): number {
    return propertyStride;
}

export function propertyCount(): number {
    return propertyMap.size;
}

export function hasProperties(): boolean {
    return propertyMap.size > 0;
}

export function property(name: string): FieldProxy | null {
    const entry = propertyMap.get(name);
    return entry ? createFieldProxy(entry.data, 1, 0) : null;
}

export function clearProperties(): void {
    propertyMap.clear();
    propertyEntries = [];
    propertyStride = 0;
}

export function instanceStructWGSL(): string {
    if (propertyEntries.length === 0) return "";

    const fields = propertyEntries.map((f) => `    ${f.name}: ${f.type},`);
    const padCount = (propertyStride - propertyEntries.length * 4) / 4;
    for (let i = 0; i < padCount; i++) {
        fields.push(`    _pad${i}: u32,`);
    }

    return `struct InstanceData {\n${fields.join("\n")}\n}`;
}

export function instanceBindingWGSL(binding: number): string {
    if (propertyEntries.length === 0) return "";
    return `@group(0) @binding(${binding}) var<storage, read> instanceData: array<InstanceData>;`;
}

export function instancePackingShader(): string {
    if (propertyEntries.length === 0) return "";

    const structWGSL = instanceStructWGSL();
    const fieldAssignments = propertyEntries
        .map((f, i) => {
            const read = `source[${i}u * count + eid]`;
            return `    d.${f.name} = ${f.type === "u32" ? read : `bitcast<${f.type}>(${read})`};`;
        })
        .join("\n");

    return /* wgsl */ `
${structWGSL}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> instanceData: array<InstanceData>;
@group(0) @binding(2) var<storage, read> entityCount: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    let count = entityCount[0];
    if (eid >= count) { return; }

    var d: InstanceData;
${fieldAssignments}
    instanceData[eid] = d;
}
`;
}

export function compileSurface(data: SurfaceData): string {
    const surfaceBlock = compileSurfaceBlock([data], {
        lighting: {
            params: "shadowFactor: f32",
            body: () => `${WGSL_LIGHTING_CALC}
    return litColor;`,
        },
    });

    return /* wgsl */ `
${WGSL_STRUCTS}
${SPECULAR_WGSL}

${surfaceBlock}

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    let eid = entityIds[input.instance];
    let world = matrices[eid];
    let vtx = pullVertex(input.vertexIndex, eid);
    let position = vtx.position;
    let normal = vtx.normal;
    let result = dispatchVertexTransform(0u, position, normal, vtx.uv, eid);
    let scaledPos = result.position * sizes[eid].xyz;
    let finalWorldPos = (world * vec4<f32>(scaledPos, 1.0)).xyz;
    let worldNormal = normalize((world * vec4<f32>(normal, 0.0)).xyz);

    var output: VertexOutput;
    output.position = scene.viewProj * vec4<f32>(finalWorldPos, 1.0);
    output.color = data[eid].baseColor;
    output.worldNormal = worldNormal;
    output.entityId = eid;
    output.worldPos = finalWorldPos;
    output.objectPos = position * sizes[eid].xyz;
    output.objectNormal = normal;
    output.uv = result.uv;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    let eid = input.entityId;
    let d = data[eid];

    var surface: SurfaceData;
    surface.worldPos = input.worldPos;
    surface.objectPos = input.objectPos;
    surface.worldNormal = normalize(input.worldNormal);
    surface.objectNormal = normalize(input.objectNormal);
    surface.baseColor = input.color.rgb;
    surface.emission = d.emission.rgb * d.emission.a;
    surface.uv = input.uv;
    surface.roughness = d.pbr.x;
    surface.reflectivity = d.pbr.y;
    surface.opacity = 1.0;

    dispatchFragment(0u, &surface, input.position, eid);
    if (surface.opacity <= 0.0) { discard; }

    let shadowFactor = 1.0;
    let litColor = dispatchLighting(0u, surface, shadowFactor);

    var output: FragmentOutput;
    output.color = vec4<f32>(litColor, input.color.a);
    output.entityId = input.entityId;
    return output;
}
`;
}
