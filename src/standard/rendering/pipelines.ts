// StandardRenderer's pipeline compilation: the compiled-surface / compiled-background caches and the async
// TypeGPU pipeline factories that fill them. `atlas.ts` supplies the shadow-atlas bind-group layouts every color +
// point + cascade pipeline binds group 1 against. `forward.ts` owns bind-group *resolution* per draw
// (`record()`) — this file compiles pipelines and caches them by surface name and spec identity.

import type { TgpuBindGroupLayout, TgpuRenderPipeline } from "typegpu";
import tgpu from "typegpu";
import type { AnyData, AnyWgslData, AnyWgslStruct } from "typegpu/data";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { decodePos, decodeUv, MeshQuant, meshIdOf } from "../../core/mesh";
import {
    Camera,
    DEPTH_FORMAT,
    Frame,
    fullscreenVertex,
    NORMAL_FORMAT,
    NormalPrepass,
    NormalPrepassRequests,
    RenderContext,
    SAMPLE_COUNT,
    Views,
    Xform,
    xformNormal,
    xformPoint,
} from "../../core/rendering";
import { precompile, precompileScope, type World } from "../../engine";
import { octDecodeNormal } from "../../engine/utils";
import { cascadeLayout, pointLayout, shadowLayout } from "./atlas";
import { validateBindingLimits } from "./binding-limits";
import { LightCull } from "./cluster";
import type { Background, BackgroundLayout } from "./contract";
import { BackgroundContext, Backgrounds } from "./contract";
import {
    engineLayout,
    fragCoord,
    fragWorld,
    lightingSlot,
    litNormal,
    litRecorded,
    pointScale,
    prepassDepthRead,
    prepassDepthSlot,
    prepassNormalRead,
    prepassNormalSlot,
    recordLitNormal,
    sunVisibility,
} from "./engine";
import { foggedColor, fogSlot, fogSource, unfoggedColor } from "./fog";
import type { Recorded } from "./forward";
import { indirectLightSlot, indirectLightSources } from "./indirect";
import { viewLighting } from "./lighting";
import type { MaterialBinding, MaterialLayout, MaterialType } from "./material-type";
import {
    ALPHA_PIPELINE_KEY_COUNT,
    AlphaModeCode,
    AlphaPipelineKey,
    CullModeCode,
    MaterialTypes,
    MeshVertexInput as MaterialVertexInput,
    materialSource,
    rethrowMaterialError,
} from "./material-type";
import { prepassTextures } from "./prepass-textures";
import type { Draw } from "./registry";
import { sampleSunShadow } from "./shade";
import { ssaoTexture } from "./ssao";

const ALPHA_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};
const PREMULTIPLIED_BLEND: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};
const MULTIPLY_BLEND: GPUBlendState = {
    color: { srcFactor: "dst", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

const alphaColorOutput = tgpu
    .fn(
        [d.vec4f, d.u32, d.f32, d.u32],
        d.vec4f,
    )((color, alphaMode, alphaCutoff, fallback) => {
        "use gpu";
        const mask = alphaMode === AlphaModeCode.Mask;
        const coverageFallback =
            fallback === 0xffffffff && alphaMode === AlphaModeCode.AlphaToCoverage;
        if (mask || coverageFallback) {
            const cutoff = alphaCutoff;
            if (color.a < cutoff) std.discard();
            return d.vec4f(color.rgb, 1);
        }
        if (alphaMode === AlphaModeCode.Add) {
            return d.vec4f(std.mul(color.rgb, color.a), 0);
        }
        if (alphaMode === AlphaModeCode.Multiply) {
            return d.vec4f(std.mul(color.rgb, color.a), color.a);
        }
        // Scene alpha is coverage for a camera stacked over others, so opaque ignores material alpha,
        // as Bevy's alpha_discard does.
        if (alphaMode === AlphaModeCode.Opaque) return d.vec4f(color.rgb, 1);
        return d.vec4f(color);
    })
    .$name("materialAlphaOutput");

const colorOutput = { color: d.vec4f };
const coverageOutput = { color: d.vec4f, mask: d.builtin.sampleMask };

/**
 * AlphaToCoverage's sample mask on a camera stacked over others: the first round(alpha × SAMPLE_COUNT)
 * samples, each storing alpha 1, so resolved scene alpha is coverage, which that camera composites by.
 * Hardware alphaToCoverage stores the material alpha in each covered sample too, counting it twice;
 * it stays the path for a lone or bottom camera, which composites by nothing.
 */
const coverageMask = tgpu
    .fn(
        [d.f32],
        d.u32,
    )((alpha) => {
        "use gpu";
        const samples = d.u32(std.round(std.saturate(alpha) * SAMPLE_COUNT));
        return (d.u32(1) << samples) - 1;
    })
    .$name("coverageMask");

export type BindResource =
    | import("typegpu").TgpuBuffer<AnyData>
    | GPUBuffer
    | GPUTexture
    | GPUSampler;

interface PipelineState {
    compiledMaterials: Map<string, CompiledMaterial>;
    materialGroups: Map<string, MaterialGroupEntry>;
    engineCaches: WeakMap<GPUBuffer, Map<number, GPUBindGroup>>;
    bgQuant: GPUBuffer | null;
    compiledBackgrounds: Map<AnyBackground, CompiledBackground>;
}

const pipelineStateKey = { create: createPipelineState };

function createPipelineState(): PipelineState {
    return {
        compiledMaterials: new Map(),
        materialGroups: new Map(),
        engineCaches: new WeakMap(),
        bgQuant: null,
        compiledBackgrounds: new Map(),
    };
}

function pipelineState(world: World): PipelineState {
    return world.resource(pipelineStateKey);
}

export function initializePipelineState(world: World): void {
    world.resource(pipelineStateKey);
}

export function clearGroups(world: World): void {
    pipelineState(world).materialGroups.clear();
}

export function resetPipelineCaches(world: World): void {
    pipelineState(world).compiledMaterials.clear();
    pipelineState(world).compiledBackgrounds.clear();
    pipelineState(world).materialGroups.clear();
    pipelineState(world).engineCaches = new WeakMap();
    pipelineState(world).bgQuant?.destroy();
    pipelineState(world).bgQuant = null;
}

export interface CompiledMaterial {
    /** exact registry spec + layout this entry was compiled from; name alone cannot distinguish
     * a same-name replacement after warm (or an in-place layout swap). */
    owner: AnyMaterialType;
    layout: MaterialLayout<Record<string, MaterialBinding>, AnyWgslStruct>;
    color: TgpuRenderPipeline<{ color: d.Vec4f }> | null;
    transparent: TgpuRenderPipeline<{ color: d.Vec4f }> | null;
    // Alpha surfaces write no prepass depth. Opaque surfaces use the compact depth layout;
    // clipped surfaces use the main stream for their authored cutoff. The lane matches the camera's MSAA count.
    prepass: TgpuRenderPipeline<any> | null;
    prepassSingle: TgpuRenderPipeline<any> | null;
    normal: TgpuRenderPipeline<any> | null;
    normalSingle: TgpuRenderPipeline<any> | null;
    // the point/cascade shadow-atlas pipelines. `null` when the type opts out of that shadow route.
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
    /** AlphaToCoverage's coverage-mask variant, created by ensureStacked on the first frame a 4× MSAA
     * camera stacked over another draws it. */
    stacked: TgpuRenderPipeline<{ color: d.Vec4f }> | null;
    /** AA-off variant, created by ensureSingle on the first frame a no-AA camera draws it. */
    single: {
        color: TgpuRenderPipeline<{ color: d.Vec4f }> | null;
        transparent: TgpuRenderPipeline<{ color: d.Vec4f }> | null;
    } | null;
    // the fixed inputs `ensureSingle` re-compiles the 1× twin from; the entry fns are reused, so the
    // twin shares the authored vs/fs, differing only in multisample
    args: {
        vertex: ReturnType<typeof colorVs> | ReturnType<typeof varyingVs>;
        fragment: ReturnType<typeof colorFs>;
        singleFragment: ReturnType<typeof colorFs>;
        alphaPipelineKey: number;
        cullMode: number;
        primitive: GPUPrimitiveState;
        name: string;
    };
}

/**
 * Slot-independent group-2 state, one entry per draw. Color uses the layout's 16-byte main stream;
 * opaque depth uses the distinct layout.depthVariant object and 8-byte position stream, while clipped
 * depth and a material vertex function need the main stream's normals and UVs. Atlas groups substitute re-gathered eids.
 * engineCache holds group-0 instances per view slot, shared by draws using the same quant buffer.
 * Every atlas tile binds the camera the shadows are fit to; projection uses the tile viewProj.
 * The light's view is not exposed.
 */
export type MaterialGroupEntry = {
    producer: import("./producer-table").ProducerTable | null;
    /** exact registry spec this group was built for — resource identity alone is insufficient when a
     * same-name surface replacement carries a different layout with the same buffers. */
    owner: AnyMaterialType;
    layout: MaterialLayout<Record<string, MaterialBinding>, AnyWgslStruct>;
    quant: GPUBuffer;
    color: GPUBindGroup;
    /** The same material bindings with standard's per-view sorted identity table in place of `eids`. */
    transparentColor: GPUBindGroup | null;
    transparentEids: object | null;
    values: Record<string, unknown>;
    depth: GPUBindGroup | null;
    point: GPUBindGroup | null;
    cascade: GPUBindGroup | null;
    /** the surface's resolved instance-id source before the atlas swaps in its re-gathered list. */
    eids: GPUBuffer | null;
    engineCache: Map<number, GPUBindGroup>;
    resources: BindResource[];
    /** the layout's own binding names in `resources` order (after the four mesh streams), used to
     * compare live resource identities on each draw. */
    names: string[];
    /** per compiled pipeline, that pipeline with this entry's group 2 and index buffer bound, built on the
     * entry's first draw through it (the pass binds groups 0 and 1 per draw). */
    bound: Map<TgpuRenderPipeline<any>, TgpuRenderPipeline<any>>;
    /** the entry's frame-draw record, rewritten in place when a steady frame re-resolves the draw. */
    item: { draw: Draw; r: Recorded };
};

/** the cached per-draw group-2 state for a Draw name, or `undefined` on a cache miss (`record`
 * rebuilds it). Supplying the current surface also invalidates a same-name replacement: bind groups
 * are layout-object-specific even when every resolved GPU resource is unchanged. */
export function getGroup(
    world: World,
    name: string,
    materialType?: AnyMaterialType,
): MaterialGroupEntry | undefined {
    const entry = pipelineState(world).materialGroups.get(name);
    if (
        entry &&
        materialType &&
        (entry.owner !== materialType || entry.layout !== materialType.layout)
    ) {
        pipelineState(world).materialGroups.delete(name);
        return undefined;
    }
    return entry;
}

/** cache a draw's resolved group-2 state (`record`, on a resource-identity change). */
export function setGroup(world: World, name: string, entry: MaterialGroupEntry): void {
    pipelineState(world).materialGroups.set(name, entry);
}

/** the engine group-0 cache for one meshQuant buffer, shared by every draw over it. Held weakly per
 * world, so a churned quant buffer (a glTF import's) drops its groups with its last entry. */
export function engineCache(world: World, quant: GPUBuffer): Map<number, GPUBindGroup> {
    const caches = pipelineState(world).engineCaches;
    let cache = caches.get(quant);
    if (!cache) {
        cache = new Map();
        caches.set(quant, cache);
    }
    return cache;
}

/** the engine group-0 bind group for a view slot against one meshQuant buffer — the shared live
 * `engineLayout` instance a draw at that slot binds (frame / per-slot View / lighting /
 * light-cull outputs / materials / the dequant table), built lazily into `quant`'s
 * {@link engineCache}. */
const engineCacheLights = new WeakMap<Map<number, GPUBindGroup>, GPUBuffer>();
const engineCacheOcclusion = new WeakMap<GPUBindGroup, GPUTextureView>();
const engineCacheDepth = new WeakMap<GPUBindGroup, GPUTextureView>();
const engineCacheNormal = new WeakMap<GPUBindGroup, GPUTextureView>();

export function engineGroup(
    world: World,
    cache: Map<number, GPUBindGroup>,
    slot: number,
    quant: GPUBuffer,
    depthPass = false,
): GPUBindGroup {
    const _lightCull = world.resource(LightCull);
    const pointLights = _lightCull.lights;
    if (!pointLights) throw new Error("[render] engine group used before light-cull warm");
    if (engineCacheLights.get(cache) !== pointLights) {
        cache.clear();
        engineCacheLights.set(cache, pointLights);
    }

    const occlusion = ssaoTexture(world, slot);
    const textures = world.resource(prepassTextures);
    const depth = (!depthPass && textures.depth[slot]) || textures.stubDepth!;
    const normal = (!depthPass && textures.normal[slot]) || textures.stubNormal!;
    const key = depthPass ? -slot - 1 : slot;
    const cached = cache.get(key);
    if (
        cached &&
        engineCacheOcclusion.get(cached) === occlusion &&
        engineCacheDepth.get(cached) === depth &&
        engineCacheNormal.get(cached) === normal
    )
        return cached;
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(engineLayout, {
            frame: world.resource(Frame).buffer!,
            view: world.resource(RenderContext).viewBuffers[slot],
            lighting: viewLighting(world, slot),
            pointLights,
            meshQuant: quant,
            ambientOcclusion: occlusion,
            prepassDepth: depth,
            prepassNormal: normal,
        }),
    );
    engineCacheOcclusion.set(group, occlusion);
    engineCacheDepth.set(group, depth);
    engineCacheNormal.set(group, normal);
    cache.set(key, group);
    return group;
}

// Backgrounds must fill engineLayout's meshQuant binding even though they read no mesh.

/** the never-read `meshQuant` placeholder a background's engine group binds. */
export function bgQuant(world: World): GPUBuffer {
    const resources = pipelineState(world);
    resources.bgQuant ??= world.gpu.device.createBuffer({
        label: "standard-bg-quant",
        size: d.sizeOf(MeshQuant),
        usage: GPUBufferUsage.STORAGE,
    });
    return resources.bgQuant;
}

/** the widest `MaterialType` shape (any bindings, any varyings) — the bare `MaterialType` default pins
 *  varyings to `Record<string, never>`, so every internal pipeline helper takes this wider alias to
 *  accept a real varyings-carrying surface (`vertex`) at the call boundary. */
type AnyMaterialType = MaterialType<
    any,
    Record<string, MaterialBinding>,
    Record<string, AnyWgslData>
>;

const identityXform = tgpu
    .fn(
        [],
        Xform,
    )(() => {
        "use gpu";
        return Xform({ pos: d.vec3f(0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) });
    })
    .$name("identityXform");

const producerRow = tgpu.fn(
    [d.vec4u],
    d.u32,
)((instance) => {
    "use gpu";
    return instance.x === 0xffffffff ? instance.z - 1 : d.u32(0xffffffff);
});

// Bit 31 of the flat material word marks a mirrored instance, one whose placement has an odd count of
// negative scale axes, as Bevy's per-instance SIGN_DETERMINANT_MODEL_3X3 mesh flag does; a material
// vertex function's output does not change it. Fragment contexts take the row from materialRow and the
// face from logicalFrontFacing.
const MIRRORED = 0x80000000;

const mirroredMaterial = tgpu.fn(
    [d.u32, d.vec3f],
    d.u32,
)((material, scale) => {
    "use gpu";
    const xy = scale.x < 0 !== scale.y < 0;
    const mirrored = xy !== scale.z < 0;
    return std.select(material, material | d.u32(MIRRORED), mirrored);
});

const materialRow = tgpu.fn(
    [d.u32],
    d.u32,
)((word) => {
    "use gpu";
    return word & d.u32(MIRRORED - 1);
});

// A mirrored instance's front face rasterizes as a back face (Bevy's winding_corrected_front_facing).
// ctx.frontFacing is logical for every material, where Bevy corrects only its PBR paths, since it is
// already derived from the cull key outside cull-none pipelines.
const logicalFrontFacing = tgpu.fn(
    [d.bool, d.u32],
    d.bool,
)((rasterFront, word) => {
    "use gpu";
    const mirrored = word >= d.u32(MIRRORED);
    return rasterFront !== mirrored;
});

// Reserve locations after the fixed fragment fields to prevent custom varying collisions.
const VARYING_BASE = 7;

function fragmentInterstage(surface: AnyMaterialType): Record<string, AnyWgslData> {
    return {
        ...(surface.fragmentInputs?.uv ? { uv: d.vec2f } : {}),
        ...(surface.fragmentInputs?.localPos ? { localPos: d.vec3f } : {}),
    };
}

const CULL_FACES: readonly GPUCullMode[] = ["back", "none", "front"];

/** Shared triangle-list state for standard material color, depth and atlas pipelines of one CullModeCode. */
function materialPrimitive(cullMode: number): GPUPrimitiveState {
    return { topology: "triangle-list", cullMode: CULL_FACES[cullMode], frontFace: "ccw" };
}

/** A pipeline variant's name: its alpha pipeline key, and its cull mode where it does not cull back. */
function variantName(surface: AnyMaterialType, alphaPipelineKey: number, cullMode: number): string {
    return cullMode === CullModeCode.Back
        ? `${surface.name}-${alphaPipelineKey}`
        : `${surface.name}-${alphaPipelineKey}-${CULL_FACES[cullMode]}`;
}

/** front_facing is a fragment input only where a pipeline culls no face; elsewhere culling fixes it. */
function facingInput(cullMode: number): { frontFacing?: typeof d.builtin.frontFacing } {
    return cullMode === CullModeCode.None ? { frontFacing: d.builtin.frontFacing } : {};
}

function isInstanced(surface: AnyMaterialType): boolean {
    return "eids" in surface.layout.entries && "globalTransforms" in surface.layout.entries;
}

// A zero-custom-varying surface stays entirely TGSL. One fixed function computes the vertex payload;
// four thin entry shapes select only the optional built-in fragment inputs the surface declared. The
// payload is an ordinary function result, not an interstage struct, so carrying uv/localPos here costs no
// raster bandwidth. Custom varying names alone need the WGSL-bodied copier below.
const MaterialTypeVertex = d
    .struct({
        pos: d.vec4f,
        worldNormal: d.vec3f,
        eid: d.u32,
        row: d.u32,
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        material: d.u32,
        alphaMode: d.u32,
        alphaCutoff: d.f32,
    })
    .$name("MaterialTypeVertex");

function materialFragmentFunction(surface: AnyMaterialType) {
    const fragment = surface.fragment;
    const Context = (fragment as any).shell.argTypes[0];
    const { parameters: _, ...fields } = Context.propTypes;
    const Input = d.struct(fields);
    const bound = surface.layout.$;
    const extension = surface.extension;
    const Parameters = Context.propTypes.parameters;
    const row = "bound.materialParameters[input.material]";
    const parameters = extension
        ? `Parameters(${Object.keys(Parameters.propTypes)
              .map((key) => `${row}.${key}`)
              .join(", ")})`
        : row;
    const extensionArgument = extension
        ? `, Extension(${Object.keys(extension.propTypes)
              .map((key) => `${row}.${key}`)
              .join(", ")})`
        : "";
    return tgpu
        .fn(
            [Input],
            d.vec4f,
        )(`(input: Input) -> vec4f {
        return fragment(Context(${parameters}, ${Object.keys(fields)
            .map((key) => `input.${key}`)
            .join(", ")})${extensionArgument});
    }`)
        .$uses({
            Input,
            Context,
            fragment,
            bound,
            Parameters,
            ...(extension ? { Extension: extension } : {}),
        });
}

function materialVertexFunction(surface: AnyMaterialType) {
    const vertex = surface.vertex;
    if (!vertex) return undefined;
    const shell = (vertex as any).shell;
    const Input = shell.argTypes[0];
    const layout = surface.layout;
    if (surface.extension) {
        const Extension = surface.extension;
        const Parameters = Input.propTypes.parameters;
        const row = "bound.materialParameters[input.material]";
        const fields = Object.keys(Input.propTypes).filter((key) => key !== "parameters");
        return tgpu
            .fn(
                [MaterialVertexInput],
                shell.returnType,
            )(`(input: MaterialVertexInput) -> Output {
            return vertex(Input(Parameters(${Object.keys(Parameters.propTypes)
                .map((key) => `${row}.${key}`)
                .join(
                    ", ",
                )}), ${fields.map((key) => `input.${key}`).join(", ")}), Extension(${Object.keys(
                Extension.propTypes,
            )
                .map((key) => `${row}.${key}`)
                .join(", ")}));
        }`)
            .$uses({
                MaterialVertexInput,
                Output: shell.returnType,
                Input,
                Parameters,
                Extension,
                vertex,
                bound: layout.$,
            });
    }
    return tgpu.fn(
        [MaterialVertexInput],
        shell.returnType,
    )((input) => {
        "use gpu";
        return vertex(
            Input({
                parameters: layout.$.materialParameters[input.material],
                localPos: input.localPos,
                localNormal: input.localNormal,
                uv: input.uv,
                vidx: input.vidx,
                eid: input.eid,
                row: input.row,
                iid: input.iid,
                xform: input.xform,
                world: input.world,
                worldNormal: input.worldNormal,
                material: input.material,
            }),
        );
    });
}

function colorVertex(surface: AnyMaterialType, clip: boolean, suffix = clip ? "Clip" : "") {
    const instanced = isInstanced(surface);
    const hasVs = !!surface.vertex;
    const vsFn = materialVertexFunction(surface);
    const layout = surface.layout;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    return tgpu
        .fn(
            [d.u32, d.u32],
            MaterialTypeVertex,
        )((vidx, iid) => {
            "use gpu";
            const v = layout.$.vertices[vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            let row = d.u32(0xffffffff);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let material = d.u32(0);
            let alphaMode = d.u32(AlphaModeCode.Opaque);
            let alphaCutoff = d.f32(0.5);
            let xform = identityXform();
            if (instanced) {
                const instance = bound.eids[iid];
                eid = instance.x;
                row = producerRow(instance);
                const encodedMeshInstance = instance.z;
                if (encodedMeshInstance !== 0) {
                    const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                    material = meshInstance.material;
                    alphaMode = meshInstance.alphaMode;
                    alphaCutoff = meshInstance.alphaCutoff;
                }
                xform = Xform(bound.globalTransforms[instance.y]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let pos = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    MaterialVertexInput({
                        localPos,
                        localNormal,
                        uv,
                        vidx,
                        eid,
                        row,
                        iid,
                        xform,
                        world,
                        worldNormal,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            pos = d.vec4f(std.mul(engineLayout.$.view.clipFromWorld, world));
            if (clip) pos = d.vec4f(std.add(pos, d.vec4f(shadowLayout.$.tileRects.rects[0].x * 0)));
            return MaterialTypeVertex({
                pos,
                worldNormal: std.normalize(worldNormal),
                eid,
                row,
                world: world.xyz,
                uv,
                localPos,
                material: mirroredMaterial(material, xform.scale),
                alphaMode,
                alphaCutoff,
            });
        })
        .$name(`${surface.name}${suffix}Vertex`);
}

function colorVs(surface: AnyMaterialType, clip = false, suffix = clip ? "Clip" : "") {
    const vertex = colorVertex(surface, clip, suffix);
    const name = `${surface.name}${suffix}Vs`;
    const input = { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex };
    const fixed = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
    };
    const uv = !!surface.fragmentInputs?.uv;
    const localPos = !!surface.fragmentInputs?.localPos;
    if (uv && localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    uv: v.uv,
                    localPos: v.localPos,
                    material: v.material,
                    alphaMode: v.alphaMode,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    if (uv) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    uv: v.uv,
                    material: v.material,
                    alphaMode: v.alphaMode,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    if (localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    localPos: v.localPos,
                    material: v.material,
                    alphaMode: v.alphaMode,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    return tgpu
        .vertexFn({ in: input, out: fixed })((i) => {
            "use gpu";
            const v = vertex(i.vidx, i.iid);
            return {
                pos: v.pos,
                worldNormal: v.worldNormal,
                eid: v.eid,
                row: v.row,
                world: v.world,
                material: v.material,
                alphaMode: v.alphaMode,
                alphaCutoff: v.alphaCutoff,
            };
        })
        .$name(name);
}

function colorFs(
    surface: AnyMaterialType,
    alphaToCoverageFallback = false,
    fogged = false,
    coverage = false,
    cullMode: number = CullModeCode.Back,
) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const fogColor = fogged ? foggedColor : unfoggedColor;
    // Reuse the function's schema identity: a structurally equal context forces an implicit conversion.
    const fragment = materialFragmentFunction(surface);
    const CtxSchema = (fragment as any).shell.argTypes[0];
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const inputSchema = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        ...facingInput(cullMode),
    };
    return tgpu
        .fragmentFn({
            in: inputSchema,
            out: (coverage ? coverageOutput : colorOutput) as typeof colorOutput,
        })((input: any) => {
            "use gpu";
            const worldNormal = std.normalize(input.worldNormal);
            sunVisibility.$ = sampleSunShadow(
                input.world,
                worldNormal,
                d.u32(engineLayout.$.view.shadowFilteringMethod),
            );
            fragWorld.$ = d.vec3f(input.world);
            fragCoord.$ = d.vec4f(input.pos);
            pointScale.$ = 1;
            // Shadow receivers read free WGSL names that TypeGPU's call graph cannot discover.
            // Touch their layout bindings in the returned value: the transpiler removes an unused
            // touch as dead code. Multiplying by zero preserves the fragment color.
            const forcedZero =
                (shadowLayout.$.pointShadows.casters[0].pos.x +
                    shadowLayout.$.tileRects.rects[0].x +
                    shadowLayout.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.pointAtlas,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.shadowMap,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            // The no-varyings default adds a Record<string, never> index signature that rejects
            // these known fields in TypeScript; the cast does not change the actual WGSL context schema.
            const ctx = CtxSchema({
                eid: input.eid,
                row: input.row,
                world: input.world,
                worldNormal,
                uv: needUv ? input.uv : d.vec2f(0),
                localPos: needLocalPos ? input.localPos : d.vec3f(0),
                material: materialRow(input.material),
                alphaMode: alphaToCoverageFallback ? AlphaModeCode.Opaque : input.alphaMode,
                alphaCutoff: input.alphaCutoff,
                frontFacing: logicalFrontFacing(
                    readsFacing ? input.frontFacing : facing,
                    input.material,
                ),
                fragCoord: input.pos,
                prepass: false,
            } as any);
            const col = d.vec4f(std.add(fragment(ctx), d.vec4f(forcedZero)));
            const output = alphaColorOutput(
                fogColor(col, input.world, input.alphaMode),
                input.alphaMode,
                input.alphaCutoff,
                alphaToCoverageFallback ? 0xffffffff : 0,
            );
            if (coverage)
                return { color: d.vec4f(output.xyz, 1), mask: coverageMask(output.w) } as never;
            return { color: output };
        })
        .$name(`${surface.name}${alphaToCoverageFallback ? "Single" : ""}Fs`);
}

// Host surface objects cannot cross a TGSL helper boundary; this entry closes over its depth layout.
// Only a material without a vertex function reaches it: the 8-byte stream carries no normal or uv.
function prepassVs(surface: AnyMaterialType) {
    const instanced = isInstanced(surface);
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            let world = d.vec4f(decodePos(v.x, v.y, mq), 1);
            if (instanced) {
                const instance = bound.eids[input.iid];
                const xform = Xform(bound.globalTransforms[instance.y]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
            }
            const clip = std.mul(engineLayout.$.view.clipFromWorld, world);
            // TypeGPU indexes layouts by group number; using only groups 0 and 2 leaves a hole
            // that Dawn rejects at pipeline creation. The returned zero keeps group 1 live;
            // renderPrepass binds shadowGroup there, but the stub receiver never samples it.
            const forcedZero = shadowLayout.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)) };
        })
        .$name(`${surface.name}PrepassVs`);
}

// The normal the fragment's last lit or applyPbrLighting call received, else its interpolated normal.
const prepassNormalColor = tgpu.fn(
    [d.vec3f],
    d.vec4f,
)((worldNormal) => {
    "use gpu";
    let normal = std.normalize(worldNormal);
    if (litRecorded.$) normal = d.vec3f(litNormal.$);
    return d.vec4f(std.add(std.mul(normal, 0.5), d.vec3f(0.5)), 1);
});

/** The material fragment in a depth pass: `clip` discards below the cutoff, `normals` writes
 * {@link prepassNormalColor}. */
function clipFs(
    surface: AnyMaterialType,
    normals = false,
    clip = true,
    cullMode: number = CullModeCode.Back,
) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const fragment = materialFragmentFunction(surface);
    const Ctx = (fragment as any).shell.argTypes[0];
    const input = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        ...facingInput(cullMode),
    };
    return tgpu
        .fragmentFn({ in: input, out: normals ? colorOutput : d.Void })((fin: any) => {
            "use gpu";
            const ctx = Ctx({
                eid: fin.eid,
                row: fin.row,
                world: fin.world,
                worldNormal: std.normalize(fin.worldNormal),
                uv: needUv ? (fin as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (fin as any).localPos : d.vec3f(0),
                material: materialRow(fin.material),
                alphaMode: d.u32(0),
                alphaCutoff: fin.alphaCutoff,
                frontFacing: logicalFrontFacing(
                    readsFacing ? fin.frontFacing : facing,
                    fin.material,
                ),
                fragCoord: fin.pos,
                prepass: true,
            });
            const color = fragment(ctx);
            const cutoff = fin.alphaCutoff;
            if (clip && color.a < cutoff) std.discard();
            if (normals) return { color: prepassNormalColor(fin.worldNormal) } as never;
        })
        .$name(`${surface.name}ClipFs`);
}

function varyingClipFs(
    surface: AnyMaterialType,
    normals = false,
    clip = true,
    cullMode: number = CullModeCode.Back,
) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    const schemas = keys.map((key) => varyings[key]);
    const slots = keys.map((_, i) => `v${i}`);
    const located = Object.fromEntries(
        slots.map((slot, i) => [slot, d.location(VARYING_BASE + i, schemas[i])]),
    );
    const fragment = materialFragmentFunction(surface);
    const Ctx = (fragment as any).shell.argTypes[0];
    const entryIn = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        ...facingInput(cullMode),
        ...located,
    };
    return tgpu
        .fragmentFn({
            in: entryIn,
            out: normals ? colorOutput : d.Void,
        })((input: any) => {
            "use gpu";
            const ctx = Ctx();
            ctx.eid = input.eid;
            ctx.row = input.row;
            ctx.world = d.vec3f(input.world);
            ctx.worldNormal = std.normalize(input.worldNormal);
            ctx.uv = needUv ? d.vec2f((input as any).uv) : d.vec2f(0);
            ctx.localPos = needLocalPos ? d.vec3f((input as any).localPos) : d.vec3f(0);
            ctx.material = materialRow(input.material);
            ctx.alphaMode = d.u32(0);
            ctx.alphaCutoff = input.alphaCutoff;
            ctx.frontFacing = logicalFrontFacing(
                readsFacing ? input.frontFacing : facing,
                input.material,
            );
            ctx.fragCoord = d.vec4f(input.pos);
            ctx.prepass = true;
            for (const i of tgpu.unroll(std.range(keys.length))) {
                (ctx as any)[keys[i]] = (schemas[i] as any)((input as any)[slots[i]]);
            }
            const color = fragment(ctx);
            const cutoff = input.alphaCutoff;
            if (clip && color.a < cutoff) std.discard();
            if (normals) return { color: prepassNormalColor(input.worldNormal) } as never;
        })
        .$name(`${surface.name}ClipFs`);
}

// TGSL cannot construct dynamically named varying fields; generate the per-surface WGSL entry.
function varyingVs(surface: AnyMaterialType, clip = false, suffix = clip ? "Clip" : "") {
    const varyings = surface.varyings ?? {};
    const vsFn = materialVertexFunction(surface);
    if (Object.keys(varyings).length === 0) {
        throw new Error(
            `standard: surface "${surface.name}" reached the raw varying copier without a custom varying`,
        );
    }
    if (!vsFn) {
        throw new Error(
            `standard: surface "${surface.name}" declares varyings with no vs — a varying can only be written by the surface's own vs chunk`,
        );
    }
    const hasVs = !!vsFn;
    const fragmentFields = fragmentInterstage(surface);
    const instanced = isInstanced(surface);
    const layout = surface.layout;
    // The FS calls custom fields v0..v3, so TypeGPU cannot match them to author-named VS fields.
    // Unmatched fields auto-assign from location zero and collide with fixed fields; pin both
    // stages to the same explicit locations so location, not field name, pairs the values.
    const located = Object.fromEntries(
        Object.entries(varyings).map(([k, s], i) => [k, d.location(VARYING_BASE + i, s)]),
    );
    // Construct TypeGPU's entry IO struct directly. Returning a separate copier's struct through
    // a TGSL entry requires an implicit struct conversion and produces a warning at each resolve.
    const assigns = Object.keys(varyings)
        .map((k) => `    out.${k} = patched.${k};`)
        .join("\n");
    const fragmentAssigns = `${surface.fragmentInputs?.uv ? "    out.uv = uv;\n" : ""}${surface.fragmentInputs?.localPos ? "    out.localPos = localPos;\n" : ""}`;
    // Reading layout.$.vertices in host code throws, even inside tgpu.lazy (which runs in normal mode).
    // Pass the whole layout.$ proxy as an external so WGSL's bound.vertices lookup happens at resolve.
    const bound = layout.$;
    const engine = engineLayout.$;
    const shadowG = shadowLayout.$;
    const uses: Record<string, unknown> = {
        bound,
        engine,
        decodePos,
        decodeUv,
        meshIdOf,
        octDecodeNormal,
        Xform,
        producerRow,
    };
    if (hasVs) {
        uses.MaterialVertexInput = MaterialVertexInput;
        uses.vs = vsFn;
    }
    // The engine view is retained only when the generated vertex body uses it; a dead dot-chain
    // reference is removed from the pipeline layout.
    if (clip) uses.shadowG = shadowG;
    uses.mirroredMaterial = mirroredMaterial;
    if (instanced) {
        uses.xformPoint = xformPoint;
        uses.xformNormal = xformNormal;
    }
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                row: d.interpolate("flat", d.u32),
                world: d.vec3f,
                material: d.interpolate("flat", d.u32),
                alphaMode: d.interpolate("flat", d.u32),
                alphaCutoff: d.interpolate("flat", d.f32),
                ...fragmentFields,
                // No flat interpolation is inserted for integer varyings; they are unsupported
                // and fail at resolve or device compilation.
                ...located,
            },
        })(/* wgsl */ `{
    let v = bound.vertices[in.vidx];
    let mq = engine.meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    var eid: u32 = 0u;
    var row: u32 = 0xFFFFFFFFu;
    var xform = Xform(vec3f(0.0), vec4f(0.0, 0.0, 0.0, 1.0), vec3f(1.0));
    var world = vec4f(localPos, 1.0);
    var worldNormal = vec3f(localNormal);
    var material: u32 = 0u;
    var alphaMode: u32 = 0u;
    var alphaCutoff: f32 = 0.5;
${
    instanced
        ? `    let instance = bound.eids[in.iid];
    eid = instance.x;
    row = producerRow(instance);
    let encodedMeshInstance = instance.z;
    if (encodedMeshInstance != 0u) {
        let meshInstance = bound.meshInstances[encodedMeshInstance - 1u];
        material = meshInstance.material;
        alphaMode = meshInstance.alphaMode;
        alphaCutoff = meshInstance.alphaCutoff;
    }
    xform = bound.globalTransforms[instance.y];
    world = vec4f(xformPoint(xform, world.xyz), world.w);
    worldNormal = vec3f(xformNormal(xform, worldNormal));
`
        : ""
}${
    hasVs
        ? `    let patched = vs(MaterialVertexInput(localPos, localNormal, uv, vidx, eid, row, iid, xform, world, worldNormal, material));
    world = patched.world;
    worldNormal = patched.worldNormal;
`
        : ""
}
    var out: Out;
    out.pos = engine.view.clipFromWorld * world${clip ? " + vec4f(shadowG.tileRects.rects[0].x * 0.0)" : ""};
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.row = row;
    out.world = world.xyz;
    out.material = mirroredMaterial(material, xform.scale);
    out.alphaMode = alphaMode;
    out.alphaCutoff = alphaCutoff;
${fragmentAssigns}
${assigns}
    return out;
}`)
        .$uses(uses)
        .$name(`${surface.name}${suffix}Vs`);
}

// Shadow receivers read free WGSL names that TypeGPU's call graph cannot discover.
// Callers add this zero-valued result to their output so the transpiler cannot prune the binding
// touches as dead code.
const shadowForce = tgpu
    .fn(
        [],
        d.f32,
    )(() => {
        "use gpu";
        return (
            (shadowLayout.$.pointShadows.casters[0].pos.x +
                shadowLayout.$.tileRects.rects[0].x +
                shadowLayout.$.sunShadow.enabled +
                std.textureSampleCompareLevel(
                    shadowLayout.$.pointAtlas,
                    shadowLayout.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                ) +
                std.textureSampleCompareLevel(
                    shadowLayout.$.shadowMap,
                    shadowLayout.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                )) *
            0
        );
    })
    .$name("shadowForce");

function varyingFs(
    surface: AnyMaterialType,
    alphaToCoverageFallback = false,
    fogged = false,
    coverage = false,
    cullMode: number = CullModeCode.Back,
) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const out = (coverage ? coverageOutput : colorOutput) as typeof colorOutput;
    const fogColor = fogged ? foggedColor : unfoggedColor;
    const varyings = surface.varyings ?? {};
    const varyingKeys = Object.keys(varyings);
    const schemas = varyingKeys.map((k) => varyings[k]);
    const fsFn = materialFragmentFunction(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const Ctx = (fsFn as any).shell.argTypes[0];
    const slots = varyingKeys.map((_, i) => `v${i}`);
    const located = Object.fromEntries(
        slots.map((slot, i) => [slot, d.location(VARYING_BASE + i, schemas[i])]),
    );
    const base = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        ...facingInput(cullMode),
    };
    const name = `${surface.name}${alphaToCoverageFallback ? "Single" : ""}Fs`;
    const entryIn = { ...base, ...located };
    return tgpu
        .fragmentFn({ in: entryIn, out })((input: any) => {
            "use gpu";
            const worldNormal = std.normalize(input.worldNormal);
            const ctx = Ctx();
            ctx.eid = input.eid;
            ctx.row = input.row;
            ctx.world = d.vec3f(input.world);
            ctx.worldNormal = d.vec3f(worldNormal);
            ctx.uv = needUv ? d.vec2f((input as any).uv) : d.vec2f(0);
            ctx.localPos = needLocalPos ? d.vec3f((input as any).localPos) : d.vec3f(0);
            ctx.material = materialRow(input.material);
            ctx.alphaMode = alphaToCoverageFallback ? AlphaModeCode.Opaque : input.alphaMode;
            ctx.alphaCutoff = input.alphaCutoff;
            ctx.frontFacing = logicalFrontFacing(
                readsFacing ? input.frontFacing : facing,
                input.material,
            );
            ctx.fragCoord = d.vec4f(input.pos);
            ctx.prepass = false;
            for (const i of tgpu.unroll(std.range(varyingKeys.length))) {
                (ctx as any)[varyingKeys[i]] = (schemas[i] as any)((input as any)[slots[i]]);
            }
            const col = fsFn(ctx);
            const shaded = fogColor(
                d.vec4f(std.add(col, d.vec4f(shadowForce()))),
                input.world,
                input.alphaMode,
            );
            const output = alphaColorOutput(
                shaded,
                input.alphaMode,
                input.alphaCutoff,
                alphaToCoverageFallback ? 0xffffffff : 0,
            );
            if (coverage)
                return { color: d.vec4f(output.xyz, 1), mask: coverageMask(output.w) } as never;
            return { color: output };
        })
        .$name(name);
}

function materialRoot(world: World, surface: AnyMaterialType, fogged = false) {
    let root = world.gpu.root.with(indirectLightSlot, indirectLightSources(world));
    const fog = fogged ? fogSource(world) : null;
    if (fog) {
        validateBindingLimits(`material "${surface.name}"`, [
            engineLayout,
            shadowLayout,
            surface.layout,
            fog.layout,
        ]);
        root = root.with(fogSlot, fog.source);
    }
    for (const [accessor, value] of surface.slots ?? []) root = root.with(accessor, value);
    if (surface.readsPrepass?.depth) root = root.with(prepassDepthSlot, prepassDepthRead);
    if (surface.readsPrepass?.normal) root = root.with(prepassNormalSlot, prepassNormalRead);
    return root;
}
function prepareMaterialPipeline<T extends TgpuRenderPipeline<any>>(
    world: World,
    name: string,
    pipeline: T,
): T {
    try {
        world.gpu.root.unwrap(pipeline);
    } catch (error) {
        rethrowMaterialError(name, error);
    }
    return pipeline;
}

/**
 * Compile a `MaterialType`'s opaque color pipeline or its alpha-blended transparent pipeline. Cached
 * by name and exact type/layout identity, so replacing a registry entry after warm cannot inherit the
 * previous owner's pipelines.
 */
export function compileMaterial<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData>,
>(
    world: World,
    surface: MaterialType<P, B, V>,
    capacity: number,
    alphaPipelineKey: number,
    cullMode: number = CullModeCode.Back,
): CompiledMaterial {
    const varyingCount = Object.keys(surface.varyings ?? {}).length;
    const uv = !!surface.fragmentInputs?.uv;
    const localPos = !!surface.fragmentInputs?.localPos;
    // WebGPU counts @builtin(front_facing) against the fragment stage's inter-stage variables.
    const frontFacing = cullMode === CullModeCode.None;
    const interstageCount =
        VARYING_BASE + Number(uv) + Number(localPos) + varyingCount + Number(frontFacing);
    const limit = world.gpu.device.limits.maxInterStageShaderVariables;
    if (interstageCount > limit) {
        const fields = [
            `${VARYING_BASE} engine`,
            ...(uv ? ["uv"] : []),
            ...(localPos ? ["localPos"] : []),
            `${varyingCount} varyings`,
            ...(frontFacing ? ["frontFacing"] : []),
        ];
        throw new Error(
            `standard: material "${surface.name}" uses ${interstageCount} inter-stage variables (${fields.join(", ")}); device.limits.maxInterStageShaderVariables is ${limit}`,
        );
    }
    const _render = world.resource(RenderContext);
    const cacheKey = `${surface.name}:${alphaPipelineKey}:${cullMode}`;
    const cached = pipelineState(world).compiledMaterials.get(cacheKey);
    if (cached?.owner === surface && cached.layout === surface.layout) return cached;
    const hasVaryings = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const vertex = hasVaryings ? varyingVs(surface) : colorVs(surface);
    const fogged =
        !!fogSource(world) &&
        (alphaPipelineKey === AlphaPipelineKey.Blend ||
            alphaPipelineKey === AlphaPipelineKey.Premultiplied ||
            alphaPipelineKey === AlphaPipelineKey.Multiply);
    const fragment = (
        hasVaryings
            ? varyingFs(surface, false, fogged, false, cullMode)
            : colorFs(surface, false, fogged, false, cullMode)
    ) as ReturnType<typeof colorFs>;
    const singleFragment = (
        hasVaryings
            ? varyingFs(surface, true, fogged, false, cullMode)
            : colorFs(surface, true, fogged, false, cullMode)
    ) as ReturnType<typeof colorFs>;
    const name = variantName(surface, alphaPipelineKey, cullMode);
    const primitive = materialPrimitive(cullMode);
    const args: CompiledMaterial["args"] = {
        vertex,
        fragment,
        singleFragment,
        alphaPipelineKey,
        cullMode,
        primitive,
        name,
    };
    const transparent =
        alphaPipelineKey === AlphaPipelineKey.Blend ||
        alphaPipelineKey === AlphaPipelineKey.Premultiplied ||
        alphaPipelineKey === AlphaPipelineKey.Multiply;
    const blend =
        alphaPipelineKey === AlphaPipelineKey.Blend
            ? ALPHA_BLEND
            : alphaPipelineKey === AlphaPipelineKey.Premultiplied
              ? PREMULTIPLIED_BLEND
              : alphaPipelineKey === AlphaPipelineKey.Multiply
                ? MULTIPLY_BLEND
                : undefined;
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: !transparent,
        depthCompare: transparent ? "greater-equal" : "greater",
    };
    const root = materialRoot(world, surface, fogged);
    const pipeline = root
        .createRenderPipeline({
            vertex,
            fragment,
            targets: { color: { format: _render.format, ...(blend ? { blend } : {}) } },
            primitive,
            depthStencil,
            multisample: {
                count: SAMPLE_COUNT,
                alphaToCoverageEnabled: alphaPipelineKey === AlphaPipelineKey.AlphaToCoverage,
            },
        })
        .$name(`standard-${name}`);
    const compiled: CompiledMaterial = {
        owner: surface as AnyMaterialType,
        layout: surface.layout as MaterialLayout<Record<string, MaterialBinding>, AnyWgslStruct>,
        color: transparent ? null : pipeline,
        transparent: transparent ? pipeline : null,
        prepass:
            surface.depthPass?.prepass === false
                ? null
                : compilePrepass(world, surface, alphaPipelineKey, cullMode),
        prepassSingle: null,
        normal: null,
        normalSingle: null,
        point: null,
        cascade: null,
        single: null,
        stacked: null,
        args,
    };
    if (surface.depthPass?.shadows !== false && !transparent) {
        const shadows = compileShadow(world, surface, capacity, alphaPipelineKey, cullMode);
        compiled.point = shadows.point;
        compiled.cascade = shadows.cascade;
    }
    pipelineState(world).compiledMaterials.set(cacheKey, compiled);
    return compiled;
}

/**
 * Creates the AA-off variant once, on the first frame a no-AA camera draws this material.
 * Reuses the compiled vertex and stored single-sample fragment entries with multisample.count=1;
 * the latter supplies the AA-off alpha-to-coverage fallback. TypeGPU defers resolve and device
 * pipeline creation until the variant's first use.
 */
export function ensureSingle(world: World, t: CompiledMaterial): void {
    const _render = world.resource(RenderContext);
    if (t.single) {
        if (t.prepass && !t.prepassSingle)
            t.prepassSingle = compilePrepass(
                world,
                t.owner,
                t.args.alphaPipelineKey,
                t.args.cullMode,
                1,
            );
        return;
    }
    const { vertex, singleFragment, alphaPipelineKey, cullMode, primitive, name } = t.args;
    const transparent =
        alphaPipelineKey === AlphaPipelineKey.Blend ||
        alphaPipelineKey === AlphaPipelineKey.Premultiplied ||
        alphaPipelineKey === AlphaPipelineKey.Multiply;
    const blend =
        alphaPipelineKey === AlphaPipelineKey.Blend
            ? ALPHA_BLEND
            : alphaPipelineKey === AlphaPipelineKey.Premultiplied
              ? PREMULTIPLIED_BLEND
              : alphaPipelineKey === AlphaPipelineKey.Multiply
                ? MULTIPLY_BLEND
                : undefined;
    const root = materialRoot(world, t.owner, transparent);
    const pipeline = root
        .createRenderPipeline({
            vertex,
            fragment: singleFragment,
            targets: { color: { format: _render.format, ...(blend ? { blend } : {}) } },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: !transparent,
                depthCompare: transparent ? "greater-equal" : "greater",
            },
            multisample: { count: 1 },
        })
        .$name(`standard-${name}-1x`);
    t.single = { color: transparent ? null : pipeline, transparent: transparent ? pipeline : null };
    if (t.prepass) t.prepassSingle = compilePrepass(world, t.owner, alphaPipelineKey, cullMode, 1);
}

/**
 * Creates the AlphaToCoverage variant a 4× MSAA camera stacked over another draws, once: the same
 * vertex and material fragment writing {@link coverageMask} instead of hardware alphaToCoverage.
 * A lone or bottom camera never creates it.
 */
export function ensureStacked(world: World, t: CompiledMaterial): TgpuRenderPipeline<any> {
    if (t.stacked) return t.stacked;
    const { vertex, primitive, name, cullMode } = t.args;
    const surface = t.owner;
    const hasVaryings = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const fragment = (
        hasVaryings
            ? varyingFs(surface, false, false, true, cullMode)
            : colorFs(surface, false, false, true, cullMode)
    ) as ReturnType<typeof colorFs>;
    t.stacked = materialRoot(world, surface)
        .createRenderPipeline({
            vertex,
            fragment,
            targets: { color: { format: world.resource(RenderContext).format } },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "greater",
            },
            multisample: { count: SAMPLE_COUNT },
        })
        .$name(`standard-${name}-stacked`);
    return t.stacked;
}

// Every pipeline recorded into a render bundle has a fragment stage: WebKit rejects a vertex-only pipeline
// in a bundle encoder (`setPipeline: validation failed`), though it accepts one in a pass and the spec
// allows both. The opaque prepass writes depth alone, so its stage is empty.
const depthOnlyFs = tgpu
    .fragmentFn({ in: { pos: d.builtin.position }, out: d.Void })(() => {
        "use gpu";
    })
    .$name("prepassDepthOnlyFs");

/** Opaque uses the compact depth stream unless normals or a vertex function need the main stream; clipped inputs retain material alpha.
 * Every depth pipeline shades under {@link recordLitNormal}; normals write the normal it records. */
function compilePrepass(
    world: World,
    surface: AnyMaterialType,
    alphaPipelineKey: number,
    cullMode: number,
    sampleCount = SAMPLE_COUNT,
    normals = false,
): TgpuRenderPipeline<any> | null {
    if (
        alphaPipelineKey === AlphaPipelineKey.Blend ||
        alphaPipelineKey === AlphaPipelineKey.Premultiplied ||
        alphaPipelineKey === AlphaPipelineKey.Multiply
    )
        return null;
    const primitive = materialPrimitive(cullMode);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    const root = materialRoot(world, surface).with(lightingSlot, recordLitNormal);
    const clip = alphaPipelineKey !== AlphaPipelineKey.Opaque;
    const varying = Object.keys(surface.varyings ?? {}).length > 0;
    const depthOnly = root
        .createRenderPipeline({
            // Depth alone reads no varyings, so a vertex function's opaque depth takes the main
            // stream without them.
            vertex:
                clip || normals
                    ? varying
                        ? varyingVs(surface, true)
                        : colorVs(surface, true)
                    : surface.vertex
                      ? colorVs(surface, true)
                      : prepassVs(surface),
            fragment: clip
                ? ((varying
                      ? varyingClipFs(surface, normals, true, cullMode)
                      : clipFs(surface, normals, true, cullMode)) as never)
                : normals
                  ? ((varying
                        ? varyingClipFs(surface, true, false, cullMode)
                        : clipFs(surface, true, false, cullMode)) as never)
                  : depthOnlyFs,
            targets: (normals ? { color: { format: NORMAL_FORMAT } } : undefined) as never,
            primitive,
            depthStencil,
            multisample: { count: sampleCount },
        })
        .$name(
            `standard-prepass-${variantName(surface, alphaPipelineKey, cullMode)}-${sampleCount}x`,
        );
    return depthOnly;
}

// Tile bounds arrive in pixels with atlas scaling already applied by the VS, so one seam-discard
// fragment serves both atlas types at any resolution. Clipped materials use their own fragment
// instead so material cutouts also remove atlas depth.
const shadowFs = tgpu
    .fragmentFn({
        in: { pos: d.builtin.position, tileBox: d.interpolate("flat", d.vec4f) },
        out: d.Void,
    })((input) => {
        "use gpu";
        const p = input.pos.xy;
        const mn = input.tileBox.xy;
        const sz = input.tileBox.z;
        if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
            std.discard();
        }
    })
    .$name("shadowAtlasFs");

// Only instanced surfaces reach this entry: eids + globalTransforms provide the members to re-gather.
// The eids payload carries (eid, transform row, encoded mesh-instance slot, combo index).
// Point tile rects use casterSlot*6+face; cascade rects use the original cascade slot.
// comboMeta.z supplies the live atlas side, so changing shadow settings needs no pipeline recompilation.
function shadowVs(
    surface: AnyMaterialType,
    shadowGroup: TgpuBindGroupLayout<any>,
    cascade: boolean,
    _capacity: number,
) {
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
    };
    const shadowBound = shadowGroup.$ as unknown as {
        faceVP: { m: any[] };
        comboMeta: { m: any[] };
        tileRects: { rects: any[] };
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position, tileBox: d.interpolate("flat", d.vec4f) },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const instance = bound.eids[input.iid];
            const combo = instance.w;
            const xform = Xform(bound.globalTransforms[instance.y]);
            const world = d.vec4f(xformPoint(xform, localPos), 1);
            const m = shadowBound.comboMeta.m[combo];
            const rect = cascade
                ? shadowBound.tileRects.rects[m.x]
                : shadowBound.tileRects.rects[m.x * 6 + m.y];
            const clip = std.mul(shadowBound.faceVP.m[combo], world);
            const side = d.f32(m.z);
            const tileBox = d.vec4f(std.mul(side, rect.xy), rect.z * side, 0);
            return { pos: clip, tileBox };
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}Vs`);
}

const ClipShadowVertex = d
    .struct({
        pos: d.vec4f,
        tileBox: d.vec4f,
        worldNormal: d.vec3f,
        eid: d.u32,
        row: d.u32,
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        material: d.u32,
        alphaCutoff: d.f32,
    })
    .$name("ClipShadowVertex");

function clipShadowVertex(
    surface: AnyMaterialType,
    shadowGroup: TgpuBindGroupLayout<any>,
    cascade: boolean,
    _capacity: number,
) {
    const hasVs = !!surface.vertex;
    const vsFn = materialVertexFunction(surface);
    const layout = surface.layout;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    const shadowBound = shadowGroup.$ as unknown as {
        faceVP: { m: any[] };
        comboMeta: { m: any[] };
        tileRects: { rects: any[] };
    };
    return tgpu
        .fn(
            [d.u32, d.u32],
            ClipShadowVertex,
        )((vidx, iid) => {
            "use gpu";
            const v = layout.$.vertices[vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            const instance = bound.eids[iid];
            const eid = instance.x;
            const row = producerRow(instance);
            const combo = instance.w;
            let material = d.u32(0);
            let alphaCutoff = d.f32(0.5);
            const encodedMeshInstance = instance.z;
            if (encodedMeshInstance !== 0) {
                const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                material = meshInstance.material;
                alphaCutoff = meshInstance.alphaCutoff;
            }
            const xform = Xform(bound.globalTransforms[instance.y]);
            let world = d.vec4f(xformPoint(xform, localPos), 1);
            let worldNormal = d.vec3f(xformNormal(xform, localNormal));
            if (hasVs) {
                const patched = vsFn!(
                    MaterialVertexInput({
                        localPos,
                        localNormal,
                        uv,
                        vidx,
                        eid,
                        row,
                        iid,
                        xform,
                        world,
                        worldNormal,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            const m = shadowBound.comboMeta.m[combo];
            const rect = cascade
                ? shadowBound.tileRects.rects[m.x]
                : shadowBound.tileRects.rects[m.x * 6 + m.y];
            const side = d.f32(m.z);
            return ClipShadowVertex({
                pos: std.mul(shadowBound.faceVP.m[combo], world),
                tileBox: d.vec4f(std.mul(side, rect.xy), rect.z * side, 0),
                worldNormal: std.normalize(worldNormal),
                eid,
                row,
                world: world.xyz,
                uv,
                localPos,
                material: mirroredMaterial(material, xform.scale),
                alphaCutoff,
            });
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVertex`);
}

function clipShadowVs(
    surface: AnyMaterialType,
    shadowGroup: TgpuBindGroupLayout<any>,
    cascade: boolean,
    capacity: number,
) {
    const vertex = clipShadowVertex(surface, shadowGroup, cascade, capacity);
    const name = `${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`;
    const input = { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex };
    const fixed = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
    };
    const uv = !!surface.fragmentInputs?.uv;
    const localPos = !!surface.fragmentInputs?.localPos;
    if (uv && localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    uv: v.uv,
                    localPos: v.localPos,
                    material: v.material,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    if (uv) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    uv: v.uv,
                    material: v.material,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    if (localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    row: v.row,
                    world: v.world,
                    localPos: v.localPos,
                    material: v.material,
                    alphaCutoff: v.alphaCutoff,
                };
            })
            .$name(name);
    }
    return tgpu
        .vertexFn({ in: input, out: fixed })((i) => {
            "use gpu";
            const v = vertex(i.vidx, i.iid);
            return {
                pos: v.pos,
                tileBox: v.tileBox,
                worldNormal: v.worldNormal,
                eid: v.eid,
                row: v.row,
                world: v.world,
                material: v.material,
                alphaCutoff: v.alphaCutoff,
            };
        })
        .$name(name);
}

/** Projects authored vertex outputs into the shadow atlas at explicit varying locations. */
function varyingShadowVs(
    surface: AnyMaterialType,
    shadowGroup: TgpuBindGroupLayout<any>,
    cascade: boolean,
    _capacity: number,
) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (!surface.vertex) {
        throw new Error(
            `standard: surface "${surface.name}" declares varyings with no vertex function`,
        );
    }
    const hasVs = !!surface.vertex;
    const fragmentFields = fragmentInterstage(surface);
    const layout = surface.layout;
    // see varyingVs's why: `layout.$.x` throws outside codegen mode (including inside `tgpu.lazy`),
    // so the whole `$` proxy rides as one external and the WGSL text's own dot chain defers the field read.
    const bound = layout.$;
    const shadow = shadowGroup.$;
    const Out = d
        .struct({
            pos: d.vec4f,
            tileBox: d.vec4f,
            worldNormal: d.vec3f,
            eid: d.u32,
            row: d.u32,
            world: d.vec3f,
            material: d.u32,
            alphaCutoff: d.f32,
            ...fragmentFields,
            ...varyings,
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipOut`);
    const assigns = keys.map((key) => `    out.${key} = patched.${key};`).join("\n");
    const fragmentAssigns = `${surface.fragmentInputs?.uv ? "    out.uv = uv;\n" : ""}${surface.fragmentInputs?.localPos ? "    out.localPos = localPos;\n" : ""}`;
    const rect = cascade ? "m.x" : "m.x * 6u + m.y";
    const copier = tgpu
        .fn(
            [d.u32, d.u32],
            Out,
        )(/* wgsl */ `(vidx: u32, iid: u32) -> Out {
    let v = bound.vertices[vidx];
    let mq = engine.meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    let instance = bound.eids[iid];
    let eid = instance.x;
    let row = producerRow(instance);
    let combo = instance.w;
    let encodedMeshInstance = instance.z;
    var material: u32 = 0u;
    var alphaCutoff: f32 = 0.5;
    if (encodedMeshInstance != 0u) {
        let meshInstance = bound.meshInstances[encodedMeshInstance - 1u];
        material = meshInstance.material;
        alphaCutoff = meshInstance.alphaCutoff;
    }
    let xform = bound.globalTransforms[instance.y];
    var world = vec4f(xformPoint(xform, localPos), 1.0);
    var worldNormal = vec3f(xformNormal(xform, localNormal));
${
    hasVs
        ? `    let patched = vs(MaterialVertexInput(localPos, localNormal, uv, vidx, eid, row, iid, xform, world, worldNormal, material));
    world = patched.world;
    worldNormal = patched.worldNormal;
`
        : ""
}
    let m = shadow.comboMeta.m[combo];
    let rect = shadow.tileRects.rects[${rect}];
    var out: Out;
    out.pos = shadow.faceVP.m[combo] * world;
    out.tileBox = vec4f(f32(m.z) * rect.xy, rect.z * f32(m.z), 0.0);
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.row = row;
    out.world = world.xyz;
    out.material = mirroredMaterial(material, xform.scale);
    out.alphaCutoff = alphaCutoff;
${fragmentAssigns}
${assigns}
    return out;
}`)
        .$uses({
            Out,
            bound,
            engine: engineLayout.$,
            shadow,
            decodePos,
            decodeUv,
            meshIdOf,
            octDecodeNormal,
            xformPoint,
            xformNormal,
            producerRow,
            mirroredMaterial,
            ...(hasVs ? { MaterialVertexInput, vs: materialVertexFunction(surface) } : {}),
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipCopier`);
    const located = Object.fromEntries(
        Object.entries(varyings).map(([key, schema], i) => [
            key,
            d.location(VARYING_BASE + i, schema),
        ]),
    );
    const outputFields = [
        "pos",
        "tileBox",
        "worldNormal",
        "eid",
        "row",
        "world",
        "material",
        "alphaCutoff",
        ...Object.keys(fragmentFields),
        ...keys,
    ];
    const outputAssignments = outputFields
        .map((key) => `    out.${key} = copied.${key};`)
        .join("\n");
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                tileBox: d.interpolate("flat", d.vec4f),
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                row: d.interpolate("flat", d.u32),
                world: d.vec3f,
                material: d.interpolate("flat", d.u32),
                alphaCutoff: d.interpolate("flat", d.f32),
                ...fragmentFields,
                ...located,
            },
        })(/* wgsl */ `{
    let copied = copier(in.vidx, in.iid);
    var out: Out;
${outputAssignments}
    return out;
}`)
        .$uses({ copier })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`);
}

function varyingShadowFs(surface: AnyMaterialType, cullMode: number) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    const schemas = keys.map((key) => varyings[key]);
    const slots = keys.map((_, i) => `v${i}`);
    const located = Object.fromEntries(
        slots.map((slot, i) => [slot, d.location(VARYING_BASE + i, schemas[i])]),
    );
    const fragment = materialFragmentFunction(surface);
    const Ctx = (fragment as any).shell.argTypes[0];
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const entryIn = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        material: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        ...facingInput(cullMode),
        ...located,
    };
    return tgpu
        .fragmentFn({
            in: entryIn,
            out: d.Void,
        })((input: any) => {
            "use gpu";
            const p = input.pos.xy;
            const mn = input.tileBox.xy;
            const sz = input.tileBox.z;
            if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
                std.discard();
            }
            const ctx = Ctx();
            ctx.eid = input.eid;
            ctx.row = input.row;
            ctx.world = d.vec3f(input.world);
            ctx.worldNormal = std.normalize(input.worldNormal);
            ctx.uv = needUv ? d.vec2f((input as any).uv) : d.vec2f(0);
            ctx.localPos = needLocalPos ? d.vec3f((input as any).localPos) : d.vec3f(0);
            ctx.material = materialRow(input.material);
            ctx.alphaMode = d.u32(0);
            ctx.alphaCutoff = input.alphaCutoff;
            ctx.frontFacing = logicalFrontFacing(
                readsFacing ? input.frontFacing : facing,
                input.material,
            );
            ctx.fragCoord = d.vec4f(input.pos);
            ctx.prepass = true;
            for (const i of tgpu.unroll(std.range(keys.length))) {
                (ctx as any)[keys[i]] = (schemas[i] as any)((input as any)[slots[i]]);
            }
            const color = fragment(ctx);
            const cutoff = input.alphaCutoff;
            if (color.a < cutoff) std.discard();
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/** the clipped atlas fragment: preserve the tile-seam discard, then call the surface fs solely for its
 * cutoff discard. Its color result is intentionally ignored by this depth-only pipeline. */
function clipShadowFs(surface: AnyMaterialType, cullMode: number) {
    const readsFacing = cullMode === CullModeCode.None;
    const facing = cullMode === CullModeCode.Back;
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const fragment = materialFragmentFunction(surface);
    const Ctx = (fragment as any).shell.argTypes[0];
    return tgpu
        .fragmentFn({
            in: {
                pos: d.builtin.position,
                tileBox: d.interpolate("flat", d.vec4f),
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                row: d.interpolate("flat", d.u32),
                world: d.vec3f,
                material: d.interpolate("flat", d.u32),
                alphaCutoff: d.interpolate("flat", d.f32),
                ...fragmentInterstage(surface),
                ...facingInput(cullMode),
            },
            out: d.Void,
        })((input: any) => {
            "use gpu";
            const p = input.pos.xy;
            const mn = input.tileBox.xy;
            const sz = input.tileBox.z;
            if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
                std.discard();
            }
            const ctx = Ctx({
                eid: input.eid,
                row: input.row,
                world: input.world,
                worldNormal: std.normalize(input.worldNormal),
                uv: needUv ? (input as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (input as any).localPos : d.vec3f(0),
                material: materialRow(input.material),
                alphaMode: d.u32(0),
                alphaCutoff: input.alphaCutoff,
                frontFacing: logicalFrontFacing(
                    readsFacing ? input.frontFacing : facing,
                    input.material,
                ),
                fragCoord: input.pos,
                prepass: true,
            });
            const color = fragment(ctx);
            const cutoff = input.alphaCutoff;
            if (color.a < cutoff) std.discard();
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/**
 * Compile a `MaterialType`'s point and cascade shadow-atlas pipelines when its depth settings enable
 * them. Opaque types share {@link shadowFs}, behind the clipped vertex stage when a vertex function
 * needs the main stream; clipped types use their wider cutoff vertex/fragment pair. Each closes over its own `pointLayout` / `cascadeLayout` group-1 and
 * reads its atlas side from `comboMeta.z`, so neither bakes a shadow setting.
 */
function compileShadow(
    world: World,
    surface: AnyMaterialType,
    capacity: number,
    alphaPipelineKey: number,
    cullMode: number,
): {
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
} {
    if (!isInstanced(surface)) return { point: null, cascade: null };
    const layout =
        surface.vertex || alphaPipelineKey !== AlphaPipelineKey.Opaque
            ? surface.layout
            : surface.layout.depthVariant;
    validateBindingLimits(`material "${surface.name}" point shadow`, [
        engineLayout,
        pointLayout,
        layout,
    ]);
    validateBindingLimits(`material "${surface.name}" cascade shadow`, [
        engineLayout,
        cascadeLayout,
        layout,
    ]);
    const primitive = materialPrimitive(cullMode);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    const root = materialRoot(world, surface).with(lightingSlot, recordLitNormal);
    const clip = alphaPipelineKey !== AlphaPipelineKey.Opaque;
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const point = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, pointLayout, false, capacity)
                    : clipShadowVs(surface, pointLayout, false, capacity)
                : surface.vertex
                  ? clipShadowVs(surface, pointLayout, false, capacity)
                  : shadowVs(surface, pointLayout, false, capacity),
            fragment: clip
                ? ((varying
                      ? varyingShadowFs(surface, cullMode)
                      : clipShadowFs(surface, cullMode)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-point-${variantName(surface, alphaPipelineKey, cullMode)}`);
    const cascade = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, cascadeLayout, true, capacity)
                    : clipShadowVs(surface, cascadeLayout, true, capacity)
                : surface.vertex
                  ? clipShadowVs(surface, cascadeLayout, true, capacity)
                  : shadowVs(surface, cascadeLayout, true, capacity),
            fragment: clip
                ? ((varying
                      ? varyingShadowFs(surface, cullMode)
                      : clipShadowFs(surface, cullMode)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-cascade-${variantName(surface, alphaPipelineKey, cullMode)}`);
    return { point, cascade };
}

/** the compiled pipeline(s) for a `MaterialTypes` entry, or `undefined` until
 * {@link compileMaterial} has run for it. */
export function getCompiledMaterial(
    world: World,
    name: string,
    alphaPipelineKey: number = AlphaPipelineKey.Opaque,
    cullMode: number = CullModeCode.Back,
): CompiledMaterial | undefined {
    return pipelineState(world).compiledMaterials.get(`${name}:${alphaPipelineKey}:${cullMode}`);
}

// A background with only group 0 would drop group 1 when switching pipelines in the same pass.
// Retain shadowLayout so the transparent geometry drawn after the background keeps its shadow bindings.

/** the widest `Background` shape (any bindings) — the bare `Background` default pins `B` to
 *  `Record<string, Binding>` already, so this alias exists only to name the widened form at call
 *  boundaries, matching {@link AnyMaterialType}'s shape. */
type AnyBackground = Background<Record<string, MaterialBinding>>;

function bgFs(bg: AnyBackground) {
    return tgpu
        .fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            const uv = std.div(input.pos.xy, engineLayout.$.view.resolution);
            const ndc = d.vec3f(uv.x * 2 - 1, 1 - uv.y * 2, 0);
            const far = std.mul(engineLayout.$.view.worldFromClip, d.vec4f(ndc, 1));
            let dir = d.vec3f(0);
            if (engineLayout.$.view.projection.z > 0.5) {
                // Infinite reverse-Z maps perspective depth 0 to a homogeneous direction at infinity.
                dir = std.normalize(far.xyz);
            } else {
                // Orthographic depth 0 remains a finite point on the far projection plane.
                dir = std.normalize(std.sub(std.div(far.xyz, far.w), engineLayout.$.view.eye.xyz));
            }
            // Shadow receivers read free WGSL names that TypeGPU's call graph cannot discover.
            // Touch their layout bindings in the returned value: the transpiler removes an unused
            // touch as dead code. Multiplying by zero preserves the background color.
            const forcedZero =
                (shadowLayout.$.pointShadows.casters[0].pos.x +
                    shadowLayout.$.tileRects.rects[0].x +
                    shadowLayout.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.pointAtlas,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.shadowMap,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            const col = bg.fs(BackgroundContext({ dir }));
            return d.vec4f(std.add(col, d.vec3f(forcedZero)), 1);
        })
        .$name(`${bg.name}Fs`);
}

/** a compiled background: the 4× MSAA + single-sample twins (a camera binds whichever its
 *  `Camera.antialias` selects). */
export interface CompiledBackground {
    /** exact registry spec + layout this pipeline/group state was derived from. */
    owner: AnyBackground;
    layout: BackgroundLayout<Record<string, MaterialBinding>>;
    color: TgpuRenderPipeline<d.Vec4f>;
    single: TgpuRenderPipeline<d.Vec4f>;
    // the background's own group-2 bind group, built lazily on first draw (`renderColor`'s
    // backdrop pick) and cached on the resolved resource identities; null for a binding-free background
    // (its empty layout never enters the pipeline layout, so no group is bound at 2)
    group2: { group: GPUBindGroup; resources: BindResource[] } | null;
    // the background's engine group-0 instances per view slot: `bgQuant()`'s shared engineCache
    engineCache: Map<number, GPUBindGroup>;
}

/**
 * Creates both 4x MSAA and single-sample variants eagerly: backgrounds are few and a camera's
 * AA selection is known only at draw time. Depth is not written; reverse-Z greater-equal at
 * clip z=0 admits clear-depth pixels (0 >= 0) but rejects geometry (depth > 0).
 * Caches by recipe identity; backgrounds with the same label keep separate pipelines and groups.
 */
export function compileBackground(world: World, bg: AnyBackground): CompiledBackground {
    const _render = world.resource(RenderContext);

    const cached = pipelineState(world).compiledBackgrounds.get(bg);
    if (cached?.owner === bg && cached.layout === bg.layout) return cached;
    const fragment = bgFs(bg);
    const primitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: "greater-equal",
    };
    const color = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
        })
        .$name(`standard-bg-${bg.name}`);
    const single = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-bg-${bg.name}-1x`);
    const compiled: CompiledBackground = {
        owner: bg,
        layout: bg.layout,
        color,
        single,
        group2: null,
        engineCache: engineCache(world, bgQuant(world)),
    };
    pipelineState(world).compiledBackgrounds.set(bg, compiled);
    return compiled;
}

/** the compiled pipeline(s) for a `Backgrounds` entry, or `undefined` until
 * {@link compileBackground} has run for it. */
export function getBackground(world: World, bg: AnyBackground): CompiledBackground | undefined {
    const compiled = pipelineState(world).compiledBackgrounds.get(bg);
    if (compiled && (compiled.owner !== bg || compiled.layout !== bg.layout)) {
        pipelineState(world).compiledBackgrounds.delete(bg);
        return undefined;
    }
    return compiled;
}

export function ensureNormalPrepass(
    world: World,
    t: CompiledMaterial,
    single: boolean,
): TgpuRenderPipeline<any> | null {
    if (!t.prepass) return null;
    if (single)
        return (t.normalSingle ??= compilePrepass(
            world,
            t.owner,
            t.args.alphaPipelineKey,
            t.args.cullMode,
            1,
            true,
        ));
    return (t.normal ??= compilePrepass(
        world,
        t.owner,
        t.args.alphaPipelineKey,
        t.args.cullMode,
        SAMPLE_COUNT,
        true,
    ));
}

/** Prepare every material parameter table, pipeline and background at warm, before the first draw.
 * `singleSample` also prepares each material's AA-off variant, for a no-AA camera present at warm.
 * `stacked` prepares AlphaToCoverage for an initially stacked multisampled camera.
 * A camera added or reconfigured after warm may still compile a variant on its first draw and warn. */
export async function preparePipelines(
    world: World,
    capacity: number,
    singleSample = false,
    stacked = false,
): Promise<void> {
    // A material may first acquire a draw in the draw group, after that frame's table upload.
    // Create its parameter table at warm so the defaults reach the first frame's upload.
    // TypeGPU defers resolve and device pipeline creation until first use. root.unwrap forces
    // both at warm so resolution/validation errors surface here, not mid-frame on the first draw.
    const normalSamples = new Set<number>();
    const normalRequests = world.resource(NormalPrepassRequests);
    for (const eid of world.query([Camera])) {
        const view = world.resource(Views).get(eid);
        if (
            world.has(eid, NormalPrepass) ||
            world.resource(MaterialTypes).types.some((type) => type?.readsPrepass?.normal) ||
            normalRequests.some((request) => request(world, eid, view))
        ) {
            normalSamples.add(world.storage(Camera).antialias.get(eid) === 0 ? 1 : SAMPLE_COUNT);
        }
    }
    for (const type of world.resource(MaterialTypes).types) {
        if (!type) continue;
        const source = materialSource(world, type);
        source.table;
        const pipelineKeys = source.pipelineKeys;
        for (let key = 0; key < 32; key++) {
            if (!(pipelineKeys & (1 << key))) continue;
            const alphaPipelineKey = key % ALPHA_PIPELINE_KEY_COUNT;
            const cullMode = (key - alphaPipelineKey) / ALPHA_PIPELINE_KEY_COUNT;
            const compiled = compileMaterial(world, type, capacity, alphaPipelineKey, cullMode);
            if (singleSample) ensureSingle(world, compiled);
            for (const samples of normalSamples) {
                const pipeline = ensureNormalPrepass(world, compiled, samples === 1);
                if (pipeline)
                    precompile(world, precompileScope(world, "standard-normal-prepass"), () =>
                        prepareMaterialPipeline(world, type.name, pipeline),
                    );
            }
            if (stacked && alphaPipelineKey === AlphaPipelineKey.AlphaToCoverage) {
                const pipeline = ensureStacked(world, compiled);
                precompile(world, precompileScope(world, "standard-stacked"), () =>
                    prepareMaterialPipeline(world, type.name, pipeline),
                );
            }
            for (const p of [
                compiled.color,
                compiled.transparent,
                compiled.point,
                compiled.cascade,
                compiled.prepass,
                compiled.single?.color,
                compiled.single?.transparent,
                compiled.prepassSingle,
                compiled.stacked,
            ]) {
                if (p) prepareMaterialPipeline(world, type.name, p);
            }
        }
    }
    for (const bg of world.resource(Backgrounds).values()) {
        const cb = compileBackground(world, bg);
        world.gpu.root.unwrap(cb.color);
        world.gpu.root.unwrap(cb.single);
    }
}
