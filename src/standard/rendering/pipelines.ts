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
    DEPTH_FORMAT,
    Frame,
    RenderContext,
    SAMPLE_COUNT,
    Xform,
    xformNormal,
    xformPoint,
} from "../../core/rendering";
import type { World } from "../../engine";
import { octDecodeNormal } from "../../engine/utils";
import { cascadeLayout, pointLayout, shadowLayout } from "./atlas";
import { LightCull } from "./cluster";
import type { Background, BackgroundLayout } from "./contract";
import { BackgroundContext, Backgrounds } from "./contract";
import {
    engineLayout,
    fragCoord,
    fragWorld,
    pointScale,
    pointShadowSlot,
    pointShadowStub,
    sunVisibility,
} from "./engine";
import { foggedColor, fogSlot, fogSource, unfoggedColor } from "./fog";
import type { Recorded } from "./forward";
import { indirectLightSlot, indirectLightSources } from "./indirect";
import { Lighting } from "./lighting";
import type { MaterialBinding, MaterialLayout, MaterialType } from "./material-type";
import {
    ALPHA_PIPELINE_KEY_COUNT,
    AlphaModeCode,
    AlphaPipelineKey,
    MaterialTypes,
    MeshVertexInput as MaterialVertexInput,
    materialSource,
} from "./material-type";
import type { Draw } from "./registry";
import { sampleSunShadow } from "./shade";

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
        return d.vec4f(color);
    })
    .$name("materialAlphaOutput");

const colorOutput = { color: d.vec4f };

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
    // the point/cascade shadow-atlas pipelines. `null` when the type opts out of that shadow route.
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
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
        primitive: GPUPrimitiveState;
        name: string;
    };
}

/**
 * Slot-independent group-2 state, one entry per draw. Color uses the layout's 16-byte main stream;
 * opaque depth uses the distinct layout.depthVariant object and 8-byte position stream, while clipped
 * depth needs main-stream UVs. Atlas groups substitute re-gathered eids.
 * engineCache holds group-0 instances per view slot, shared by draws using the same quant buffer.
 * Atlas passes use its unread slot-zero ViewUniforms placeholder; their VS uses a tile viewProj instead.
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

export function engineGroup(
    world: World,
    cache: Map<number, GPUBindGroup>,
    slot: number,
    quant: GPUBuffer,
): GPUBindGroup {
    const _lightCull = world.resource(LightCull);
    const pointLights = _lightCull.lights;
    if (!pointLights) throw new Error("[render] engine group used before light-cull warm");
    if (engineCacheLights.get(cache) !== pointLights) {
        cache.clear();
        engineCacheLights.set(cache, pointLights);
    }

    const cached = cache.get(slot);
    if (cached) return cached;
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(engineLayout, {
            frame: world.resource(Frame).buffer!,
            view: world.resource(RenderContext).viewBuffers[slot],
            lighting: world.resource(Lighting).buffer!,
            pointLights,
            meshQuant: quant,
        }),
    );
    cache.set(slot, group);
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

// Reserve locations after the fixed fragment fields to prevent custom varying collisions.
const VARYING_BASE = 8;

// Eight fixed locations, uv + localPos and four custom varyings use 14 of WebGPU's default
// 16 inter-stage variables. The vs side is N-general (the
// copier templates over `Object.keys`), so this bound is the fragment entry's — its transpiled body must
// statically name `input.v0`…`input.v3`, one arm per count (`varyingFs`)
const MAX_VARYINGS = 4;

function fragmentInterstage(surface: AnyMaterialType): Record<string, AnyWgslData> {
    return {
        ...(surface.fragmentInputs?.uv ? { uv: d.vec2f } : {}),
        ...(surface.fragmentInputs?.localPos ? { localPos: d.vec3f } : {}),
    };
}

/** Shared triangle-list state for standard material color, depth and atlas pipelines. */
function materialPrimitive(): GPUPrimitiveState {
    return { topology: "triangle-list", cullMode: "back", frontFace: "ccw" };
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
        color: d.vec4f,
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
    return tgpu
        .fn(
            [Input],
            d.vec4f,
        )(`(input: Input) -> vec4f {
        return fragment(Context(bound.materialParameters[input.material], ${Object.keys(fields)
            .map((key) => `input.${key}`)
            .join(", ")}));
    }`)
        .$uses({ Input, Context, fragment, bound });
}

function materialVertexFunction(surface: AnyMaterialType) {
    const vertex = surface.vertex;
    if (!vertex) return undefined;
    const shell = (vertex as any).shell;
    const Input = shell.argTypes[0];
    const layout = surface.layout;
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
                color: input.color,
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
            const color = d.vec4f(1);
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
                        color,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            pos = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            if (clip) pos = d.vec4f(std.add(pos, d.vec4f(shadowLayout.$.tileRects.rects[0].x * 0)));
            return MaterialTypeVertex({
                pos,
                worldNormal: std.normalize(worldNormal),
                eid,
                row,
                world: world.xyz,
                uv,
                localPos,
                color,
                material,
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
        color: d.vec4f,
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
                    color: v.color,
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
                    color: v.color,
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
                    color: v.color,
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
                color: v.color,
                material: v.material,
                alphaMode: v.alphaMode,
                alphaCutoff: v.alphaCutoff,
            };
        })
        .$name(name);
}

function colorFs(surface: AnyMaterialType, alphaToCoverageFallback = false, fogged = false) {
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
        color: d.vec4f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
    };
    return tgpu
        .fragmentFn({
            in: inputSchema,
            out: colorOutput,
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
                color: input.color,
                material: input.material,
                alphaMode: alphaToCoverageFallback ? AlphaModeCode.Opaque : input.alphaMode,
                alphaCutoff: input.alphaCutoff,
            } as any);
            const col = d.vec4f(std.add(fragment(ctx), d.vec4f(forcedZero)));
            const output = alphaColorOutput(
                fogColor(col, input.world, input.alphaMode),
                input.alphaMode,
                input.alphaCutoff,
                alphaToCoverageFallback ? 0xffffffff : 0,
            );
            return {
                color: output,
            };
        })
        .$name(`${surface.name}${alphaToCoverageFallback ? "Single" : ""}Fs`);
}

// Host surface objects cannot cross a TGSL helper boundary; this entry closes over its depth layout.
function prepassVs(surface: AnyMaterialType) {
    const instanced = isInstanced(surface);
    const hasVs = !!surface.vertex;
    const vsFn = materialVertexFunction(surface);
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // the depth-only default, pinned
            // for the life of the vs, never touched by the instance transform below; a `vs` chunk reading `vsIn.localNormal` sees this default,
            // not the transformed `worldNormal`
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            let row = d.u32(0xffffffff);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            const color = d.vec4f(1);
            let material = d.u32(0);
            let xform = identityXform();
            if (instanced) {
                const instance = bound.eids[input.iid];
                eid = instance.x;
                row = producerRow(instance);
                const encodedMeshInstance = instance.z;
                if (encodedMeshInstance !== 0) {
                    const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                    material = meshInstance.material;
                }
                xform = Xform(bound.globalTransforms[instance.y]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let clip = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    MaterialVertexInput({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        row,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                        color,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            // TypeGPU indexes layouts by group number; using only groups 0 and 2 leaves a hole
            // that Dawn rejects at pipeline creation. The returned zero keeps group 1 live;
            // renderPrepass binds shadowGroup there, but the stub receiver never samples it.
            const forcedZero = shadowLayout.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)) };
        })
        .$name(`${surface.name}PrepassVs`);
}

function clipFs(surface: AnyMaterialType) {
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const fragment = materialFragmentFunction(surface);
    const Ctx = (fragment as any).shell.argTypes[0];
    const input = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
    };
    return tgpu
        .fragmentFn({ in: input, out: d.Void })((fin) => {
            "use gpu";
            const ctx = Ctx({
                eid: fin.eid,
                row: fin.row,
                world: fin.world,
                worldNormal: std.normalize(fin.worldNormal),
                uv: needUv ? (fin as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (fin as any).localPos : d.vec3f(0),
                color: fin.color,
                material: fin.material,
                alphaMode: d.u32(0),
                alphaCutoff: fin.alphaCutoff,
            });
            const color = fragment(ctx);
            const cutoff = fin.alphaCutoff;
            if (color.a < cutoff) std.discard();
        })
        .$name(`${surface.name}ClipFs`);
}

/** Build the locked one-varying cutoff copier: the fragment entry exposes a fixed `v0` slot while this
 * raw helper reconstructs the author's exact FsCtx positionally. The schema comes from the authored fs
 * shell, never a freshly minted lookalike. */
function clipVaryingCopier(surface: AnyMaterialType) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1) {
        throw new Error(
            `standard: surface "${surface.name}" declares ${keys.length} varyings — the clip copier carries exactly one custom varying`,
        );
    }
    const varyingSchema = varyings[keys[0]];
    const varyingType = (varyingSchema as unknown as { type: string }).type;
    const fsFn = materialFragmentFunction(surface);
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec3f, d.u32, d.u32, d.vec3f, d.vec2f, d.vec3f, d.vec4f, d.u32, varyingSchema],
            d.vec4f,
        )(/* wgsl */ `(worldNormalIn: vec3f, eid: u32, row: u32, world: vec3f, uv: vec2f, localPos: vec3f, color: vec4f, material: u32, v0: ${varyingType}) -> vec4f {
    let ctx = Ctx(eid, row, world, normalize(worldNormalIn), uv, localPos, color, material, 0u, 0.5f, v0);
    return fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}ClipFsCopier`);
    return { varyingSchema, copier };
}

function varyingClipFs(surface: AnyMaterialType) {
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const entryIn = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        v0: d.location(VARYING_BASE, varyingSchema as d.Vec3f),
    };
    return tgpu
        .fragmentFn({
            in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
            out: d.Void,
        })((input) => {
            "use gpu";
            const color = copier(
                input.worldNormal,
                input.eid,
                input.row,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
            );
            const cutoff = input.alphaCutoff;
            if (color.a < cutoff) std.discard();
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
                color: d.vec4f,
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
    var color = vec4f(1.0);
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
        ? `    let patched = vs(MaterialVertexInput(localPos, localNormal, uv, vidx, eid, row, iid, xform, world, worldNormal, color, material));
    world = patched.world;
    worldNormal = patched.worldNormal;
`
        : ""
}
    var out: Out;
    out.pos = engine.view.viewProj * world${clip ? " + vec4f(shadowG.tileRects.rects[0].x * 0.0)" : ""};
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.row = row;
    out.world = world.xyz;
    out.color = color;
    out.material = material;
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
// touches as dead code. Sharing the function keeps the four varyingFs arities in agreement.
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

// TGSL cannot pass its entry-input router by value or read dynamically named input fields.
// Each arity names v0..v3 statically and passes values in declaration order to a positional
// WGSL context constructor, which puts them in the author's actual varying fields.
// The 1..MAX_VARYINGS arms bound the static entry; the copier signature and vertex side already
// accept arbitrary counts.
// Reuse fsFn.shell.argTypes[0]: WGSL struct arguments are not typed purely structurally.
function varyingFs(surface: AnyMaterialType, alphaToCoverageFallback = false, fogged = false) {
    const fogColor = fogged ? foggedColor : unfoggedColor;
    const varyings = surface.varyings ?? {};
    const varyingKeys = Object.keys(varyings);
    if (varyingKeys.length < 1 || varyingKeys.length > MAX_VARYINGS) {
        throw new Error(
            `standard: surface "${surface.name}" declares ${varyingKeys.length} varyings — the fs entry carries 1 to ${MAX_VARYINGS} (gpu.md rule 9's custom interpolator budget)`,
        );
    }
    const schemas = varyingKeys.map((k) => varyings[k]);
    const params = schemas
        .map((s, i) => `v${i}: ${(s as unknown as { type: string }).type}`)
        .join(", ");
    const fsFn = materialFragmentFunction(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec4f, d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, d.vec4f, d.u32, ...schemas, d.u32],
            d.vec4f,
        )(/* wgsl */ `(pos: vec4f, worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, color: vec4f, material: u32, ${params}, row: u32) -> vec4f {
    let worldNormal = normalize(worldNormalIn);
    let ctx = Ctx(eid, row, world, worldNormal, uv, localPos, color, material, 0u, 0.5f, ${schemas.map((_, i) => `v${i}`).join(", ")});
    return fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}FsCopier`);

    // The runtime schema is concrete, but its widened type cannot satisfy fragmentFn's input constraint.
    // Locations pair fixed FS names with author-named VS fields. No flat interpolation is inserted,
    // so integer varyings are unsupported and fail at resolve or device compilation.
    const slot = (i: number) => d.location(VARYING_BASE + i, schemas[i] as d.Vec3f);
    const base = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.u32),
        alphaMode: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
    };
    const name = `${surface.name}${alphaToCoverageFallback ? "Single" : ""}Fs`;
    if (varyingKeys.length === 1) {
        const entryIn = { ...base, v0: slot(0) } as unknown as typeof base & { v0: d.Vec3f };
        return tgpu
            .fragmentFn({ in: entryIn, out: colorOutput })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    input.v0,
                    input.row,
                );
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
                return {
                    color: output,
                };
            })
            .$name(name);
    }
    if (varyingKeys.length === 2) {
        const entryIn = { ...base, v0: slot(0), v1: slot(1) } as unknown as typeof base & {
            v0: d.Vec3f;
            v1: d.Vec3f;
        };
        return tgpu
            .fragmentFn({ in: entryIn, out: colorOutput })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    input.v0,
                    input.v1,
                    input.row,
                );
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
                return {
                    color: output,
                };
            })
            .$name(name);
    }
    if (varyingKeys.length === 3) {
        const entryIn = {
            ...base,
            v0: slot(0),
            v1: slot(1),
            v2: slot(2),
        } as unknown as typeof base & {
            v0: d.Vec3f;
            v1: d.Vec3f;
            v2: d.Vec3f;
        };
        return tgpu
            .fragmentFn({ in: entryIn, out: colorOutput })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    input.v0,
                    input.v1,
                    input.v2,
                    input.row,
                );
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
                return {
                    color: output,
                };
            })
            .$name(name);
    }
    const entryIn = {
        ...base,
        v0: slot(0),
        v1: slot(1),
        v2: slot(2),
        v3: slot(3),
    } as unknown as typeof base & { v0: d.Vec3f; v1: d.Vec3f; v2: d.Vec3f; v3: d.Vec3f };
    return tgpu
        .fragmentFn({ in: entryIn, out: colorOutput })((input) => {
            "use gpu";
            const col = copier(
                input.pos,
                input.worldNormal,
                input.eid,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
                input.v1,
                input.v2,
                input.v3,
                input.row,
            );
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
            return {
                color: output,
            };
        })
        .$name(name);
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
): CompiledMaterial {
    const _render = world.resource(RenderContext);
    const cacheKey = `${surface.name}:${alphaPipelineKey}`;
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
        hasVaryings ? varyingFs(surface, false, fogged) : colorFs(surface, false, fogged)
    ) as ReturnType<typeof colorFs>;
    const singleFragment = (
        hasVaryings ? varyingFs(surface, true, fogged) : colorFs(surface, true, fogged)
    ) as ReturnType<typeof colorFs>;
    const name = `${surface.name}-${alphaPipelineKey}`;
    const primitive = materialPrimitive();
    const args: CompiledMaterial["args"] = {
        vertex,
        fragment,
        singleFragment,
        alphaPipelineKey,
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
    let root = world.gpu.root.with(indirectLightSlot, indirectLightSources(world));
    const fog = fogSource(world);
    if (fogged && fog) root = root.with(fogSlot, fog.source);
    const pipeline = root.createRenderPipeline({
        vertex,
        fragment,
        targets: { color: { format: _render.format, ...(blend ? { blend } : {}) } },
        primitive,
        depthStencil,
        multisample: {
            count: SAMPLE_COUNT,
            alphaToCoverageEnabled: alphaPipelineKey === AlphaPipelineKey.AlphaToCoverage,
        },
    });
    const compiled: CompiledMaterial = {
        owner: surface as AnyMaterialType,
        layout: surface.layout as MaterialLayout<Record<string, MaterialBinding>, AnyWgslStruct>,
        color: transparent ? null : pipeline,
        transparent: transparent ? pipeline : null,
        prepass:
            surface.depthPass?.prepass === false
                ? null
                : compilePrepass(world, surface, alphaPipelineKey),
        prepassSingle: null,
        point: null,
        cascade: null,
        single: null,
        args,
    };
    if (surface.depthPass?.shadows !== false && !transparent) {
        const shadows = compileShadow(world, surface, capacity, alphaPipelineKey);
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
            t.prepassSingle = compilePrepass(world, t.owner, t.args.alphaPipelineKey, 1);
        return;
    }
    const { vertex, singleFragment, alphaPipelineKey, primitive, name } = t.args;
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
    let root = world.gpu.root.with(indirectLightSlot, indirectLightSources(world));
    const fog = fogSource(world);
    if (transparent && fog) root = root.with(fogSlot, fog.source);
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
    if (t.prepass) t.prepassSingle = compilePrepass(world, t.owner, alphaPipelineKey, 1);
}

/** Opaque uses the compact depth stream; Mask and AlphaToCoverage use alpha-tested material inputs. */
function compilePrepass(
    world: World,
    surface: AnyMaterialType,
    alphaPipelineKey: number,
    sampleCount = SAMPLE_COUNT,
): TgpuRenderPipeline<any> | null {
    if (
        alphaPipelineKey === AlphaPipelineKey.Blend ||
        alphaPipelineKey === AlphaPipelineKey.Premultiplied ||
        alphaPipelineKey === AlphaPipelineKey.Multiply
    )
        return null;
    const primitive = materialPrimitive();
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    const root = world.gpu.root
        .with(pointShadowSlot, pointShadowStub)
        .with(indirectLightSlot, indirectLightSources(world));
    const clip = alphaPipelineKey !== AlphaPipelineKey.Opaque;
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const depthOnly = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingVs(surface, true)
                    : colorVs(surface, true)
                : prepassVs(surface),
            ...(clip
                ? {
                      fragment: (varying ? varyingClipFs(surface) : clipFs(surface)) as never,
                  }
                : {}),
            primitive,
            depthStencil,
            multisample: { count: sampleCount },
        })
        .$name(`standard-prepass-${surface.name}-${alphaPipelineKey}-${sampleCount}x`);
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
    const hasVs = !!surface.vertex;
    const vsFn = materialVertexFunction(surface);
    const layout = surface.layout.depthVariant;
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
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position, tileBox: d.interpolate("flat", d.vec4f) },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // the depth-only default — never touched by the instance transform, matching
            // `prepassVs`'s pinned law
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            const instance = bound.eids[input.iid];
            const eid = instance.x;
            const row = producerRow(instance);
            const combo = instance.w;
            const color = d.vec4f(1);
            let material = d.u32(0);
            const encodedMeshInstance = instance.z;
            if (encodedMeshInstance !== 0) {
                const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                material = meshInstance.material;
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
                        vidx: input.vidx,
                        eid,
                        row,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                        color,
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
        color: d.vec4f,
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
            const color = d.vec4f(1);
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
                        color,
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
                color,
                material,
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
        color: d.vec4f,
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
                    color: v.color,
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
                    color: v.color,
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
                    color: v.color,
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
                color: v.color,
                material: v.material,
                alphaCutoff: v.alphaCutoff,
            };
        })
        .$name(name);
}

/** The atlas-projecting half of the locked one-varying clip mechanism. The raw entry copies each
 * authored field by name into its own output struct; the fragment receives it as fixed `v0` at the
 * same explicit location. */
function varyingShadowVs(
    surface: AnyMaterialType,
    shadowGroup: TgpuBindGroupLayout<any>,
    cascade: boolean,
    _capacity: number,
) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1 || !surface.vertex) {
        throw new Error(
            `standard: surface "${surface.name}" needs at most one authored varying for its clip shadow copier`,
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
            color: d.vec4f,
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
    var color = vec4f(1.0);
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
        ? `    let patched = vs(MaterialVertexInput(localPos, localNormal, uv, vidx, eid, row, iid, xform, world, worldNormal, color, material));
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
    out.color = color;
    out.material = material;
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
        "color",
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
                color: d.vec4f,
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

function varyingShadowFs(surface: AnyMaterialType) {
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const entryIn = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        row: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.u32),
        alphaCutoff: d.interpolate("flat", d.f32),
        ...fragmentInterstage(surface),
        v0: d.location(VARYING_BASE, varyingSchema as d.Vec3f),
    };
    return tgpu
        .fragmentFn({
            in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
            out: d.Void,
        })((input) => {
            "use gpu";
            const p = input.pos.xy;
            const mn = input.tileBox.xy;
            const sz = input.tileBox.z;
            if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
                std.discard();
            }
            const color = copier(
                input.worldNormal,
                input.eid,
                input.row,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
            );
            const cutoff = input.alphaCutoff;
            if (color.a < cutoff) std.discard();
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/** the clipped atlas fragment: preserve the tile-seam discard, then call the surface fs solely for its
 * cutoff discard. Its color result is intentionally ignored by this depth-only pipeline. */
function clipShadowFs(surface: AnyMaterialType) {
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
                color: d.vec4f,
                material: d.interpolate("flat", d.u32),
                alphaCutoff: d.interpolate("flat", d.f32),
                ...fragmentInterstage(surface),
            },
            out: d.Void,
        })((input) => {
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
                color: input.color,
                material: input.material,
                alphaMode: d.u32(0),
                alphaCutoff: input.alphaCutoff,
            });
            const color = fragment(ctx);
            const cutoff = input.alphaCutoff;
            if (color.a < cutoff) std.discard();
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/**
 * Compile a `MaterialType`'s point and cascade shadow-atlas pipelines when its depth settings enable
 * them. Opaque types share {@link shadowFs}; clipped types use their wider cutoff
 * vertex/fragment pair. Each closes over its own `pointLayout` / `cascadeLayout` group-1 and
 * reads its atlas side from `comboMeta.z`, so neither bakes a shadow setting.
 */
function compileShadow(
    world: World,
    surface: AnyMaterialType,
    capacity: number,
    alphaPipelineKey: number,
): {
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
} {
    if (!isInstanced(surface)) return { point: null, cascade: null };
    const primitive = materialPrimitive();
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // A real receiver would sample the atlas this pipeline is writing.
    const root = world.gpu.root
        .with(pointShadowSlot, pointShadowStub)
        .with(indirectLightSlot, indirectLightSources(world));
    const clip = alphaPipelineKey !== AlphaPipelineKey.Opaque;
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const point = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, pointLayout, false, capacity)
                    : clipShadowVs(surface, pointLayout, false, capacity)
                : shadowVs(surface, pointLayout, false, capacity),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-point-${surface.name}-${alphaPipelineKey}`);
    const cascade = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, cascadeLayout, true, capacity)
                    : clipShadowVs(surface, cascadeLayout, true, capacity)
                : shadowVs(surface, cascadeLayout, true, capacity),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-cascade-${surface.name}-${alphaPipelineKey}`);
    return { point, cascade };
}

/** the compiled pipeline(s) for a `MaterialTypes` entry, or `undefined` until
 * {@link compileMaterial} has run for it. */
export function getCompiledMaterial(
    world: World,
    name: string,
    alphaPipelineKey: number = AlphaPipelineKey.Opaque,
): CompiledMaterial | undefined {
    return pipelineState(world).compiledMaterials.get(`${name}:${alphaPipelineKey}`);
}

// A background with only group 0 would drop group 1 when switching pipelines in the same pass.
// Retain shadowLayout so the transparent geometry drawn after the background keeps its shadow bindings.

/** the widest `Background` shape (any bindings) — the bare `Background` default pins `B` to
 *  `Record<string, Binding>` already, so this alias exists only to name the widened form at call
 *  boundaries, matching {@link AnyMaterialType}'s shape. */
type AnyBackground = Background<Record<string, MaterialBinding>>;

// Reverse-Z far-plane depth lets the background fill only pixels without geometry.
const bgVs = tgpu
    .vertexFn({ in: { vidx: d.builtin.vertexIndex }, out: { pos: d.builtin.position } })(
        (input) => {
            "use gpu";
            const c = d.vec2f(d.f32((input.vidx << 1) & 2), d.f32(input.vidx & 2));
            return { pos: d.vec4f(c.x * 2 - 1, c.y * 2 - 1, 0, 1) };
        },
    )
    .$name("bgVs");

function bgFs(bg: AnyBackground) {
    return tgpu
        .fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            const uv = std.div(input.pos.xy, engineLayout.$.view.resolution);
            const ndc = d.vec3f(uv.x * 2 - 1, 1 - uv.y * 2, 0);
            const far = std.mul(engineLayout.$.view.invViewProj, d.vec4f(ndc, 1));
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
            vertex: bgVs,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
        })
        .$name(`standard-bg-${bg.name}`);
    const single = world.gpu.root
        .createRenderPipeline({
            vertex: bgVs,
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

/** Prepare every material parameter table, pipeline and background at warm, before the first draw.
 * `singleSample` also prepares each material's AA-off variant, for a no-AA camera present at warm.
 * A camera added or switched to AA-off after warm still compiles that variant on its first draw. */
export async function preparePipelines(
    world: World,
    capacity: number,
    singleSample = false,
): Promise<void> {
    // A material may first acquire a draw in the draw group, after that frame's table upload.
    // Create its parameter table at warm so the defaults reach the first frame's upload.
    // TypeGPU defers resolve and device pipeline creation until first use. root.unwrap forces
    // both at warm so resolution/validation errors surface here, not mid-frame on the first draw.
    for (const type of world.resource(MaterialTypes).types) {
        if (!type) continue;
        const source = materialSource(world, type);
        source.table;
        const alphaKeys = source.alphaKeys;
        for (
            let alphaPipelineKey = 0;
            alphaPipelineKey < ALPHA_PIPELINE_KEY_COUNT;
            alphaPipelineKey++
        ) {
            if (!(alphaKeys & (1 << alphaPipelineKey))) continue;
            const compiled = compileMaterial(world, type, capacity, alphaPipelineKey);
            if (singleSample) ensureSingle(world, compiled);
            for (const p of [
                compiled.color,
                compiled.transparent,
                compiled.point,
                compiled.cascade,
                compiled.prepass,
                compiled.single?.color,
                compiled.single?.transparent,
                compiled.prepassSingle,
            ]) {
                if (p) world.gpu.root.unwrap(p);
            }
        }
    }
    for (const bg of world.resource(Backgrounds).values()) {
        const cb = compileBackground(world, bg);
        world.gpu.root.unwrap(cb.color);
        world.gpu.root.unwrap(cb.single);
    }
}
