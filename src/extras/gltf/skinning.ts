import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { MeshInstance, MeshPlugin } from "../../core/mesh";
import { Xform, xformNormal, xformPoint, xformQuat } from "../../core/rendering";
import { component, entity, type Plugin, type System, u32, type World } from "../../engine";
import {
    applyPbrLighting,
    extendedMaterial,
    extensionLayout,
    MaterialPlugin,
    Materials,
    MeshPreprocessSystem,
    type materialLayout,
    pbrInputFromStandardMaterial,
} from "../../standard/rendering";

import { DeformationRanges, MorphTextures, MorphWidth, WeightWidth } from "./morph";

/** Family-absolute vertex data: four packed u16 joints, morph texel/descriptor, four skin weights.
 * Morph descriptor is (targetCount << 3) | position/normal/tangent mask; zero disables morphing. */
export const VertexDeformation = d.struct({ joints: d.vec2u, morph: d.vec2u, weights: d.vec4f });
/** Bindings for deformationVertex in materialLayout and extensionLayout.
 * deformationOffsets holds (firstJoint + 1, firstWeight + 1, weightCount, 0) by eid;
 * a zero offset disables that deformation. Missing entity weights leave remaining targets unmorphed. */
export const deformationBindings = {
    deformation: { type: "attribute", element: VertexDeformation },
    jointPalette: { type: "texture", filterable: false, visibility: ["vertex"] },
    deformationOffsets: { type: "texture", sampleType: "uint", visibility: ["vertex"] },
    morphWeights: { type: "texture", filterable: false, visibility: ["vertex"] },
    morphDeltas: { type: "texture", filterable: false, visibility: ["vertex"] },
} as const;
const PaletteWidth = 256;
const JointsPerRow = PaletteWidth / 4;

/** Palette allocation owned by this entity's membership. firstJoint is derived; jointCount is
 * supplied on attachment and must remain unchanged until removal. Removing or destroying the entity
 * frees it and resets the cull override. A retained material stops skinning; attached morphing continues. */
export const SkinnedMesh = component("SkinnedMesh", { firstJoint: u32, jointCount: u32 });
/** Weight range owned by membership. firstWeight is derived; weightCount is supplied on attachment
 * and stays unchanged until removal. Removal frees the range and resets its cull override. */
export const MorphedMesh = component("MorphedMesh", { firstWeight: u32, weightCount: u32 });
/** Deformation source; the generation-checked reference becomes missing after destruction. */
export const LeaderPose = component("LeaderPose", { leader: entity });
const followers = [LeaderPose];

/** CPU staging and world-owned textures. extent is the allocated joint high-water mark, including holes. */
export const SkinPalette = {
    create: (world: World) => new Palette(world),
};
class Palette {
    enabled = false;
    readonly ranges = new DeformationRanges();
    get extent(): number {
        return this.ranges.extent;
    }
    matrices = new Float32Array(0);
    readonly morph: MorphTextures;
    private _paletteHeight = 0;
    private _offsetHeight = 0;
    private _offsets = new Uint32Array(0);
    jointPalette!: GPUTexture;
    deformationOffsets!: GPUTexture;
    readonly bindings = {
        jointPalette: undefined as unknown as GPUTexture,
        deformationOffsets: undefined as unknown as GPUTexture,
        morphWeights: undefined as unknown as GPUTexture,
        morphDeltas: undefined as unknown as GPUTexture,
    };
    private readonly _paletteTarget: GPUTexelCopyTextureInfo = {
        texture: undefined as unknown as GPUTexture,
    };
    private readonly _offsetTarget: GPUTexelCopyTextureInfo = {
        texture: undefined as unknown as GPUTexture,
    };
    private readonly _paletteLayout = { bytesPerRow: PaletteWidth * 16 };
    private readonly _offsetLayout = { bytesPerRow: PaletteWidth * 16 };
    private readonly _paletteSize = { width: PaletteWidth, height: 1 };
    private readonly _offsetSize = { width: PaletteWidth, height: 1 };
    private _world: World;
    constructor(world: World) {
        this._world = world;
        this.morph = new MorphTextures(world);
    }
    allocate(count: number): number {
        const first = this.ranges.allocate(count, JointsPerRow * 8192, "SkinnedMesh palette");
        this.growPalette(Math.max(1, Math.ceil(this.extent / JointsPerRow)));
        return first;
    }
    release(first: number, count: number): void {
        this.ranges.release(first, count);
    }
    private texture(format: GPUTextureFormat, height: number): GPUTexture {
        return this._world.gpu.device.createTexture({
            size: [PaletteWidth, height],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    }
    private growPalette(height: number): void {
        if (height <= this._paletteHeight) return;
        height = Math.min(8192, Math.max(height, this._paletteHeight * 2));
        const data = new Float32Array(height * PaletteWidth * 4);
        data.set(this.matrices);
        this.matrices = data;
        if (this.jointPalette) this._world.retireGpuResource(this.jointPalette);
        this.jointPalette = this.texture("rgba32float", height);
        this.bindings.jointPalette = this._paletteTarget.texture = this.jointPalette;
        this._paletteHeight = height;
    }
    upload(): void {
        this.growPalette(1);
        const column = this._world.fieldStorage(SkinnedMesh, "firstJoint").column;
        const weightColumn = this._world.fieldStorage(MorphedMesh, "firstWeight").column;
        const height = Math.max(
            1,
            Math.ceil(
                Math.max(
                    column.length,
                    weightColumn.length,
                    this._world.fieldStorage(LeaderPose, "leader").column.length,
                ) / PaletteWidth,
            ),
        );
        if (height > 8192)
            throw new Error(
                "glTF deformation entity map exceeds WebGPU's default texture dimensions",
            );
        if (height > this._offsetHeight) {
            this._offsetHeight = this._offsetSize.height = height;
            this._offsets = new Uint32Array(height * PaletteWidth * 4);
            if (this.deformationOffsets) this._world.retireGpuResource(this.deformationOffsets);
            this.deformationOffsets = this.texture("rgba32uint", height);
            this.bindings.deformationOffsets = this._offsetTarget.texture = this.deformationOffsets;
        }
        const counts = this._world.fieldStorage(SkinnedMesh, "jointCount").column;
        const weightCounts = this._world.fieldStorage(MorphedMesh, "weightCount").column;
        this._offsets.fill(0);
        for (let eid = 0; eid < column.length; eid++)
            this._offsets[eid * 4] = counts[eid] === 0 ? 0 : column[eid] + 1;
        for (let eid = 0; eid < weightColumn.length; eid++) {
            this._offsets[eid * 4 + 1] = weightCounts[eid] === 0 ? 0 : weightColumn[eid] + 1;
            this._offsets[eid * 4 + 2] = weightCounts[eid];
        }
        const leaders = this._world.storage(LeaderPose).leader;
        const bounds = this._world.storage(MeshInstance).cullBounds;
        for (const eid of this._world.query(followers)) {
            const leader = leaders.get(eid);
            this._offsets[eid * 4] =
                leader && this._world.has(leader, SkinnedMesh) ? column[leader] + 1 : 0;
            this._offsets[eid * 4 + 1] =
                leader && this._world.has(leader, MorphedMesh) ? weightColumn[leader] + 1 : 0;
            this._offsets[eid * 4 + 2] =
                leader && this._world.has(leader, MorphedMesh) ? weightCounts[leader] : 0;
            if (leader)
                bounds.set(
                    eid,
                    bounds.x.get(leader),
                    bounds.y.get(leader),
                    bounds.z.get(leader),
                    bounds.w.get(leader),
                );
            else bounds.set(eid, 0, 0, 0, -1);
        }
        this.morph.upload();
        this.bindings.morphWeights = this.morph.bindings.morphWeights;
        this.bindings.morphDeltas = this.morph.bindings.morphDeltas;
        this._paletteSize.height = Math.max(1, Math.ceil(this.extent / JointsPerRow));
        const queue = this._world.gpu.device.queue;
        queue.writeTexture(
            this._paletteTarget,
            this.matrices,
            this._paletteLayout,
            this._paletteSize,
        );
        queue.writeTexture(this._offsetTarget, this._offsets, this._offsetLayout, this._offsetSize);
    }
}

const DeformationOutput = d.struct({ world: d.vec4f, worldNormal: d.vec3f, worldTangent: d.vec4f });
type DeformationLayout = Pick<
    ReturnType<
        typeof materialLayout<typeof deformationBindings, d.WgslStruct<Record<never, never>>>
    >,
    "$"
>;
/** Returns a vertex function taking eid, family-absolute vidx, local position, normal, tangent and
 * placement Xform, returning world position, normal and tangent. Compose with a material declaring
 * deformationBindings and vertexTangent. Morph deltas precede skinning and placement.
 * Uses the last StandardMaterial vertex storage binding and four vertex-only sampled textures,
 * leaving the fragment texture budget unchanged. Normals use the blended matrix's inverse
 * transpose; tangents use its linear part. Tangent handedness is preserved; the renderer corrects it
 * for mirrored placement and normalizes the tangent. */
export function deformationVertex(layout: DeformationLayout) {
    const jointMatrix = tgpu.fn(
        [d.u32],
        d.mat4x4f,
    )((joint) => {
        "use gpu";
        const x = d.i32((joint % JointsPerRow) * 4);
        const y = d.i32(joint / JointsPerRow);
        return d.mat4x4f(
            std.textureLoad(layout.$.jointPalette, d.vec2i(x, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 1, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 2, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 3, y), 0),
        );
    });
    const delta = tgpu.fn(
        [d.u32],
        d.vec3f,
    )((texel) => {
        "use gpu";
        return std.textureLoad(
            layout.$.morphDeltas,
            d.vec2i(d.i32(texel % MorphWidth), d.i32(texel / MorphWidth)),
            0,
        ).xyz;
    });
    return tgpu.fn(
        [d.u32, d.u32, d.vec3f, d.vec3f, d.vec4f, Xform],
        DeformationOutput,
    )((eid, vidx, position, sourceNormal, sourceTangent, xform) => {
        "use gpu";
        const allocation = std.textureLoad(
            layout.$.deformationOffsets,
            d.vec2i(d.i32(eid % PaletteWidth), d.i32(eid / PaletteWidth)),
            0,
        );
        const influence = layout.$.deformation[vidx];
        let p = d.vec3f(position);
        let n = d.vec3f(sourceNormal);
        let t = d.vec3f(sourceTangent.xyz);
        const mask = influence.morph.y & 7;
        const count = std.min(influence.morph.y >>> 3, allocation.z);
        const stride = (mask & 1) + ((mask >>> 1) & 1) + ((mask >>> 2) & 1);
        if (allocation.y !== 0) {
            for (let target = d.u32(0); target < count; target++) {
                const wi = allocation.y - 1 + target;
                const weight = std.textureLoad(
                    layout.$.morphWeights,
                    d.vec2i(d.i32(wi % WeightWidth), d.i32(wi / WeightWidth)),
                    0,
                ).x;
                if (weight === 0) continue;
                let texel = influence.morph.x + target * stride;
                if ((mask & 1) !== 0) {
                    p = std.add(p, std.mul(weight, delta(texel)));
                    texel++;
                }
                if ((mask & 2) !== 0) {
                    n = std.add(n, std.mul(weight, delta(texel)));
                    texel++;
                }
                if ((mask & 4) !== 0) t = std.add(t, std.mul(weight, delta(texel)));
            }
        }
        if (allocation.x === 0)
            return {
                world: d.vec4f(xformPoint(xform, p), 1),
                worldNormal: xformNormal(xform, std.normalize(n)),
                worldTangent: d.vec4f(
                    xformQuat(xform.quat, std.mul(t, xform.scale)),
                    sourceTangent.w,
                ),
            };
        const first = allocation.x - 1;
        const j = influence.joints;
        const w = influence.weights;
        const m = std.add(
            std.add(
                std.mul(w.x, jointMatrix(first + (j.x & 65535))),
                std.mul(w.y, jointMatrix(first + (j.x >>> 16))),
            ),
            std.add(
                std.mul(w.z, jointMatrix(first + (j.y & 65535))),
                std.mul(w.w, jointMatrix(first + (j.y >>> 16))),
            ),
        );
        const c0 = m.columns[0].xyz;
        const c1 = m.columns[1].xyz;
        const c2 = m.columns[2].xyz;
        const x = std.cross(c1, c2);
        const y = std.cross(c2, c0);
        const z = std.cross(c0, c1);
        const det = std.dot(c2, z);
        const normal = std.normalize(
            std.add(std.add(std.mul(n.x / det, x), std.mul(n.y / det, y)), std.mul(n.z / det, z)),
        );
        return {
            world: d.vec4f(xformPoint(xform, std.mul(m, d.vec4f(p, 1)).xyz), 1),
            worldNormal: xformNormal(xform, normal),
            worldTangent: d.vec4f(
                xformQuat(xform.quat, std.mul(std.mul(m, d.vec4f(t, 0)).xyz, xform.scale)),
                sourceTangent.w,
            ),
        };
    });
}
const layout = extensionLayout(Materials, d.struct({ skinPad: d.u32 }), deformationBindings);
const vertex = deformationVertex(layout);
/** StandardMaterial with morphing followed by linear blend skinning. A replacement material draws
 * bind pose unless it declares deformationBindings, calls deformationVertex and opts into
 * vertexTangent. This uses the last vertex storage buffer and four vertex-only sampled textures,
 * leaving the fragment texture budget unchanged. */
export const SkinnedMaterial = extendedMaterial(Materials, {
    name: "SkinnedMaterial",
    vertexTangent: true,
    layout,
    defaults: { skinPad: 0 },
    bindings: (world) => world.resource(SkinPalette).bindings,
    vertex: (input) => {
        "use gpu";
        const posed = vertex(
            input.eid,
            input.vidx,
            input.localPos,
            input.localNormal,
            input.localTangent,
            input.xform,
        );
        return {
            world: posed.world,
            worldNormal: posed.worldNormal,
            worldTangent: posed.worldTangent,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
    },
});
const UploadDeformationsSystem: System = {
    name: "upload-gltf-deformations",
    group: "draw",
    before: [MeshPreprocessSystem],
    update(world) {
        world.resource(SkinPalette).upload();
    },
};
/** glTF morphing and skinning, with entity-owned weight and palette ranges. Replacement materials
 * require deformationBindings and deformationVertex to deform (one storage buffer, four textures). */
export const GltfPlugin: Plugin = {
    name: "Gltf",
    gpu: {},
    components: [SkinnedMesh, MorphedMesh, LeaderPose],
    dependencies: [MeshPlugin, MaterialPlugin(SkinnedMaterial)],
    systems: [UploadDeformationsSystem],
    initialize(world) {
        const palette = world.resource(SkinPalette);
        palette.enabled = true;
        world.onDispose(
            world.observeMembership(SkinnedMesh, (eid, present) => {
                const storage = world.storage(SkinnedMesh);
                const count = storage.jointCount.get(eid);
                if (present) storage.firstJoint.set(eid, palette.allocate(count));
                else {
                    palette.release(storage.firstJoint.get(eid), count);
                    if (world.has(eid, MeshInstance))
                        world.storage(MeshInstance).cullBounds.set(eid, 0, 0, 0, -1);
                }
            }),
        );
        world.onDispose(
            world.observeMembership(MorphedMesh, (eid, present) => {
                const storage = world.storage(MorphedMesh);
                const count = storage.weightCount.get(eid);
                if (present) storage.firstWeight.set(eid, palette.morph.allocate(count));
                else {
                    palette.morph.ranges.release(storage.firstWeight.get(eid), count);
                    if (world.has(eid, MeshInstance))
                        world.storage(MeshInstance).cullBounds.set(eid, 0, 0, 0, -1);
                }
            }),
        );
        palette.upload();
    },
};
